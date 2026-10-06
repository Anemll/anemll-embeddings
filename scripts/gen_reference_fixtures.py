#!/usr/bin/env python3
"""Generate Sentence-Transformers reference embeddings for EmbeddingGemma 2 (T1).

Loads the local text-only checkpoint (vision/audio configs disabled), encodes the
fixed prompt set in tests/fixtures/prompts.json, and writes small numpy + JSON
fixtures under tests/fixtures/ for Core ML parity checks.

Environment:
  HF_HOME / HUGGINGFACE_HUB_CACHE  — keep under TB36 hf-cache
  ANEMLL_EMBEDDINGS_MODEL          — override model dir (default: TB36 checkpoint)
  ANEMLL_EMBEDDINGS_VENV note      — runtime venv should be local (not SMB); see README
  ANEMLL_EMBEDDINGS_ARTIFACTS      — placeholder for future .mlpackage/.mlmodelc disk
                                     (do NOT put compile artifacts on TB36/SAN512/internal)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path(
    "/Volumes/TB36/Models/anemll-embeddings/google-embeddinggemma-2"
)
DEFAULT_PROMPTS = REPO_ROOT / "tests" / "fixtures" / "prompts.json"
DEFAULT_OUT_DIR = REPO_ROOT / "tests" / "fixtures"


def _resolve_dtype_device() -> tuple[torch.dtype, str]:
    """BF16 when available, else FP32. Never FP16."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16, "cuda"
    if torch.backends.mps.is_available():
        # Apple Silicon: prefer BF16 on MPS when usable; fall back to FP32.
        try:
            t = torch.tensor([1.0], dtype=torch.bfloat16, device="mps")
            _ = (t * 2).item()
            return torch.bfloat16, "mps"
        except Exception:
            return torch.float32, "mps"
    return torch.float32, "cpu"


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _vector_digest(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def load_model(model_path: Path) -> tuple[SentenceTransformer, dict]:
    dtype, device = _resolve_dtype_device()
    if dtype == torch.float16:
        raise RuntimeError("FP16 is forbidden for EmbeddingGemma 2 reference loads")

    model = SentenceTransformer(
        str(model_path),
        trust_remote_code=True,
        device=device,
        config_kwargs={"vision_config": None, "audio_config": None},
        model_kwargs={"torch_dtype": dtype},
    )
    meta = {
        "model_path": str(model_path.resolve()),
        "torch_dtype": str(dtype).replace("torch.", ""),
        "device": device,
        "text_only": True,
        "config_kwargs": {"vision_config": None, "audio_config": None},
    }
    return model, meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(os.environ.get("ANEMLL_EMBEDDINGS_MODEL", DEFAULT_MODEL)),
        help="Local EmbeddingGemma 2 checkpoint directory",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=DEFAULT_PROMPTS,
        help="JSON prompt fixture path",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for embeddings.npy + reference_meta.json",
    )
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match ST Normalize module (default: true)",
    )
    args = parser.parse_args()

    # Refuse writing large artifacts into the wrong places if someone mis-sets out-dir
    artifacts = os.environ.get("ANEMLL_EMBEDDINGS_ARTIFACTS")
    # T1 only writes small fixtures into the git repo; artifacts env is for later tickets.

    if not args.model.is_dir():
        print(f"ERROR: model path missing: {args.model}", file=sys.stderr)
        return 1
    if not args.prompts.is_file():
        print(f"ERROR: prompts file missing: {args.prompts}", file=sys.stderr)
        return 1

    # Keep HF caches on TB36 when unset
    os.environ.setdefault(
        "HF_HOME", "/Volumes/TB36/Models/anemll-embeddings/hf-cache"
    )
    os.environ.setdefault(
        "HUGGINGFACE_HUB_CACHE",
        "/Volumes/TB36/Models/anemll-embeddings/hf-cache",
    )

    with args.prompts.open() as f:
        prompt_pack = json.load(f)
    prompts = prompt_pack["prompts"]

    print(f"Loading text-only model from {args.model} …")
    model, load_meta = load_model(args.model)
    print(
        f"  dtype={load_meta['torch_dtype']} device={load_meta['device']} "
        f"st={__import__('sentence_transformers').__version__}"
    )

    ids = []
    prompt_names = []
    texts = []
    for p in prompts:
        ids.append(p["id"])
        prompt_names.append(p.get("prompt_name"))
        texts.append(p["text"])

    # Encode one-by-one so prompt_name can differ per row (ST batch API is uniform).
    vectors = []
    for text, pname in zip(texts, prompt_names):
        kwargs = {"normalize_embeddings": args.normalize}
        if pname:
            kwargs["prompt_name"] = pname
        emb = model.encode(text, **kwargs)
        vectors.append(np.asarray(emb, dtype=np.float32))

    embeddings = np.stack(vectors, axis=0)
    if embeddings.ndim != 2 or embeddings.shape[1] != 768:
        print(f"ERROR: unexpected shape {embeddings.shape}", file=sys.stderr)
        return 1
    if not np.isfinite(embeddings).all():
        print("ERROR: NaN/Inf in embeddings", file=sys.stderr)
        return 1

    norms = np.linalg.norm(embeddings, axis=1)
    digests = {i: _vector_digest(embeddings[n]) for n, i in enumerate(ids)}

    weights_sha = _sha256_file(args.model / "model.safetensors")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    npy_path = args.out_dir / "embeddings.npy"
    meta_path = args.out_dir / "reference_meta.json"

    np.save(npy_path, embeddings)

    meta = {
        "version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "packages": {
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "sentence_transformers": __import__("sentence_transformers").__version__,
            "numpy": np.__version__,
        },
        "load": load_meta,
        "normalize_embeddings": args.normalize,
        "prompts_file": str(args.prompts.relative_to(REPO_ROOT)),
        "prompt_ids": ids,
        "prompt_names": prompt_names,
        "embedding_shape": list(embeddings.shape),
        "embedding_dtype": "float32",
        "l2_norms": [float(x) for x in norms],
        "vector_sha256": digests,
        "model_safetensors_sha256": weights_sha,
        "anemll_embeddings_artifacts": artifacts,
        "notes": [
            "Text-only load via config_kwargs vision_config/audio_config=None.",
            "dtype policy: BF16 if available else FP32; never FP16.",
            "Compile/Core ML outputs belong under ANEMLL_EMBEDDINGS_ARTIFACTS (TBD disk).",
        ],
    }
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")

    print(f"Wrote {npy_path} shape={embeddings.shape}")
    print(f"Wrote {meta_path}")
    print("L2 norms:", ", ".join(f"{n:.6f}" for n in norms))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
