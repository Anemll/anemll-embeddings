#!/usr/bin/env python3
"""Generate ST reference embeddings for full 740M multimodal EmbeddingGemma 2.

Writes synthetic media + embeddings under ANEMLL_EMBEDDINGS_ARTIFACTS/fixtures/
(not TB36, not git). Does not overwrite the FLOAT32 text Core ML tree.

Environment matches T1: HF caches on TB36, artifacts on /Volumes/Models.
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
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from model.export_utils import artifacts_root, write_json  # noqa: E402
from model.load_multimodal_model import load_multimodal_sentence_transformer  # noqa: E402
from model.load_text_model import DEFAULT_MODEL, ensure_hf_cache_env  # noqa: E402
from model.multimodal_media import AUDIO_SR, write_default_media  # noqa: E402

DEFAULT_PROMPTS = REPO_ROOT / "tests" / "fixtures" / "multimodal_prompts.json"


def _vector_digest(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _wav_f32(path: Path) -> np.ndarray:
    """Mono int16 wav → float32 samples (avoids transformers' librosa path load)."""
    import wave

    with wave.open(str(path), "r") as w:
        raw = w.readframes(w.getnframes())
        sr = int(w.getframerate())
    if sr != AUDIO_SR:
        raise ValueError(f"{path} sr={sr} != {AUDIO_SR}")
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def _encode_one(model, item: dict, media_dir: Path) -> np.ndarray:
    kind = item.get("kind") or "text"
    text = item.get("text") or ""
    prompt_name = item.get("prompt_name")
    if kind == "text":
        vec = model.encode(text, prompt_name=prompt_name)
        return np.asarray(vec, dtype=np.float32).reshape(-1)

    payload: dict = {"text": text}
    if item.get("image"):
        payload["image"] = [str(media_dir / name) for name in item["image"]]
    if item.get("audio"):
        payload["audio"] = _wav_f32(media_dir / item["audio"])
    if item.get("video_frames"):
        payload["video"] = [Image.open(media_dir / name) for name in item["video_frames"]]
    kwargs = {}
    if prompt_name:
        kwargs["prompt_name"] = prompt_name
    vec = model.encode(payload, **kwargs)
    return np.asarray(vec, dtype=np.float32).reshape(-1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path(os.environ.get("ANEMLL_EMBEDDINGS_MODEL", DEFAULT_MODEL)))
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--artifacts", type=Path, default=artifacts_root())
    args = parser.parse_args()

    ensure_hf_cache_env()
    with args.prompts.open() as f:
        pack = json.load(f)
    prompts = pack["prompts"]
    media_dir = Path(args.artifacts) / pack.get("media_subdir", "fixtures/media")
    out_dir = Path(args.artifacts) / "fixtures"
    write_default_media(media_dir)
    print(f"media={media_dir}")

    print("Loading full multimodal SentenceTransformer (vision+audio) …")
    model, load_meta = load_multimodal_sentence_transformer(args.model)
    print(f"load={load_meta}")

    rows = []
    vecs = []
    for item in prompts:
        print(f"encode {item['id']} kind={item.get('kind')}")
        vec = _encode_one(model, item, media_dir)
        if vec.shape != (768,):
            print(f"ERROR: {item['id']} shape {vec.shape} != (768,)")
            return 1
        if not np.isfinite(vec).all():
            print(f"ERROR: {item['id']} non-finite")
            return 1
        norm = float(np.linalg.norm(vec))
        print(f"  l2={norm:.6f}")
        vecs.append(vec)
        rows.append(
            {
                "id": item["id"],
                "kind": item.get("kind"),
                "l2": norm,
                "sha256": _vector_digest(vec),
            }
        )

    emb = np.stack(vecs, axis=0).astype(np.float32)
    npy_path = out_dir / pack.get("embeddings_name", "multimodal_embeddings.npy")
    meta_path = out_dir / pack.get("meta_name", "multimodal_reference_meta.json")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, emb)
    write_json(
        meta_path,
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "host": platform.node(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "prompts": str(args.prompts),
            "npy": str(npy_path),
            "media_dir": str(media_dir),
            "normalize_embeddings": True,
            "shape": list(emb.shape),
            "load": load_meta,
            "rows": rows,
            "notes": [
                "Full 740M ST reference (vision+audio enabled). Not Core ML.",
                "Primary convert target is Core AI (.aimodel), not Core ML.",
                "Synthetic media only; FLOAT32 text .mlpackage tree untouched.",
            ],
        },
    )
    print(f"wrote {npy_path} shape={emb.shape}")
    print(f"wrote {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
