#!/usr/bin/env python3
"""Text-only parity of one or more ``text_embeds_s320`` packages (no vision/audio towers).

For each ``--artifacts LABEL=DIR`` (a dir holding ``coreai/*.aimodel``) the
12 fixture prompts of ``tests/fixtures/prompts.json`` run through the Core AI
host path (``api.embedder.CoreAIBackend``: tokenizer, embed lookup, f16 pad to
S=320, tower, L2). Reported cosines, per prompt and min:

* vs ``tests/fixtures/embeddings.npy`` (Sentence-Transformers reference, bf16),
* vs a fresh FP32 CPU run of the full checkpoint (``--model-full``),
* between every pair of packages.

Run in the host venv. Set ``CFFIXED_USER_HOME`` for a private Core AI cache.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from api.embedder import CoreAIBackend, default_coreai_python  # noqa: E402

PROMPTS = REPO_ROOT / "tests" / "fixtures" / "prompts.json"
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "embeddings.npy"


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    return v / np.linalg.norm(v)


def run_package(artifacts: Path, host: Path, prompts: list[dict]) -> np.ndarray:
    backend = CoreAIBackend(
        artifacts=artifacts, model_path=host, coreai_python=default_coreai_python()
    )
    backend.warmup()
    print(f"  placement={backend.placement} towers={list(backend.health().get('towers', []))[:3]}")
    out = []
    for p in prompts:
        prefix = backend._prompts.get(p["prompt_name"]) if p.get("prompt_name") else ""
        ids, mask = backend._tokenize((prefix or "") + p["text"])
        embeds = backend._scatter(ids, image_soft=None, audio_soft=None)
        vec, _ = backend._run_text(embeds, mask)
        out.append(_unit(vec))
    backend.close()
    return np.stack(out)


def fp32_reference(model_full: Path, prompts: list[dict]) -> np.ndarray:
    from model.embed_wrapper import EmbeddingGemma2Wrapper, tokenize_with_st_prompt
    from model.load_text_model import load_sentence_transformer

    st, _ = load_sentence_transformer(model_full, dtype=torch.float32, device="cpu")
    wrapper = EmbeddingGemma2Wrapper.from_sentence_transformer(st, normalize=True).eval()
    out = []
    for p in prompts:
        batch = tokenize_with_st_prompt(st, p["text"], p.get("prompt_name"))
        with torch.no_grad():
            out.append(_unit(wrapper(batch["input_ids"], batch["attention_mask"]).numpy()))
    return np.stack(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", action="append", required=True, metavar="LABEL=DIR")
    ap.add_argument("--host", type=Path, required=True, help="slim host dir (tokenizer, embed table)")
    ap.add_argument("--model-full", type=Path, default=None, help="full checkpoint for the FP32 CPU reference")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    prompts = json.loads(PROMPTS.read_text())["prompts"]
    ids = [p["id"] for p in prompts]
    refs: dict[str, np.ndarray] = {"st_fixture": np.stack([_unit(v) for v in np.load(FIXTURE)])}
    if args.model_full:
        print("FP32 CPU reference …")
        refs["fp32_cpu"] = fp32_reference(args.model_full, prompts)
    pkgs: dict[str, np.ndarray] = {}
    for item in args.artifacts:
        label, _, path = item.partition("=")
        print(f"package {label}: {path}")
        pkgs[label] = run_package(Path(path), args.host, prompts)

    report: dict = {"prompts": ids, "cosine": {}}

    def cos(a, b):
        return [float(np.dot(x, y)) for x, y in zip(a, b)]

    for label, vecs in pkgs.items():
        for rname, rvecs in refs.items():
            c = cos(vecs, rvecs)
            report["cosine"][f"{label} vs {rname}"] = c
    for (la, a), (lb, b) in combinations(pkgs.items(), 2):
        report["cosine"][f"{la} vs {lb}"] = cos(a, b)
    if "fp32_cpu" in refs:
        report["cosine"]["st_fixture vs fp32_cpu"] = cos(refs["st_fixture"], refs["fp32_cpu"])
    print(f"{'comparison':<42} {'min':>9} {'mean':>9}")
    for k, c in report["cosine"].items():
        print(f"{k:<42} {min(c):9.6f} {np.mean(c):9.6f}")
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
