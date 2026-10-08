#!/usr/bin/env python3
"""Smoke-test T3 wrapper embeddings vs committed T1 ST fixtures (cosine).

Loads the text-only model, builds EmbeddingGemma2Wrapper (mask mean-pool +
512→768 + L2), encodes tests/fixtures/prompts.json, and reports per-row
cosine vs tests/fixtures/embeddings.npy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from model.embed_wrapper import (  # noqa: E402
    EmbeddingGemma2Wrapper,
    tokenize_with_st_prompt,
)
from model.load_text_model import load_sentence_transformer  # noqa: E402

DEFAULT_PROMPTS = REPO_ROOT / "tests" / "fixtures" / "prompts.json"
DEFAULT_NPY = REPO_ROOT / "tests" / "fixtures" / "embeddings.npy"
COSINE_GATE = 0.999


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--reference", type=Path, default=DEFAULT_NPY)
    parser.add_argument("--cosine-gate", type=float, default=COSINE_GATE)
    parser.add_argument(
        "--device",
        default=None,
        help="Override device (default: auto from loader)",
    )
    args = parser.parse_args()

    artifacts = os.environ.get("ANEMLL_EMBEDDINGS_ARTIFACTS")
    if artifacts:
        print(f"ANEMLL_EMBEDDINGS_ARTIFACTS={artifacts}")

    with args.prompts.open() as f:
        prompt_pack = json.load(f)
    prompts = prompt_pack["prompts"]
    ref = np.load(args.reference)
    if ref.shape != (len(prompts), 768):
        print(f"ERROR: reference shape {ref.shape} != ({len(prompts)}, 768)")
        return 1

    print("Loading text-only ST model …")
    st_model, meta = load_sentence_transformer(
        device=args.device if args.device else None,
    )
    print(f"  dtype={meta['torch_dtype']} device={meta['device']}")

    wrapper = EmbeddingGemma2Wrapper.from_sentence_transformer(
        st_model, normalize=True
    )
    wrapper.eval()
    device = meta["device"]
    # Keep wrapper params on the same device as the backbone.
    wrapper.to(device)

    vectors = []
    for p in prompts:
        batch = tokenize_with_st_prompt(
            st_model,
            p["text"],
            p.get("prompt_name"),
            device=device,
        )
        with torch.inference_mode():
            emb = wrapper(batch["input_ids"], batch["attention_mask"])
        vectors.append(emb.detach().float().cpu().numpy().reshape(-1))

    got = np.stack(vectors, axis=0).astype(np.float32)
    if not np.isfinite(got).all():
        print("ERROR: NaN/Inf in wrapper embeddings")
        return 1

    cosines = [_cosine(got[i], ref[i]) for i in range(len(prompts))]
    print("id                 cosine")
    for p, c in zip(prompts, cosines):
        print(f"{p['id']:<18} {c:.8f}")

    min_c = float(np.min(cosines))
    mean_c = float(np.mean(cosines))
    print(f"min={min_c:.8f} mean={mean_c:.8f} gate={args.cosine_gate}")

    if min_c < args.cosine_gate:
        print("FAIL: cosine below gate")
        return 1
    print("OK: wrapper matches T1 fixtures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
