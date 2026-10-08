#!/usr/bin/env python3
"""Smoke-check committed ST reference fixtures (T1).

Reloads tests/fixtures/embeddings.npy + reference_meta.json and asserts:
  - shape (N, 768) matching prompt count
  - float32 finite values (no NaN/Inf)
  - L2 norms ~ 1 (normalized embeddings)
  - per-vector SHA-256 digests match metadata
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

FIXTURES = Path(__file__).resolve().parent / "fixtures"
PROMPTS = FIXTURES / "prompts.json"
NPY = FIXTURES / "embeddings.npy"
META = FIXTURES / "reference_meta.json"
DIM = 768
NORM_TOL = 5e-3  # BF16 MPS norms can drift ~1e-3


def _digest(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def main() -> int:
    errors: list[str] = []

    for path in (PROMPTS, NPY, META):
        if not path.is_file():
            errors.append(f"missing {path}")
    if errors:
        print("FAIL:", "; ".join(errors))
        return 1

    with PROMPTS.open() as f:
        prompt_pack = json.load(f)
    with META.open() as f:
        meta = json.load(f)

    prompts = prompt_pack["prompts"]
    emb = np.load(NPY)

    n = len(prompts)
    if emb.shape != (n, DIM):
        errors.append(f"shape {emb.shape} != ({n}, {DIM})")
    if emb.dtype != np.float32:
        errors.append(f"dtype {emb.dtype} != float32")
    if not np.isfinite(emb).all():
        bad = int((~np.isfinite(emb)).sum())
        errors.append(f"{bad} non-finite values")

    norms = np.linalg.norm(emb, axis=1)
    if meta.get("normalize_embeddings", True):
        for i, nrm in enumerate(norms):
            if abs(float(nrm) - 1.0) > NORM_TOL:
                errors.append(f"norm[{i}]={nrm:.6f} not ~1")

    if meta.get("prompt_ids") != [p["id"] for p in prompts]:
        errors.append("meta.prompt_ids != prompts.json ids")

    digests = meta.get("vector_sha256") or {}
    for i, p in enumerate(prompts):
        got = _digest(emb[i])
        expected = digests.get(p["id"])
        if expected and got != expected:
            errors.append(f"digest mismatch for {p['id']}")

    if list(meta.get("embedding_shape", [])) != list(emb.shape):
        errors.append("meta.embedding_shape mismatch")

    if errors:
        print("FAIL:")
        for e in errors:
            print(" -", e)
        return 1

    print(
        f"OK: {emb.shape[0]} vectors x {emb.shape[1]}d, "
        f"norms in [{norms.min():.6f}, {norms.max():.6f}], no NaNs"
    )
    return 0


def test_reference_fixtures() -> None:
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
