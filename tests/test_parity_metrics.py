#!/usr/bin/env python3
"""Unit checks for T6 parity metrics (no checkpoint, no Core ML)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.parity_metrics import (  # noqa: E402
    cosine,
    finite_counts,
    l2_norm,
    pair_deltas,
    pairwise_cosines,
    rel_l2,
)


def _fail(msg: str) -> None:
    raise AssertionError(msg)


def test_cosine_identical() -> None:
    x = np.array([0.6, 0.8], dtype=np.float32)
    c = cosine(x, x)
    if abs(c - 1.0) > 1e-12:
        _fail(f"identical cosine {c}")


def test_rel_l2_zero_and_known() -> None:
    ref = np.array([3.0, 4.0], dtype=np.float64)
    if rel_l2(ref, ref) != 0.0:
        _fail("rel_l2 identical")
    pred = np.array([3.0, 7.0], dtype=np.float64)
    got = rel_l2(pred, ref)
    # ||(0,3)|| / 5 = 0.6
    if abs(got - 0.6) > 1e-12:
        _fail(f"rel_l2 {got}")


def test_l2_norm() -> None:
    if abs(l2_norm(np.array([3.0, 4.0])) - 5.0) > 1e-12:
        _fail("l2_norm")


def test_finite_counts() -> None:
    x = np.array([1.0, np.nan, np.inf, -np.inf], dtype=np.float64)
    c = finite_counts(x)
    if c != {"n": 4, "finite": 1, "nan": 1, "inf": 2}:
        _fail(f"finite_counts {c}")


def test_pairwise_and_delta() -> None:
    ids = ["sq_short", "doc_short", "other"]
    # 90 degrees vs aligned
    vecs = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=np.float64,
    )
    pairs = pairwise_cosines(vecs, ids, pairs=(("sq_short", "doc_short"),))
    if abs(pairs[0]["cosine"] - 0.0) > 1e-12:
        _fail(f"pair cosine {pairs[0]}")
    st = pairwise_cosines(vecs, ids, pairs=(("sq_short", "doc_short"),))
    cml_vecs = vecs.copy()
    cml_vecs[1] = np.array([1.0, 0.0])
    cml = pairwise_cosines(cml_vecs, ids, pairs=(("sq_short", "doc_short"),))
    deltas = pair_deltas(st, cml)
    if not math.isclose(deltas[0]["abs_delta"], 1.0, abs_tol=1e-12):
        _fail(f"pair delta {deltas[0]}")


def main() -> int:
    tests = [
        test_cosine_identical,
        test_rel_l2_zero_and_known,
        test_l2_norm,
        test_finite_counts,
        test_pairwise_and_delta,
    ]
    for fn in tests:
        fn()
        print(f"OK {fn.__name__}")
    print(f"OK: {len(tests)} parity-metrics checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
