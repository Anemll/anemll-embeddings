#!/usr/bin/env python3
"""Unit checks for T4 export helpers (no checkpoint, no coremltools)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.export_utils import (  # noqa: E402
    MASK_NEG,
    additive_full_attention_bias,
    additive_sliding_attention_bias,
    example_trace_inputs,
    package_stem,
    pad_to_seq_len,
)
from src.trace_patches import _rotate_half_chunk  # noqa: E402


def _fail(msg: str) -> None:
    raise AssertionError(msg)


def test_pad_to_seq_len() -> None:
    ids = torch.tensor([[3, 4, 5]], dtype=torch.int32)
    mask = torch.tensor([[1, 1, 1]], dtype=torch.int32)
    pids, pmask = pad_to_seq_len(ids, mask, 6, pad_token_id=0)
    if tuple(pids.shape) != (1, 6) or tuple(pmask.shape) != (1, 6):
        _fail(f"pad shape {tuple(pids.shape)}")
    if pids[0, 3:].tolist() != [0, 0, 0]:
        _fail(f"pad ids {pids.tolist()}")
    if pmask[0].tolist() != [1, 1, 1, 0, 0, 0]:
        _fail(f"pad mask {pmask.tolist()}")
    tids, tmask = pad_to_seq_len(pids, pmask, 4)
    if tids.shape[-1] != 4 or tmask[0].tolist() != [1, 1, 1, 0]:
        _fail("truncate failed")


def test_example_inputs_have_padding() -> None:
    ids, mask = example_trace_inputs(16, pad_last=4)
    if tuple(ids.shape) != (1, 16):
        _fail(f"example shape {tuple(ids.shape)}")
    if int(mask[:, -4:].sum()) != 0:
        _fail("expected trailing pads")
    if int(mask[:, :-4].sum()) != 12:
        _fail("expected leading valid tokens")


def test_full_bias_keys_only() -> None:
    mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.int32)
    bias = additive_full_attention_bias(mask, torch.float32)
    if tuple(bias.shape) != (1, 1, 4, 4):
        _fail(f"bias shape {tuple(bias.shape)}")
    # valid keys (0,1) stay 0; pad keys (2,3) are MASK_NEG for every query
    if not torch.equal(bias[0, 0, :, :2], torch.zeros(4, 2)):
        _fail("valid keys should be 0")
    if not torch.allclose(bias[0, 0, :, 2:], torch.full((4, 2), MASK_NEG)):
        _fail("pad keys should be MASK_NEG")


def test_sliding_window_inclusive() -> None:
    # S=5, window=1 → each token sees itself ±1, plus pad on last key
    mask = torch.ones(1, 5, dtype=torch.int32)
    bias = additive_sliding_attention_bias(mask, sliding_window=1, dtype=torch.float32)
    keep = bias[0, 0] == 0
    # row q=2 should keep keys 1,2,3
    if keep[2].tolist() != [False, True, True, True, False]:
        _fail(f"window row 2 {keep[2].tolist()}")
    mask[:, -1] = 0
    bias_p = additive_sliding_attention_bias(mask, sliding_window=1, dtype=torch.float32)
    if float(bias_p[0, 0, 3, 4]) != MASK_NEG:
        _fail("pad key must stay masked even if in window")


def test_package_stem() -> None:
    if package_stem(512) != "embeddinggemma2-text-s512":
        _fail(package_stem(512))


def test_rotate_half_chunk() -> None:
    x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
    got = _rotate_half_chunk(x)
    # Original rotate_half: cat(-x2, x1) on the last dim split in half.
    expected = torch.tensor([[[[-3.0, -4.0, 1.0, 2.0]]]])
    if not torch.equal(got, expected):
        _fail(f"rotate_half {got}")


def main() -> int:
    tests = [
        test_pad_to_seq_len,
        test_example_inputs_have_padding,
        test_full_bias_keys_only,
        test_sliding_window_inclusive,
        test_package_stem,
        test_rotate_half_chunk,
    ]
    for fn in tests:
        fn()
        print(f"OK {fn.__name__}")
    print(f"OK: {len(tests)} export-utils checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
