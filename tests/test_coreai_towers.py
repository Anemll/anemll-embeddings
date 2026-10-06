#!/usr/bin/env python3
"""Unit checks for Core AI tower I/O specs (no checkpoint)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.coreai_towers import (  # noqa: E402
    AUDIO_FEAT,
    AUDIO_FRAMES,
    VISION_PATCH_DIM,
    VISION_PATCHES,
    VISION_SOFT_TOKENS,
    audio_example,
    tower_io_spec,
    vision_example,
)


def _fail(msg: str) -> None:
    raise AssertionError(msg)


def test_vision_example_shape() -> None:
    pixels, pos = vision_example()
    if tuple(pixels.shape) != (1, VISION_PATCHES, VISION_PATCH_DIM):
        _fail(str(pixels.shape))
    if tuple(pos.shape) != (1, VISION_PATCHES, 2):
        _fail(str(pos.shape))
    if pos.min() < 0:
        _fail("example grid should have no pad (-1)")


def test_audio_example_shape() -> None:
    feat, mask = audio_example()
    if tuple(feat.shape) != (1, AUDIO_FRAMES, AUDIO_FEAT):
        _fail(str(feat.shape))
    if tuple(mask.shape) != (1, AUDIO_FRAMES) or mask.dtype != torch.bool:
        _fail(str(mask.shape))
    if int(mask.sum()) != AUDIO_FRAMES:
        _fail("expected full-valid mask")


def test_io_specs() -> None:
    v = tower_io_spec("vision")
    if v["outputs"]["soft_tokens"] != [1, VISION_SOFT_TOKENS, 512]:
        _fail(str(v))
    if tower_io_spec("text")["outputs"]["embedding"] != [1, 768]:
        _fail("text embed")


def main() -> int:
    tests = [test_vision_example_shape, test_audio_example_shape, test_io_specs]
    for fn in tests:
        fn()
        print(f"OK {fn.__name__}")
    print(f"OK: {len(tests)} coreai-tower checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
