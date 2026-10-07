#!/usr/bin/env python3
"""Unit checks for Core AI tower I/O specs (no checkpoint)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.audio_export_patches import gather_seq_windows  # noqa: E402
from src.coreai_towers import (  # noqa: E402
    AUDIO_FEAT,
    AUDIO_FRAMES,
    AUDIO_SOFT_TOKENS,
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
    if pos.dtype != torch.int16:
        _fail(f"pos dtype {pos.dtype} (want int16)")
    if pos.min() < 0:
        _fail("example grid should have no pad (-1)")


def test_audio_example_shape() -> None:
    feat, mask = audio_example()
    if tuple(feat.shape) != (1, AUDIO_FRAMES, AUDIO_FEAT):
        _fail(str(feat.shape))
    if tuple(mask.shape) != (1, AUDIO_FRAMES) or mask.dtype != torch.int16:
        _fail(f"{tuple(mask.shape)} {mask.dtype}")
    if int(mask.sum()) != AUDIO_FRAMES:
        _fail("expected full-valid mask")


def test_gather_seq_windows_matches_unfold() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 70, 8, 16)
    window, step = 24, 12
    ref = torch.movedim(x.unfold(1, window, step), -1, 2)
    got = gather_seq_windows(x, window, step)
    if got.shape != ref.shape:
        _fail(f"shape {tuple(got.shape)} != {tuple(ref.shape)}")
    if not torch.equal(got, ref):
        _fail("gather_seq_windows != unfold+movedim")


def test_gather_seq_windows_export_has_no_unfold() -> None:
    class _Win(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return gather_seq_windows(x, 24, 12)

    ep = torch.export.export(_Win(), (torch.randn(1, 70, 4, 8),), strict=False)
    targets = [str(n.target) for n in ep.graph.nodes]
    if any("unfold" in t for t in targets):
        _fail(f"export still contains unfold: {targets}")


def test_io_specs() -> None:
    v = tower_io_spec("vision")
    if v["outputs"]["soft_tokens"] != [1, VISION_SOFT_TOKENS, 512]:
        _fail(str(v))
    if tower_io_spec("audio")["outputs"]["soft_tokens"] != [1, AUDIO_SOFT_TOKENS, 512]:
        _fail("audio tokens")
    if tower_io_spec("text")["outputs"]["embedding"] != [1, 768]:
        _fail("text embed")


def main() -> int:
    tests = [
        test_vision_example_shape,
        test_audio_example_shape,
        test_gather_seq_windows_matches_unfold,
        test_gather_seq_windows_export_has_no_unfold,
        test_io_specs,
    ]
    for fn in tests:
        fn()
        print(f"OK {fn.__name__}")
    print(f"OK: {len(tests)} coreai-tower checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
