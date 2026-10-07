#!/usr/bin/env python3
"""Unit checks for Core AI tower I/O specs (no checkpoint)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.audio_export_patches import (  # noqa: E402
    AUDIO_CHUNK,
    AUDIO_FUTURE,
    AUDIO_PAST,
    blocked_additive_attention_mask,
    gather_seq_windows,
    slice_seq_windows,
)
from src.vision_export_patches import embedding_from_int_indices  # noqa: E402
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
    if pos.dtype != torch.float16:
        _fail(f"pos dtype {pos.dtype} (want float16)")
    if pos.min() < 0:
        _fail("example grid should have no pad (-1)")


def test_audio_example_shape() -> None:
    feat, mask = audio_example()
    if tuple(feat.shape) != (1, AUDIO_FRAMES, AUDIO_FEAT):
        _fail(str(feat.shape))
    if tuple(mask.shape) != (1, AUDIO_FRAMES) or mask.dtype != torch.float16:
        _fail(f"{tuple(mask.shape)} {mask.dtype}")
    if int((mask != 0).sum()) != AUDIO_FRAMES - 8:
        _fail("expected trailing-zero mask (avoid dummy_pool const-all-ones)")


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


def test_slice_seq_windows_matches_unfold() -> None:
    torch.manual_seed(1)
    x = torch.randn(1, 93, 8, 16)
    window, step = 24, 12
    ref = torch.movedim(x.unfold(1, window, step), -1, 2)
    got = slice_seq_windows(x, window, step)
    if got.shape != ref.shape:
        _fail(f"shape {tuple(got.shape)} != {tuple(ref.shape)}")
    if not torch.equal(got, ref):
        _fail("slice_seq_windows != unfold+movedim")
    if not torch.equal(got, gather_seq_windows(x, window, step)):
        _fail("slice_seq_windows != gather_seq_windows")


def test_slice_seq_windows_export_has_no_gather() -> None:
    class _Win(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return slice_seq_windows(x, 24, 12)

    ep = torch.export.export(_Win(), (torch.randn(1, 93, 4, 8),), strict=False)
    targets = [str(n.target) for n in ep.graph.nodes]
    if any(("gather" in t or "index_select" in t or "unfold" in t) for t in targets):
        _fail(f"export still contains gather/index_select/unfold: {targets}")


def test_gather_seq_windows_export_has_no_unfold() -> None:
    class _Win(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return gather_seq_windows(x, 24, 12)

    ep = torch.export.export(_Win(), (torch.randn(1, 70, 4, 8),), strict=False)
    targets = [str(n.target) for n in ep.graph.nodes]
    if any("unfold" in t for t in targets):
        _fail(f"export still contains unfold: {targets}")


def test_blocked_additive_mask_shape_and_pad() -> None:
    keep = torch.ones(1, AUDIO_SOFT_TOKENS, dtype=torch.float32)
    keep[:, -8:] = 0
    mask = blocked_additive_attention_mask(keep)
    n_blocks = (AUDIO_SOFT_TOKENS + AUDIO_CHUNK - 1) // AUDIO_CHUNK
    context = AUDIO_CHUNK + AUDIO_PAST + AUDIO_FUTURE
    if tuple(mask.shape) != (1, 1, n_blocks, AUDIO_CHUNK, context):
        _fail(str(tuple(mask.shape)))
    if mask.dtype != torch.float32:
        _fail(f"dtype {mask.dtype}")
    # Last 8 keys of seq 70 land in the final block's current-chunk columns.
    # Those positions must be masked (large negative), earlier keys not.
    if not bool((mask[0, 0, -1, :, AUDIO_PAST:] < -1.0e6).any()):
        _fail("expected padded keys to be invalid in last block")
    if float(mask[0, 0, 0, 0, AUDIO_PAST]) < -1.0:
        _fail(f"first valid key of block 0 should be keep, got {float(mask[0, 0, 0, 0, AUDIO_PAST])}")


def test_blocked_additive_mask_export_has_no_gather() -> None:
    class _Mask(torch.nn.Module):
        def forward(self, keep: torch.Tensor) -> torch.Tensor:
            return blocked_additive_attention_mask(keep)

    ep = torch.export.export(
        _Mask(), (torch.ones(1, AUDIO_SOFT_TOKENS, dtype=torch.float32),), strict=False
    )
    targets = [str(n.target) for n in ep.graph.nodes]
    if any("gather" in t for t in targets):
        _fail(f"export still contains gather: {targets}")


def test_embedding_from_int_indices_matches_embedding() -> None:
    torch.manual_seed(0)
    table = torch.randn(8, 4)
    idx = torch.tensor([[0, 3, 7, 1]], dtype=torch.int16)
    ref = torch.nn.functional.embedding(idx.to(torch.long), table)
    got = embedding_from_int_indices(idx, table)
    if got.shape != ref.shape:
        _fail(str(tuple(got.shape)))
    if not torch.allclose(got, ref, atol=1e-6):
        _fail("int16 one-hot embed != F.embedding")


def test_embedding_from_int_indices_export_has_no_i64() -> None:
    class _Emb(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.table = torch.nn.Parameter(torch.randn(6, 3))

        def forward(self, idx: torch.Tensor) -> torch.Tensor:
            return embedding_from_int_indices(idx, self.table)

    ep = torch.export.export(_Emb(), (torch.zeros(1, 4, dtype=torch.int16),), strict=False)
    for node in ep.graph.nodes:
        blob = f"{node.target} {node.meta.get('val', '')}"
        if "embedding" in str(node.target) or "int64" in blob or "torch.long" in blob:
            _fail(f"export still widens/embeds with i64: {blob}")


def test_blocked_additive_mask_export_stays_float() -> None:
    class _Mask(torch.nn.Module):
        def forward(self, keep: torch.Tensor) -> torch.Tensor:
            return blocked_additive_attention_mask(keep)

    ep = torch.export.export(
        _Mask(), (torch.ones(1, AUDIO_SOFT_TOKENS, dtype=torch.float32),), strict=False
    )
    for node in ep.graph.nodes:
        blob = f"{node.target} {node.meta.get('val', '')}"
        if "i1" in blob or "torch.bool" in blob:
            _fail(f"export still contains bool/i1: {blob}")


def test_io_specs() -> None:
    v = tower_io_spec("vision")
    if v["outputs"]["soft_tokens"] != [1, VISION_SOFT_TOKENS, 512]:
        _fail(str(v))
    if v.get("output_dtypes", {}).get("soft_tokens") != "float16":
        _fail("vision out dtype")
    if v.get("input_dtypes", {}).get("pixel_position_ids") != "float16":
        _fail("vision pos dtype")
    if tower_io_spec("audio")["outputs"]["soft_tokens"] != [1, AUDIO_SOFT_TOKENS, 512]:
        _fail("audio tokens")
    if tower_io_spec("audio").get("output_dtypes", {}).get("soft_tokens") != "float16":
        _fail("audio out dtype")
    if tower_io_spec("text")["outputs"]["embedding"] != [1, 768]:
        _fail("text embed")


def main() -> int:
    tests = [
        test_vision_example_shape,
        test_audio_example_shape,
        test_gather_seq_windows_matches_unfold,
        test_gather_seq_windows_export_has_no_unfold,
        test_slice_seq_windows_matches_unfold,
        test_slice_seq_windows_export_has_no_gather,
        test_blocked_additive_mask_shape_and_pad,
        test_blocked_additive_mask_export_has_no_gather,
        test_blocked_additive_mask_export_stays_float,
        test_embedding_from_int_indices_matches_embedding,
        test_embedding_from_int_indices_export_has_no_i64,
        test_io_specs,
    ]
    for fn in tests:
        fn()
        print(f"OK {fn.__name__}")
    print(f"OK: {len(tests)} coreai-tower checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
