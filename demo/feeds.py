"""Package feeds for the Core AI towers.

Audio and text attention masks are the processor / tokenizer masks, padded
with zeros out to the fixed package length. They are never replaced with
all ones. An all-ones audio mask lets fp16 pad noise into real frames on
the ANE. The graph's additive mask constant is ``-1e4`` (fp16-safe); the
host only passes the 0/1 keep-mask, same as ``scripts/parity_coreai_host.py``.

Vision position ids are the processor grid, including trailing ``-1`` pads.
"""

from __future__ import annotations

import numpy as np

from src.coreai_host import (
    adapt_audio_features_nchw,
    adapt_audio_mask_nchw,
    adapt_vision_pixels,
    adapt_vision_position_ids,
    pad_audio_to_package,
)

# Same contract as ``src.coreai_towers``. Kept here for the mask-length check.
AUDIO_FRAMES = 280
AUDIO_FEAT = 128


def audio_tower_feed(
    features: np.ndarray,
    mask: np.ndarray | None,
) -> dict[str, np.ndarray]:
    """Mel + real keep-mask in the audio package's NCHW layout.

    Uses ``pad_audio_to_package`` from the existing host. Does not fill the
    mask with ones.
    """
    feat, amask = pad_audio_to_package(features, mask)
    assert_audio_mask_preserved(mask, amask)
    return {
        "input_features": adapt_audio_features_nchw(feat),
        "input_features_mask": adapt_audio_mask_nchw(amask),
    }


def vision_tower_feed(
    pixels: np.ndarray,
    position_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    """Pixels + processor position ids. Pad slots (``-1``) stay pad slots."""
    src = np.asarray(position_ids)
    pos = adapt_vision_position_ids(position_ids)
    if np.any(src < 0) and not np.any(np.asarray(pos) < 0):
        raise RuntimeError("vision pad positions were dropped from the tower feed")
    return {
        "pixel_values": adapt_vision_pixels(pixels),
        "pixel_position_ids": pos,
    }


def assert_audio_mask_preserved(
    src_mask: np.ndarray | None,
    packed_mask: np.ndarray,
) -> None:
    """Fail if pad frames were turned on or source keep-bits were rewritten."""
    packed = np.asarray(packed_mask)
    if packed.ndim == 1:
        packed = packed[None, :]
    elif packed.ndim == 4:
        packed = packed[:, 0, :, 0]
    elif packed.ndim != 2:
        raise RuntimeError(f"audio mask rank {packed.ndim}")
    frames = int(packed.shape[-1])
    if src_mask is None:
        # Host treats a missing mask as "valid up to the incoming frame count",
        # which ``pad_audio_to_package`` already encoded. Nothing to compare.
        return
    src = np.asarray(src_mask)
    if src.ndim == 1:
        src = src[None, :]
    src_n = min(int(src.shape[-1]), frames)
    expect = (src[:, :src_n] != 0).astype(np.int16)
    got = (packed[:, :src_n] != 0).astype(np.int16)
    if not np.array_equal(expect, got):
        raise RuntimeError(
            "audio attention mask was altered before the tower "
            f"(kept {int(got.sum())}, source {int(expect.sum())})"
        )
    if src_n < frames and int((packed[:, src_n:] != 0).sum()) != 0:
        raise RuntimeError(
            "audio pad frames must stay masked out (all-ones mask breaks ANE audio)"
        )


def assert_text_mask_preserved(valid_count: int, packed_mask: np.ndarray) -> None:
    """The text-embeds mask is ones on real tokens and zeros on the right pad."""
    flat = np.asarray(packed_mask).reshape(-1)
    got = int((flat != 0).sum())
    if got != int(valid_count):
        raise RuntimeError(
            f"text attention mask changed ({int(valid_count)} tokens -> {got})"
        )
    if got < flat.size:
        if np.any(flat[got:] != 0):
            raise RuntimeError("text pad must be a trailing zero suffix")
        if np.any(flat[:got] == 0):
            raise RuntimeError("text keep-mask must be a leading ones prefix")
