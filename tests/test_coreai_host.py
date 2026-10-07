#!/usr/bin/env python3
"""Unit checks for Core AI host interleave (no checkpoint, no .aimodel)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.coreai_host import (  # noqa: E402
    AUDIO_FRAMES,
    IMAGE_SLOTS,
    has_caption_words,
    uses_text_package,
    adapt_vision_pixels,
    adapt_vision_position_ids,
    crop_audio_soft_to_src,
    crop_vision_soft_to_valid,
    expand_media_placeholders,
    hf_audio_slots_from_frames,
    hf_image_slots_from_positions,
    pad_audio_to_package,
    pad_embeds_to_package,
    TEXT_EMBEDS_S,
    placeholder_masks,
    scatter_soft_tokens,
)


def _fail(msg: str) -> None:
    raise AssertionError(msg)


def test_caption_routing() -> None:
    if not uses_text_package("What causes the northern lights?"):
        _fail("pure text should use text_s128")
    if uses_text_package("title: aurora | text: Green sky glow. <|image|>"):
        _fail("caption+image should interleave")
    if uses_text_package("tone then picture <|audio|> <|image|>"):
        _fail("mix should interleave")
    if uses_text_package("<|image|>"):
        _fail("image-only should interleave")
    if uses_text_package("<|audio|>"):
        _fail("audio-only should interleave")
    if has_caption_words("<|image|> <|audio|>"):
        _fail("media-only is not a caption")


def test_expand_counts() -> None:
    got = expand_media_placeholders("hi <|image|> bye <|audio|>")
    if got.count("<|image|>") != IMAGE_SLOTS:
        _fail(f"image {got.count('<|image|>')}")
    if got.count("<|audio|>") != 70:
        _fail(f"audio {got.count('<|audio|>')}")


def test_scatter_matches_slots() -> None:
    image_id, audio_id = 7, 8
    ids = torch.tensor([[1, 7, 7, 2, 8, 3]])
    embeds = torch.zeros(1, 6, 4)
    img = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
    aud = torch.full((1, 1, 4), 9.0)
    out = scatter_soft_tokens(
        embeds,
        ids,
        image_token_id=image_id,
        audio_token_id=audio_id,
        image_soft=img,
        audio_soft=aud,
    )
    if not torch.equal(out[0, 1], img[0, 0]):
        _fail("image slot 0")
    if not torch.equal(out[0, 2], img[0, 1]):
        _fail("image slot 1")
    if not torch.equal(out[0, 4], aud[0, 0]):
        _fail("audio slot")
    im, au = placeholder_masks(ids, image_token_id=image_id, audio_token_id=audio_id)
    if int(im.sum()) != 2 or int(au.sum()) != 1:
        _fail("masks")


def test_scatter_count_mismatch() -> None:
    ids = torch.tensor([[7, 7, 7]])
    embeds = torch.zeros(1, 3, 2)
    try:
        scatter_soft_tokens(
            embeds,
            ids,
            image_token_id=7,
            image_soft=torch.zeros(1, 2, 2),
        )
    except ValueError:
        return
    _fail("expected ValueError")


def test_vision_pos_si16() -> None:
    pos = np.zeros((1, 2520, 2), dtype=np.int64)
    got = adapt_vision_position_ids(pos)
    if got.dtype != np.int16 or got.shape != (1, 2520, 2):
        _fail(str(got.dtype))
    pix = adapt_vision_pixels(np.zeros((1, 2520, 768), dtype=np.float64))
    if pix.dtype != np.float32:
        _fail(str(pix.dtype))


def test_crop_trailing_vision_pads() -> None:
    pos = np.zeros((1, 2520, 2), dtype=np.int16)
    pos[:, 2304:] = -1
    if hf_image_slots_from_positions(pos) != 256:
        _fail("256 valid groups")
    soft = np.arange(280 * 2, dtype=np.float32).reshape(1, 280, 2)
    got = crop_vision_soft_to_valid(soft, pos)
    if got.shape != (1, 256, 2) or not np.array_equal(got, soft[:, :256]):
        _fail(str(got.shape))


def test_crop_audio_ceil_frames() -> None:
    if hf_audio_slots_from_frames(99) != 25:
        _fail("99 frames → 25 tokens")
    soft = np.ones((1, 70, 3), dtype=np.float32)
    got = crop_audio_soft_to_src(soft, 99)
    if got.shape != (1, 25, 3):
        _fail(str(got.shape))


def test_pad_embeds() -> None:
    emb = torch.ones(1, 10, 4)
    mask = torch.ones(1, 10)
    got, m = pad_embeds_to_package(emb, mask, seq_len=16)
    if tuple(got.shape) != (1, 16, 4) or int(m.sum()) != 10:
        _fail(f"{tuple(got.shape)} {int(m.sum())}")
    try:
        pad_embeds_to_package(emb, mask, seq_len=8)
    except ValueError:
        pass
    else:
        _fail("expected overflow")
    if TEXT_EMBEDS_S < 288:
        _fail("S must fit mm_mix 288")


def test_audio_pad() -> None:
    feat = np.ones((1, 99, 128), dtype=np.float32)
    x, m = pad_audio_to_package(feat)
    if x.shape != (1, AUDIO_FRAMES, 128) or m.dtype != np.int16:
        _fail(f"{x.shape} {m.dtype}")
    if int(m.sum()) != 99 or int(m[0, 98]) != 1 or int(m[0, 99]) != 0:
        _fail(str(m.sum()))


def main() -> int:
    tests = [
        test_caption_routing,
        test_expand_counts,
        test_scatter_matches_slots,
        test_scatter_count_mismatch,
        test_vision_pos_si16,
        test_crop_trailing_vision_pads,
        test_crop_audio_ceil_frames,
        test_pad_embeds,
        test_audio_pad,
    ]
    for fn in tests:
        fn()
        print(f"OK {fn.__name__}")
    print(f"OK: {len(tests)} coreai-host checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
