"""Host interleave for separate Core AI towers (CPU).

Vision/audio ``.aimodel`` packages emit fixed soft tokens (280 / 70).
The HF processor expands ``<|image|>`` from *valid* patches (256 on the
64² fixture) and audio from duration (25 for 1 s). This host uses the
**package** slot counts so scatter lines up with exported graphs.

``text_s128.aimodel`` is ids-only and S=128 — pure text only.
``text_embeds_s320.aimodel`` takes scattered ``inputs_embeds`` [1, 320, 512]
(host token lookup + vision/audio soft tokens). ANE specialize stays a follow-up.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .coreai_towers import (
    AUDIO_FEAT,
    AUDIO_FRAMES,
    AUDIO_SOFT_TOKENS,
    VISION_PATCH_DIM,
    VISION_PATCHES,
    VISION_SOFT_TOKENS,
)

IMAGE_SLOTS = VISION_SOFT_TOKENS
AUDIO_SLOTS = AUDIO_SOFT_TOKENS
TEXT_PACKAGE_S = 128
# Caption + 256 image + 25 audio (mm_mix seq 288) + pad.
TEXT_EMBEDS_S = 320
TEXT_HIDDEN = 512


def has_caption_words(
    text: str,
    *,
    image_token: str = "<|image|>",
    audio_token: str = "<|audio|>",
    video_token: str = "<|video|>",
) -> bool:
    """True when the prompt has words besides media placeholders."""
    out = text or ""
    for tok in (image_token, audio_token, video_token):
        if tok:
            out = out.replace(tok, " ")
    return bool(out.split())


def uses_text_package(
    text: str,
    *,
    image_token: str = "<|image|>",
    audio_token: str = "<|audio|>",
    video_token: str = "<|video|>",
) -> bool:
    """True only for pure text. Any media placeholder → soft-token interleave."""
    raw = text or ""
    for tok in (image_token, audio_token, video_token):
        if tok and tok in raw:
            return False
    return bool(raw.strip())


def expand_media_placeholders(
    text: str,
    *,
    image_token: str = "<|image|>",
    audio_token: str = "<|audio|>",
    image_slots: int = IMAGE_SLOTS,
    audio_slots: int = AUDIO_SLOTS,
) -> str:
    """Replace each media token with ``slots`` copies (package contract)."""
    out = text
    if image_token and int(image_slots) > 0:
        out = out.replace(image_token, image_token * int(image_slots))
    if audio_token and int(audio_slots) > 0:
        out = out.replace(audio_token, audio_token * int(audio_slots))
    return out


def adapt_vision_position_ids(pos: np.ndarray | torch.Tensor) -> np.ndarray:
    """``[1, 2520, 2]`` → int16 for the vision package."""
    arr = np.asarray(pos)
    if arr.shape != (1, VISION_PATCHES, 2):
        raise ValueError(f"vision pos shape {arr.shape} != (1, {VISION_PATCHES}, 2)")
    return arr.astype(np.int16, copy=False)


def adapt_vision_pixels(pixels: np.ndarray | torch.Tensor) -> np.ndarray:
    arr = np.asarray(pixels)
    if arr.shape != (1, VISION_PATCHES, VISION_PATCH_DIM):
        raise ValueError(f"pixels shape {arr.shape}")
    return arr.astype(np.float16, copy=False)


def hf_image_slots_from_positions(
    pos: np.ndarray | torch.Tensor,
    *,
    group: int = 9,
) -> int:
    """Valid-patch groups (HF pooler). Pads must be a trailing suffix."""
    arr = np.asarray(pos)
    if arr.ndim == 3:
        arr = arr[0]
    valid = (arr != -1).all(axis=-1)
    n_valid = int(valid.sum())
    if n_valid % int(group) != 0:
        raise ValueError(f"valid patches {n_valid} not divisible by {group}")
    if not bool(valid[:n_valid].all()) or bool(valid[n_valid:].any()):
        raise ValueError("vision pads are not a trailing suffix; cannot crop")
    return n_valid // int(group)


def crop_vision_soft_to_valid(
    soft: np.ndarray | torch.Tensor,
    pos: np.ndarray | torch.Tensor,
) -> np.ndarray:
    """Drop trailing pad-pooled groups so slots match HF (256 on the 64² fixture)."""
    arr = np.asarray(soft)
    keep = hf_image_slots_from_positions(pos)
    if arr.shape[1] < keep:
        raise ValueError(f"soft tokens {arr.shape} shorter than {keep} valid groups")
    return arr[:, :keep]


def hf_audio_slots_from_frames(n_frames: int, *, subsample: int = 4) -> int:
    """``ceil(frames / 4)`` — matches 99 mel frames → 25 tokens on the 1 s wav."""
    n = int(n_frames)
    step = int(subsample)
    return (n + step - 1) // step


def crop_audio_soft_to_src(
    soft: np.ndarray | torch.Tensor,
    n_src_frames: int,
) -> np.ndarray:
    arr = np.asarray(soft)
    keep = hf_audio_slots_from_frames(n_src_frames)
    if arr.shape[1] < keep:
        raise ValueError(f"audio soft {arr.shape} shorter than {keep}")
    return arr[:, :keep]


def pad_audio_to_package(
    feat: np.ndarray | torch.Tensor,
    mask: np.ndarray | torch.Tensor | None = None,
    *,
    frames: int = AUDIO_FRAMES,
) -> tuple[np.ndarray, np.ndarray]:
    """Pad/crop mel to ``[1, 280, 128]`` + int16 keep-mask."""
    x = np.asarray(feat)
    if x.ndim == 2:
        x = x[None, ...]
    if x.ndim != 3 or x.shape[-1] != AUDIO_FEAT:
        raise ValueError(f"audio feat shape {x.shape}")
    cur = int(x.shape[1])
    if cur < frames:
        x = np.pad(x, ((0, 0), (0, frames - cur), (0, 0)))
    elif cur > frames:
        x = x[:, :frames, :]
    keep = np.zeros((x.shape[0], frames), dtype=np.int16)
    if mask is None:
        keep[:, : min(cur, frames)] = 1
    else:
        m = np.asarray(mask)
        if m.ndim == 1:
            m = m[None, :]
        m = m.astype(np.int16, copy=False)
        n = min(int(m.shape[-1]), frames)
        keep[:, :n] = (m[:, :n] != 0).astype(np.int16)
    return x.astype(np.float32, copy=False), keep


def placeholder_masks(
    input_ids: torch.Tensor,
    *,
    image_token_id: int,
    audio_token_id: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    image = input_ids == int(image_token_id)
    if audio_token_id is None:
        audio = torch.zeros_like(image)
    else:
        audio = input_ids == int(audio_token_id)
    return image, audio


def scatter_soft_tokens(
    inputs_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    *,
    image_token_id: int,
    audio_token_id: int | None = None,
    image_soft: torch.Tensor | None = None,
    audio_soft: torch.Tensor | None = None,
) -> torch.Tensor:
    """Replace placeholder rows with tower soft tokens. Slot counts must match."""
    image_mask, audio_mask = placeholder_masks(
        input_ids, image_token_id=image_token_id, audio_token_id=audio_token_id
    )
    hidden = int(inputs_embeds.shape[-1])
    out = inputs_embeds
    if image_soft is not None:
        n = int(image_mask.sum())
        flat = image_soft.reshape(-1, hidden)
        if int(flat.shape[0]) != n:
            raise ValueError(f"image slots {n} != soft tokens {tuple(image_soft.shape)}")
        out = out.masked_scatter(
            image_mask.unsqueeze(-1).expand_as(out),
            flat.to(device=out.device, dtype=out.dtype),
        )
    if audio_soft is not None:
        n = int(audio_mask.sum())
        flat = audio_soft.reshape(-1, hidden)
        if int(flat.shape[0]) != n:
            raise ValueError(f"audio slots {n} != soft tokens {tuple(audio_soft.shape)}")
        out = out.masked_scatter(
            audio_mask.unsqueeze(-1).expand_as(out),
            flat.to(device=out.device, dtype=out.dtype),
        )
    return out


def embeds_from_ids(
    text_model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    image_token_id: int,
    audio_token_id: int | None,
    pad_token_id: int,
) -> torch.Tensor:
    """Embed ids; multimodal slots use pad so OOV image/audio ids are safe."""
    image_mask, audio_mask = placeholder_masks(
        input_ids, image_token_id=image_token_id, audio_token_id=audio_token_id
    )
    safe = input_ids.masked_fill(image_mask | audio_mask, int(pad_token_id))
    return text_model.get_input_embeddings()(safe)


def pad_embeds_to_package(
    embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    seq_len: int = TEXT_EMBEDS_S,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad or reject ``[B, S, 512]`` embeds + mask to the text-embeds S."""
    if embeds.ndim != 3 or attention_mask.ndim != 2:
        raise ValueError(f"embeds/mask {tuple(embeds.shape)} / {tuple(attention_mask.shape)}")
    if embeds.shape[:2] != attention_mask.shape:
        raise ValueError("embeds/mask batch-seq mismatch")
    cur = int(embeds.shape[1])
    target = int(seq_len)
    if cur > target:
        raise ValueError(f"seq {cur} > text-embeds S={target}")
    if cur == target:
        return embeds, attention_mask
    pad = target - cur
    hidden = int(embeds.shape[-1])
    z = embeds.new_zeros(embeds.shape[0], pad, hidden)
    m = attention_mask.new_zeros(attention_mask.shape[0], pad)
    return torch.cat([embeds, z], dim=1), torch.cat([attention_mask, m], dim=1)


def interleaved_inputs_embeds(
    text_model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    image_token_id: int,
    audio_token_id: int | None,
    pad_token_id: int,
    image_soft: torch.Tensor | None = None,
    audio_soft: torch.Tensor | None = None,
) -> torch.Tensor:
    """Token embed lookup + scatter media soft tokens (host, not the backbone)."""
    embeds = embeds_from_ids(
        text_model,
        input_ids,
        image_token_id=image_token_id,
        audio_token_id=audio_token_id,
        pad_token_id=pad_token_id,
    )
    return scatter_soft_tokens(
        embeds,
        input_ids,
        image_token_id=image_token_id,
        audio_token_id=audio_token_id,
        image_soft=image_soft,
        audio_soft=audio_soft,
    )


def encode_interleaved(
    wrapper: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    image_token_id: int,
    audio_token_id: int | None,
    pad_token_id: int,
    image_soft: torch.Tensor | None = None,
    audio_soft: torch.Tensor | None = None,
) -> torch.Tensor:
    """Text tower on scattered embeds → L2 768-d (same pool-then-project)."""
    embeds = interleaved_inputs_embeds(
        wrapper.text_model,
        input_ids,
        image_token_id=image_token_id,
        audio_token_id=audio_token_id,
        pad_token_id=pad_token_id,
        image_soft=image_soft,
        audio_soft=audio_soft,
    )
    out = wrapper.text_model(inputs_embeds=embeds, attention_mask=attention_mask)
    hidden = out.last_hidden_state
    pooled = wrapper.pool(hidden, attention_mask)
    emb = wrapper.projection(pooled)
    if wrapper.normalize:
        emb = F.normalize(emb, p=2, dim=-1)
    return emb


def slot_report(
    input_ids: torch.Tensor,
    *,
    image_token_id: int,
    audio_token_id: int | None,
) -> dict[str, Any]:
    image_mask, audio_mask = placeholder_masks(
        input_ids, image_token_id=image_token_id, audio_token_id=audio_token_id
    )
    return {
        "seq_len": int(input_ids.shape[-1]),
        "image_slots": int(image_mask.sum()),
        "audio_slots": int(audio_mask.sum()),
    }
