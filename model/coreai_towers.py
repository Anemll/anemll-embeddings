"""Fixed-shape multimodal towers for Core AI (separate packages).

Host runs the HF processor and interleaves ``<|image|>`` / ``<|video|>`` /
``<|audio|>`` soft tokens. Each package is one encoder:

* vision: patched pixels → 280 soft tokens in text hidden space (512)
* audio: mel frames → soft tokens in text hidden space (512)
* text: ids/mask → L2 embedding (768), same graph as T4

Wrappers skip boolean-index strip / ``torch.split`` so ``torch.export`` sees
a static layout. Shapes match EmbeddingGemma2Processor defaults.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

# Processor defaults (64² PNG still resizes to this patch grid).
VISION_PATCHES = 2520
VISION_PATCH_DIM = 768  # 3 * 16 * 16
VISION_SOFT_TOKENS = 280
AUDIO_FRAMES = 280
AUDIO_FEAT = 128
AUDIO_SOFT_TOKENS = 70  # 280 frames / 4× subsample
TEXT_HIDDEN = 512
TEXT_EMBED = 768


class VisionSoftTokens(nn.Module):
    """vision_tower + embed_vision, keep ``[B, 280, 512]`` (no pad strip)."""

    def __init__(
        self,
        vision_tower: nn.Module,
        embed_vision: nn.Module,
        *,
        output_length: int = VISION_SOFT_TOKENS,
    ) -> None:
        super().__init__()
        self.vision_tower = vision_tower
        self.embed_vision = embed_vision
        self.output_length = int(output_length)

    def forward(
        self, pixel_values: torch.Tensor, pixel_position_ids: torch.Tensor
    ) -> torch.Tensor:
        # Pixels + pos are f16 I/O (ANE-legal). Encoder stays f32 — no cast16.
        # si16 pos I/O needed ane_io_cast si16→f32 (InvalidOutputType; only
        # F16 MemRef <-> F32 Tensor is legal). Grid coords fit f16 exactly.
        pixel_values = pixel_values.to(dtype=torch.float32)
        pos_f = pixel_position_ids.to(dtype=torch.float32)
        keep = torch.clamp(pos_f + 1.0, 0.0, 1.0).amin(dim=-1)
        padding = 1.0 - keep
        inputs_embeds = self.vision_tower.patch_embedder(
            pixel_values, pixel_position_ids, padding
        )
        attn = (keep - 1.0) * 1.0e4
        encoded = self.vision_tower.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attn[:, None, None, :],
            pixel_position_ids=pixel_position_ids,
        )
        hidden = encoded.last_hidden_state
        # Fixed 2520→280 is a 3×3 mean (k^2=9). Avoid int torch.div in the
        # HF one_hot pooler — Core AI cast16 turns those indices si32 vs si16.
        group = hidden.shape[1] // self.output_length
        hidden = hidden.reshape(hidden.shape[0], self.output_length, group, hidden.shape[-1])
        hidden = hidden.mean(dim=2)
        hidden = hidden.to(dtype=inputs_embeds.dtype)
        scale = float(self.vision_tower.pooler.root_hidden_size)
        hidden = hidden * scale
        # f16 out is ANE-legal I/O (f32 soft_tokens was InvalidOutputType).
        return self.embed_vision(hidden).to(dtype=torch.float16)


class AudioSoftTokens(nn.Module):
    """audio_tower + embed_audio → ``[B, T', 512]`` (subsampled T')."""

    def __init__(self, audio_tower: nn.Module, embed_audio: nn.Module) -> None:
        super().__init__()
        self.audio_tower = audio_tower
        self.embed_audio = embed_audio

    def forward(
        self, input_features: torch.Tensor, input_features_mask: torch.Tensor
    ) -> torch.Tensor:
        # 4-D NCHW I/O so ANE I/O is expandDims (not layout_conversion_permute).
        # features [B, 1, T, C], mask [B, 1, T, 1], out [B, 1, T', 512].
        input_features = input_features.to(dtype=torch.float32)
        keep = input_features_mask.to(dtype=input_features.dtype)
        keep = keep.reshape(input_features.shape[0], input_features.shape[2])
        out = self.audio_tower(input_features, keep, return_dict=True)
        hidden = self.embed_audio(out.last_hidden_state).to(dtype=torch.float16)
        return hidden.reshape(hidden.shape[0], 1, hidden.shape[1], hidden.shape[2])


def vision_example(
    batch: int = 1,
    *,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense (x, y) patch grid covering 2520 patches (70×36)."""
    patches = VISION_PATCHES
    xs = torch.arange(patches, device=device) % 70
    ys = torch.arange(patches, device=device) // 70
    pos = torch.stack((xs, ys), dim=-1).unsqueeze(0).expand(batch, -1, -1).contiguous()
    pixels = torch.full(
        (batch, patches, VISION_PATCH_DIM), 0.5, dtype=torch.float16, device=device
    )
    return pixels, pos.to(dtype=torch.float16)


def audio_example(
    batch: int = 1,
    frames: int = AUDIO_FRAMES,
    *,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    feat = torch.zeros(batch, 1, frames, AUDIO_FEAT, dtype=torch.float16, device=device)
    # Trailing zeros so the mask is not a const-all-ones live-out (dummy_pool).
    mask = torch.ones(batch, 1, frames, 1, dtype=torch.float16, device=device)
    mask[:, :, -8:, :] = 0
    return feat, mask


def tower_io_spec(name: str) -> dict[str, Any]:
    if name == "vision":
        return {
            "inputs": {
                "pixel_values": [1, VISION_PATCHES, VISION_PATCH_DIM],
                "pixel_position_ids": [1, VISION_PATCHES, 2],
            },
            "input_dtypes": {
                "pixel_values": "float16",
                "pixel_position_ids": "float16",
            },
            "outputs": {"soft_tokens": [1, VISION_SOFT_TOKENS, TEXT_HIDDEN]},
            "output_dtypes": {"soft_tokens": "float16"},
        }
    if name == "audio":
        return {
            "inputs": {
                "input_features": [1, 1, AUDIO_FRAMES, AUDIO_FEAT],
                "input_features_mask": [1, 1, AUDIO_FRAMES, 1],
            },
            "input_dtypes": {
                "input_features": "float16",
                "input_features_mask": "float16",
            },
            "outputs": {"soft_tokens": [1, 1, AUDIO_SOFT_TOKENS, TEXT_HIDDEN]},
            "output_dtypes": {"soft_tokens": "float16"},
        }
    if name == "text":
        return {
            "inputs": {"input_ids": [1, "S"], "attention_mask": [1, "S"]},
            "input_dtypes": {"input_ids": "int32", "attention_mask": "int32"},
            "outputs": {"embedding": [1, TEXT_EMBED]},
        }
    if name == "text_embeds":
        return {
            "inputs": {"inputs_embeds": [1, 320, TEXT_HIDDEN], "attention_mask": [1, 320]},
            "input_dtypes": {"inputs_embeds": "float16", "attention_mask": "float16"},
            "outputs": {"embedding": [1, TEXT_EMBED]},
        }
    raise ValueError(f"unknown tower {name!r}")
