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
        padding_positions = (pixel_position_ids == -1).all(dim=-1)
        inputs_embeds = self.vision_tower.patch_embedder(
            pixel_values, pixel_position_ids, padding_positions
        )
        encoded = self.vision_tower.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=~padding_positions,
            pixel_position_ids=pixel_position_ids,
        )
        hidden_states, _mask = self.vision_tower.pooler(
            hidden_states=encoded.last_hidden_state,
            pixel_position_ids=pixel_position_ids,
            padding_positions=padding_positions,
            output_length=self.output_length,
        )
        hidden_states = hidden_states.to(dtype=inputs_embeds.dtype)
        return self.embed_vision(hidden_states)


class AudioSoftTokens(nn.Module):
    """audio_tower + embed_audio → ``[B, T', 512]`` (subsampled T')."""

    def __init__(self, audio_tower: nn.Module, embed_audio: nn.Module) -> None:
        super().__init__()
        self.audio_tower = audio_tower
        self.embed_audio = embed_audio

    def forward(
        self, input_features: torch.Tensor, input_features_mask: torch.Tensor
    ) -> torch.Tensor:
        out = self.audio_tower(input_features, input_features_mask, return_dict=True)
        return self.embed_audio(out.last_hidden_state)


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
        (batch, patches, VISION_PATCH_DIM), 0.5, dtype=dtype, device=device
    )
    return pixels, pos.to(dtype=torch.long)


def audio_example(
    batch: int = 1,
    frames: int = AUDIO_FRAMES,
    *,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    feat = torch.zeros(batch, frames, AUDIO_FEAT, dtype=dtype, device=device)
    mask = torch.ones(batch, frames, dtype=torch.bool, device=device)
    return feat, mask


def tower_io_spec(name: str) -> dict[str, Any]:
    if name == "vision":
        return {
            "inputs": {
                "pixel_values": [1, VISION_PATCHES, VISION_PATCH_DIM],
                "pixel_position_ids": [1, VISION_PATCHES, 2],
            },
            "outputs": {"soft_tokens": [1, VISION_SOFT_TOKENS, TEXT_HIDDEN]},
        }
    if name == "audio":
        return {
            "inputs": {
                "input_features": [1, AUDIO_FRAMES, AUDIO_FEAT],
                "input_features_mask": [1, AUDIO_FRAMES],
            },
            "outputs": {"soft_tokens": "subsampled [1, T, 512]"},
        }
    if name == "text":
        return {
            "inputs": {"input_ids": [1, "S"], "attention_mask": [1, "S"]},
            "outputs": {"embedding": [1, TEXT_EMBED]},
        }
    raise ValueError(f"unknown tower {name!r}")
