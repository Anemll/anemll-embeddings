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
        # I/O is si16. F.embedding requires Int/Long — cannot keep si16 in-graph.
        pixel_position_ids = pixel_position_ids.to(dtype=torch.long)
        padding_positions = (pixel_position_ids == -1).all(dim=-1)
        inputs_embeds = self.vision_tower.patch_embedder(
            pixel_values, pixel_position_ids, padding_positions
        )
        # ANE rejects i1 bool masks (memref …x2520xi1). 4D float additive is
        # returned as-is by HF create_bidirectional_mask.
        keep = (~padding_positions).to(dtype=inputs_embeds.dtype)
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
        return self.embed_vision(hidden)


class AudioSoftTokens(nn.Module):
    """audio_tower + embed_audio → ``[B, T', 512]`` (subsampled T')."""

    def __init__(self, audio_tower: nn.Module, embed_audio: nn.Module) -> None:
        super().__init__()
        self.audio_tower = audio_tower
        self.embed_audio = embed_audio

    def forward(
        self, input_features: torch.Tensor, input_features_mask: torch.Tensor
    ) -> torch.Tensor:
        keep = input_features_mask
        if keep.dtype != torch.bool:
            keep = keep != 0
        out = self.audio_tower(input_features, keep, return_dict=True)
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
    return pixels, pos.to(dtype=torch.int16)


def audio_example(
    batch: int = 1,
    frames: int = AUDIO_FRAMES,
    *,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    feat = torch.zeros(batch, frames, AUDIO_FEAT, dtype=dtype, device=device)
    mask = torch.ones(batch, frames, dtype=torch.int16, device=device)
    return feat, mask


def tower_io_spec(name: str) -> dict[str, Any]:
    if name == "vision":
        return {
            "inputs": {
                "pixel_values": [1, VISION_PATCHES, VISION_PATCH_DIM],
                "pixel_position_ids": [1, VISION_PATCHES, 2],
            },
            "input_dtypes": {
                "pixel_values": "float32",
                "pixel_position_ids": "int16",
            },
            "outputs": {"soft_tokens": [1, VISION_SOFT_TOKENS, TEXT_HIDDEN]},
        }
    if name == "audio":
        return {
            "inputs": {
                "input_features": [1, AUDIO_FRAMES, AUDIO_FEAT],
                "input_features_mask": [1, AUDIO_FRAMES],
            },
            "input_dtypes": {
                "input_features": "float32",
                "input_features_mask": "int16",
            },
            "outputs": {"soft_tokens": [1, AUDIO_SOFT_TOKENS, TEXT_HIDDEN]},
        }
    if name == "text":
        return {
            "inputs": {"input_ids": [1, "S"], "attention_mask": [1, "S"]},
            "input_dtypes": {"input_ids": "int32", "attention_mask": "int32"},
            "outputs": {"embedding": [1, TEXT_EMBED]},
        }
    raise ValueError(f"unknown tower {name!r}")
