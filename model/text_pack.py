"""Packed text tower: several short texts in one fixed-shape forward.

``PackedTextEmbeds`` runs up to ``max_texts`` texts laid end to end in one
``[1, N, 512]`` sequence. Four inputs keep the texts apart:

* ``inputs_embeds`` ``[1, N, 512]`` f16: token rows (already scaled by the
  host lookup), zero rows after the last text.
* ``attention_bias`` ``[1, 1, N, N]`` f16: ``0`` where query and key belong
  to the same text, ``-1e4`` everywhere else (finite, so fp16 stays legal).
* ``positions`` ``[N, 1]`` f16: RoPE position of each token, restarting at
  0 for every text. Integers below 2048 are exact in f16.
* ``pool`` ``[max_texts, N]`` f16: row ``t`` is ``1/len_t`` over text ``t``'s
  tokens and 0 elsewhere (an all-zero row is an unused slot).

Output ``embedding`` ``[max_texts, 768]``: the projected mean of each text,
NOT normalized (an unused slot would divide by zero in fp16). The host
L2-normalizes the rows it filled.

Every sequence here is at most a few hundred tokens, below the 1024-token
sliding window, so the sliding and full layers take the same bias.

RoPE: the tables for positions ``0..N-1`` are constants (as in the
fixed-S towers). The per-token rows are picked with a one-hot matrix built
by arithmetic, ``relu(1 - |p - j|)``, then a matmul. That avoids sin/cos
and integer gathers in the graph, which the ANE would not take.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .embed_wrapper import EmbeddingGemma2Wrapper
from .trace_patches import _take_ple_layer

PACK_NEG = -1.0e4


class PackedTextEmbeds(nn.Module):
    def __init__(
        self,
        wrapper: EmbeddingGemma2Wrapper,
        *,
        n_tokens: int,
        max_texts: int,
    ) -> None:
        super().__init__()
        self.text_model = wrapper.text_model
        self.projection = wrapper.projection
        self.n_tokens = int(n_tokens)
        self.max_texts = int(max_texts)
        self.register_buffer(
            "position_ids",
            torch.arange(self.n_tokens, dtype=torch.long).unsqueeze(0),
            persistent=False,
        )
        self.register_buffer(
            "position_grid",
            torch.arange(self.n_tokens, dtype=torch.float32).unsqueeze(0),
            persistent=False,
        )

    def _rope(self, hidden: torch.Tensor, positions: torch.Tensor) -> dict:
        tm = self.text_model
        pos = positions.to(torch.float32).reshape(self.n_tokens, 1)
        onehot = F.relu(1.0 - (pos - self.position_grid).abs())  # [N, N]
        out = {}
        for layer_type in sorted(tm.unique_layer_types):
            cos, sin = tm.rotary_emb(hidden, self.position_ids, layer_type)
            out[layer_type] = (
                (onehot @ cos[0].to(torch.float32)).unsqueeze(0).to(hidden.dtype),
                (onehot @ sin[0].to(torch.float32)).unsqueeze(0).to(hidden.dtype),
            )
        return out

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_bias: torch.Tensor,
        positions: torch.Tensor,
        pool: torch.Tensor,
    ) -> torch.Tensor:
        tm = self.text_model
        hidden = inputs_embeds.to(dtype=torch.float32)
        bias = attention_bias.to(dtype=torch.float32)
        masks = {"full_attention": bias, "sliding_attention": bias}
        rope = self._rope(hidden, positions)
        per_layer = tm.ple(hidden)
        for i, layer in enumerate(tm.layers):
            layer_type = tm.config.layer_types[i]
            hidden = layer(
                hidden,
                _take_ple_layer(per_layer, i),
                attention_mask=masks[layer_type],
                position_embeddings=rope[layer_type],
            )
        hidden = tm.norm(hidden)
        hidden = tm.embedding_projection(hidden)  # Identity after pool/project split
        # Pool as hidden^T @ pool^T ([512, N] @ [N, T]), then transpose.
        # ``pool @ hidden`` (an [8, N] input on the left, also as a 4-D
        # matmul) put the whole tower on the GPU on macOS 27.0; this form
        # and an elementwise multiply + sum both stay on the ANE.
        flat = hidden.reshape(self.n_tokens, -1)
        pooled = (flat.transpose(0, 1) @ pool.to(torch.float32).transpose(0, 1)).transpose(0, 1)
        return self.projection(pooled)  # [T, 768], host normalizes


def example_pack_inputs(
    n_tokens: int, max_texts: int, hidden: int = 512, lengths: tuple[int, ...] = (40, 17, 90)
) -> tuple[torch.Tensor, ...]:
    """A legal packed feed for tracing: a few texts, padding after them."""
    embeds = torch.zeros(1, n_tokens, hidden, dtype=torch.float16)
    bias = torch.full((1, 1, n_tokens, n_tokens), PACK_NEG, dtype=torch.float16)
    positions = torch.zeros(n_tokens, 1, dtype=torch.float16)
    pool = torch.zeros(max_texts, n_tokens, dtype=torch.float16)
    start = 0
    for slot, length in enumerate(lengths[:max_texts]):
        end = min(n_tokens, start + int(length))
        if end <= start:
            break
        embeds[0, start:end] = torch.randn(end - start, hidden, dtype=torch.float16)
        bias[0, 0, start:end, start:end] = 0
        positions[start:end, 0] = torch.arange(end - start, dtype=torch.float16)
        pool[slot, start:end] = 1.0 / float(end - start)
        start = end
    # Padding tokens attend to themselves so no row is fully masked.
    for idx in range(start, n_tokens):
        bias[0, 0, idx, idx] = 0
    return embeds, bias, positions, pool
