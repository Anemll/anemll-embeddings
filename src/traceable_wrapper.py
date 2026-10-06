"""Fixed-S, jit.trace-friendly EmbeddingGemma 2 wrapper (T4).

Same pool-then-project graph as ``EmbeddingGemma2Wrapper``, but attention
masks are built with tensor ops and passed as HF's dict mapping so the
traced graph always materializes pad + sliding-window bias.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embed_wrapper import EmbeddingGemma2Wrapper, MaskedMeanPool
from .export_utils import (
    MASK_NEG,
    additive_full_attention_bias,
    additive_sliding_attention_bias,
)


def _last_hidden(out: object) -> torch.Tensor:
    hidden = getattr(out, "last_hidden_state", None)
    if hidden is not None:
        return hidden
    if isinstance(out, (tuple, list)):
        return out[0]
    raise TypeError(f"cannot read last_hidden_state from {type(out)}")


class TraceableEmbeddingGemma2(nn.Module):
    """ids/mask [B, S] → L2 embedding [B, 768] with in-graph 4D attention bias."""

    def __init__(
        self,
        wrapper: EmbeddingGemma2Wrapper,
        *,
        mask_neg: float = MASK_NEG,
    ) -> None:
        super().__init__()
        self.text_model = wrapper.text_model
        self.projection = wrapper.projection
        self.pool = wrapper.pool if isinstance(wrapper.pool, nn.Module) else MaskedMeanPool()
        self.normalize = bool(wrapper.normalize)
        self.mask_neg = float(mask_neg)
        cfg = self.text_model.config
        self.sliding_window = int(getattr(cfg, "sliding_window", 512))
        self.hidden_size = int(wrapper.hidden_size)
        self.embedding_dim = int(wrapper.embedding_dim)

    def _mask_dtype(self) -> torch.dtype:
        weight = self.projection.weight
        if weight.dtype == torch.float16:
            # Never compute the encoder in FP16 (upstream NaN policy).
            return torch.float32
        return weight.dtype

    def _attention_mapping(
        self, attention_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        dtype = self._mask_dtype()
        return {
            "full_attention": additive_full_attention_bias(
                attention_mask, dtype, neg=self.mask_neg
            ),
            "sliding_attention": additive_sliding_attention_bias(
                attention_mask, self.sliding_window, dtype, neg=self.mask_neg
            ),
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        input_ids = input_ids.to(dtype=torch.long)
        masks = self._attention_mapping(attention_mask)
        out = self.text_model(input_ids=input_ids, attention_mask=masks)
        hidden = _last_hidden(out)
        pooled = self.pool(hidden, attention_mask)
        emb = self.projection(pooled)
        if self.normalize:
            emb = F.normalize(emb, p=2, dim=-1)
        return emb
