"""T3: Trace-ready EmbeddingGemma 2 inference wrapper.

Graph (host tokenization stays outside — Core ML I/O is ids/mask → vector)::

    input_ids [B, S], attention_mask [B, S]
      → text backbone last_hidden_state [B, S, 512]
      → masked mean pool → [B, 512]
      → Dense 512→768 (bias-free) → [B, 768]
      → optional L2 normalize

Pool-then-project matches the upstream note that projecting per-token then
mean-pooling is mathematically equivalent for a bias-free linear map, and
keeps the export shape ladder simple for later torch.jit.trace / coremltools.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch import nn

from .load_text_model import get_text_backbone


def masked_mean_pool(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Mask-aware mean over sequence dim. Empty masks → zeros (clamp denom)."""
    # hidden: [B, S, H], mask: [B, S]
    mask = attention_mask.to(dtype=hidden_states.dtype).unsqueeze(-1)
    summed = (hidden_states * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return summed / denom


class MaskedMeanPool(nn.Module):
    """nn.Module wrapper around :func:`masked_mean_pool` for tracing."""

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return masked_mean_pool(hidden_states, attention_mask)


class EmbeddingGemma2Wrapper(nn.Module):
    """Text tower + mask mean-pool + 512→768 projection (+ optional L2).

    On construction, the backbone's in-graph ``embedding_projection`` is moved
    *after* pooling (replaced with ``Identity`` on the text model) so the
    traced graph matches the Core ML export plan.
    """

    def __init__(
        self,
        text_model: nn.Module,
        *,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        if not hasattr(text_model, "embedding_projection"):
            raise TypeError(
                "text_model must own embedding_projection "
                f"(EmbeddingGemma2TextModel); got {type(text_model)}"
            )
        proj = text_model.embedding_projection
        if not isinstance(proj, nn.Linear):
            raise TypeError(f"embedding_projection must be nn.Linear; got {type(proj)}")
        if proj.bias is not None:
            raise ValueError(
                "Expected bias-free 512→768 projection (pool/project reorder requires it)"
            )
        # Detach projection for pool-then-project; backbone now emits [B,S,512].
        text_model.embedding_projection = nn.Identity()
        self.text_model = text_model
        self.projection = proj  # registers as submodule; weight [768, 512]
        self.pool = MaskedMeanPool()
        self.normalize = bool(normalize)
        self.hidden_size = int(proj.in_features)  # 512
        self.embedding_dim = int(proj.out_features)  # 768

    @classmethod
    def from_sentence_transformer(
        cls,
        st_model: SentenceTransformer,
        *,
        normalize: bool = True,
    ) -> EmbeddingGemma2Wrapper:
        return cls(get_text_backbone(st_model), normalize=normalize)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state  # [B, S, 512]
        pooled = self.pool(hidden, attention_mask)  # [B, 512]
        emb = self.projection(pooled)  # [B, 768]
        if self.normalize:
            emb = F.normalize(emb, p=2, dim=-1)
        return emb

    def encode_tokenized(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Eval-mode forward returning float32 CPU numpy-friendly tensor."""
        was_training = self.training
        self.eval()
        with torch.inference_mode():
            emb = self.forward(input_ids, attention_mask)
        if was_training:
            self.train()
        return emb


def tokenize_with_st_prompt(
    st_model: SentenceTransformer,
    text: str,
    prompt_name: str | None,
    *,
    device: str | torch.device | None = None,
    max_length: int | None = None,
) -> dict[str, torch.Tensor]:
    """Host-side tokenization matching ST ``prompt_name`` prefixes."""
    if prompt_name:
        prompts: dict[str, Any] = getattr(st_model, "prompts", {}) or {}
        if prompt_name not in prompts:
            raise KeyError(
                f"Unknown prompt_name={prompt_name!r}; "
                f"known={sorted(prompts.keys())[:12]}…"
            )
        text = prompts[prompt_name] + text
    tokenizer = st_model.tokenizer
    kwargs: dict[str, Any] = {
        "return_tensors": "pt",
        "padding": True,
        "truncation": True,
    }
    if max_length is not None:
        kwargs["max_length"] = max_length
    batch = tokenizer(text, **kwargs)
    if device is not None:
        batch = {k: v.to(device) for k, v in batch.items()}
    return batch
