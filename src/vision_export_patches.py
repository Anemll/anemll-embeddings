"""Export-time vision patches: pos embed without si16→i64 gather.

``F.embedding`` requires Int/Long, so the wrapper used to widen
``[1,2520,2]`` si16 → i64. That i64 gather is not ANE-legal I/O.
Look up the 2-D table with a float one-hot matmul instead.
"""

from __future__ import annotations

from typing import Any

import torch


def embedding_from_int_indices(idx: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """``idx [...]`` (any integer, including si16) × ``table [V, D]`` → ``[..., D]``.

    Exact for integer coordinates: one-hot via ``clamp(1-|idx-slot|)``, then matmul.
    No ``aten.embedding`` / i64 indices.
    """
    vocab = int(table.shape[0])
    dtype = table.dtype
    slots = torch.arange(vocab, device=idx.device, dtype=dtype)
    idx_f = idx.to(dtype=dtype).clamp(min=0)
    onehot = torch.clamp(1.0 - (idx_f.unsqueeze(-1) - slots).abs(), 0, 1)
    return onehot @ table


def _position_embeddings_ane(
    self, pixel_position_ids: torch.Tensor, padding_positions: torch.Tensor
) -> torch.Tensor:
    table = self.position_embedding_table
    x_emb = embedding_from_int_indices(pixel_position_ids[..., 0], table[0])
    y_emb = embedding_from_int_indices(pixel_position_ids[..., 1], table[1])
    position_embeddings = x_emb + y_emb
    if padding_positions.dtype == torch.bool:
        keep = (~padding_positions).to(dtype=position_embeddings.dtype)
    else:
        keep = (1.0 - padding_positions.to(dtype=position_embeddings.dtype)).clamp(0, 1)
    return position_embeddings * keep.unsqueeze(-1)


def apply_vision_ane_embed_patch() -> dict[str, Any]:
    """Replace ``F.embedding`` pos lookup with float one-hot matmul."""
    import transformers.models.gemma4.modeling_gemma4 as g4

    g4.Gemma4VisionPatchEmbedder._position_embeddings = _position_embeddings_ane
    return {"patched": 1, "pos_embed": "float_onehot_matmul"}
