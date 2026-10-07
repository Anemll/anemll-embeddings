"""Export-time vision patches: pos embed + RoPE without GPU concat_slice.

``F.embedding`` requires Int/Long, so the wrapper used to widen
``[1,2520,2]`` si16 → i64. That i64 gather is not ANE-legal I/O.
Look up the 2-D table with a float one-hot matmul instead.

``GPU_region_0`` leftover after f16 pos I/O was ``concat_slice`` from
RoPE ``torch.cat([h,h,w,w])`` / ``torch.split``+``cat``. Rewrite those
as expand+reshape and last-dim matmul join.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .audio_export_patches import swap_last_two, swap_mid_dims


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


def _recomposition_frequencies_ane(self, freq: torch.Tensor) -> torch.Tensor:
    """``[h, h, w, w]`` last-dim layout via expand+reshape — no ``cat`` slices."""
    if freq.ndim != 4 or int(freq.shape[2]) != 2:
        freq_h, freq_w = freq[:, :, 0], freq[:, :, 1]
        return torch.cat([freq_h, freq_h, freq_w, freq_w], dim=-1)
    batch, seq, _, width = freq.shape
    repeated = freq.unsqueeze(-2).expand(batch, seq, 2, 2, width)
    return repeated.contiguous().reshape(batch, seq, 4 * width)


_ORIG_APPLY_ROPE = None


def _apply_multidimensional_rope_ane(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    unsqueeze_dim: int = 2,
) -> torch.Tensor:
    """2-D RoPE without ``split``+``cat`` (those lowered as GPU ``concat_slice``)."""
    import transformers.models.gemma4.modeling_gemma4 as g4

    ndim = int(position_ids.shape[-1])
    channels = int(x.shape[-1])
    per = 2 * (channels // (2 * ndim))
    if ndim != 2 or per * ndim != channels:
        if _ORIG_APPLY_ROPE is None:
            raise RuntimeError(f"RoPE fallback missing ndim={ndim} channels={channels}")
        return _ORIG_APPLY_ROPE(
            x, cos, sin, position_ids, unsqueeze_dim=unsqueeze_dim
        )
    eye = torch.eye(per, device=x.device, dtype=x.dtype)
    zeros = torch.zeros(per, per, device=x.device, dtype=x.dtype)
    take0 = torch.cat([eye, zeros], dim=0)
    take1 = torch.cat([zeros, eye], dim=0)
    y0 = g4.apply_rotary_pos_emb(
        x=x @ take0, cos=cos @ take0, sin=sin @ take0, unsqueeze_dim=unsqueeze_dim
    )
    y1 = g4.apply_rotary_pos_emb(
        x=x @ take1, cos=cos @ take1, sin=sin @ take1, unsqueeze_dim=unsqueeze_dim
    )
    return y0 @ torch.cat([eye, zeros], dim=1) + y1 @ torch.cat([zeros, eye], dim=1)


def rotate_half_matmul(x: torch.Tensor) -> torch.Tensor:
    """``cat((-x2, x1))`` via a ``[D, D]`` matmul — no last-dim slice+cat."""
    dim = int(x.shape[-1])
    if dim % 2:
        raise ValueError(f"rotate_half last dim {dim} is not even")
    half = dim // 2
    eye = torch.eye(half, device=x.device, dtype=x.dtype)
    zeros = torch.zeros(half, half, device=x.device, dtype=x.dtype)
    rot = torch.cat(
        [torch.cat([zeros, eye], dim=1), torch.cat([-eye, zeros], dim=1)], dim=0
    )
    return x @ rot


def _vision_attn_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: torch.Tensor = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Heads-first 4-D attention — no ``transpose(1,2)`` into ``sdpa``.

    Leftover ``GPU_region_0`` unnamed ``op`` was ``q/k/v.transpose(1,2)`` on
    ``[1,2520,12,64]``. Bake heads-first via one-hot swap, then matmul.
    """
    batch, seq_len, _ = hidden_states.shape
    head_dim = int(self.head_dim)
    n_heads = int(self.config.num_attention_heads)
    n_kv = int(self.config.num_key_value_heads)
    hidden_shape = (batch, seq_len, n_heads, head_dim)
    kv_shape = (batch, seq_len, n_kv, head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape))
    key_states = self.k_norm(self.k_proj(hidden_states).view(kv_shape))
    value_states = self.v_norm(self.v_proj(hidden_states).view(kv_shape))
    if position_embeddings is not None:
        cos, sin = position_embeddings
        query_states = _apply_multidimensional_rope_ane(
            query_states, cos, sin, position_ids, unsqueeze_dim=2
        )
        key_states = _apply_multidimensional_rope_ane(
            key_states, cos, sin, position_ids, unsqueeze_dim=2
        )
    query_states = swap_mid_dims(query_states)
    key_states = swap_mid_dims(key_states)
    value_states = swap_mid_dims(value_states)
    if n_kv != n_heads:
        repeats = n_heads // n_kv
        key_states = key_states.repeat_interleave(repeats, dim=1)
        value_states = value_states.repeat_interleave(repeats, dim=1)
    scale = float(self.scaling)
    attn_weights = (query_states @ swap_last_two(key_states)) * scale
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = F.softmax(attn_weights, dim=-1)
    attn_output = attn_weights @ value_states
    attn_output = swap_mid_dims(attn_output).reshape(batch, seq_len, n_heads * head_dim)
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def apply_vision_ane_embed_patch() -> dict[str, Any]:
    """Pos one-hot + RoPE + heads-first attn without gather / concat_slice / transpose."""
    import transformers.models.gemma4.modeling_gemma4 as g4

    global _ORIG_APPLY_ROPE
    g4.Gemma4VisionPatchEmbedder._position_embeddings = _position_embeddings_ane
    g4.Gemma4VisionRotaryEmbedding.recomposition_frequencies = _recomposition_frequencies_ane
    if _ORIG_APPLY_ROPE is None:
        _ORIG_APPLY_ROPE = g4.apply_multidimensional_rope
    g4.apply_multidimensional_rope = _apply_multidimensional_rope_ane
    g4.rotate_half = rotate_half_matmul
    g4.Gemma4VisionAttention.forward = _vision_attn_forward
    return {
        "patched": 1,
        "pos_embed": "float_onehot_matmul",
        "rope": "expand_reshape_no_concat_slice",
        "rotate_half": "matmul",
        "attn": "heads_first_4d_no_transpose_sdpa",
    }
