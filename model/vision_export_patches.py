"""Export-time vision patches: pos embed + RoPE without GPU concat_slice.

``F.embedding`` requires Int/Long, so the wrapper used to widen
``[1,2520,2]`` si16 → i64. That i64 gather is not ANE-legal I/O.
Look up the 2-D table with a float one-hot matmul instead.

``GPU_region_0`` leftover after f16 pos I/O was ``concat_slice`` from
RoPE ``torch.cat([h,h,w,w])`` / ``torch.split``+``cat``. Rewrite those
as expand+reshape and last-dim matmul join.
"""

from __future__ import annotations

import functools
from typing import Any

import numpy as np
import torch

from model.trace_patches import softmax_unfused

# Longest image side in patches for 280 soft tokens with a 3×3 pooler.
VISION_MAX_SIDE = 280 * 3


def embedding_from_int_indices(idx: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """``idx [...]`` (any integer, including si16) × ``table [V, D]`` → ``[..., D]``.

    Exact for integer coordinates: one-hot via ``clamp(1-|idx-slot|)``, then matmul.
    No ``aten.embedding`` / i64 indices.
    """
    vocab = int(table.shape[0])
    dtype = table.dtype
    slots = torch.arange(vocab, device=idx.device, dtype=dtype)
    idx_f = idx.to(dtype=dtype).clamp(min=0)
    # relu, not clamp(0, 1): 1 - |d| <= 1 anyway, and on the ANE (fp16) a
    # clamp feeding only this matmul gave wrong sums (x_emb absmax 95 vs 0.35).
    onehot = torch.relu(1.0 - (idx_f.unsqueeze(-1) - slots).abs())
    return onehot @ table


def _position_embeddings_ane(
    self, pixel_position_ids: torch.Tensor, padding_positions: torch.Tensor
) -> torch.Tensor:
    # The processor caps a side at (2520 // 3²) · 3 = 840 patches, so rows
    # past 840 are never hit. The full 10240-row one-hot placed on the GPU.
    table = self.position_embedding_table[:, :VISION_MAX_SIDE]
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


@functools.cache
def _chunk_rotate_half(channels: int, per: int) -> np.ndarray:
    """``[C, C]`` matrix R with ``x @ R`` = rotate_half applied to each ``per`` chunk."""
    rot = np.zeros((channels, channels), dtype=np.float32)
    half = per // 2
    for start in range(0, channels, per):
        for i in range(half):
            rot[start + half + i, start + i] = -1.0  # out[i] = -x[half + i]
            rot[start + i, start + half + i] = 1.0  # out[half + i] = x[i]
    return rot


def _apply_multidimensional_rope_ane(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    unsqueeze_dim: int = 2,
) -> torch.Tensor:
    """2-D RoPE without ``split``+``cat`` (those lowered as GPU ``concat_slice``)."""
    ndim = int(position_ids.shape[-1])
    channels = int(x.shape[-1])
    per = 2 * (channels // (2 * ndim))
    if ndim != 2 or per * ndim != channels:
        if _ORIG_APPLY_ROPE is None:
            raise RuntimeError(f"RoPE fallback missing ndim={ndim} channels={channels}")
        return _ORIG_APPLY_ROPE(
            x, cos, sin, position_ids, unsqueeze_dim=unsqueeze_dim
        )
    # HF rotates each per-dim chunk with its own slice of cos/sin, and those
    # slices already sit where the chunk does. So the whole thing is
    # x * cos + rotate_half_per_chunk(x) * sin, with the rotation one constant
    # [C, C] matmul (was six small matmuls per call; ~1 ms per layer on the ANE).
    rot = torch.from_numpy(_chunk_rotate_half(channels, per)).to(device=x.device, dtype=x.dtype)
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return x * cos + (x @ rot) * sin


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


# Softmax tiles: 6 groups of 2 heads x 2 blocks of 1260 query rows. One
# [1, 12, S, S] score tensor per layer does not fit on chip, so every softmax
# pass round-trips DRAM; tiling it matters. Measured on the ANE (4 layers):
# 1 group x 12 blocks 90.0 ms, 6 x 2 83.7, 6 x 1 84.2, 2 x 6 85.0, 4 x 6 85.1,
# 12 x 1 94.5. Earlier, 18 blocks (140 rows) computed wrong scores on the ANE,
# so re-check accuracy whenever these change.
VISION_ATTN_QUERY_BLOCKS = 2
VISION_ATTN_HEAD_GROUPS = 6


def _vision_attn_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: torch.Tensor = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """All heads at once ``[B, H, S, D]``, attention in query blocks (fp16, ANE).

    Q, K and V share the ``[B, H, *, D]`` layout. The softmax is spelled out
    (``softmax_unfused``): a plain ``softmax`` between the two matmuls is fused
    into ``mps_spi.sdpa``, which the macOS 27.2 ANE pre-check rejects.
    Softmax rows are independent, so the blocks are exact.
    """
    batch, seq_len, _ = hidden_states.shape
    head_dim = int(self.head_dim)
    n_heads = int(self.config.num_attention_heads)
    n_kv = int(self.config.num_key_value_heads)
    if n_kv != n_heads:
        raise ValueError(f"vision attention expects n_kv == n_heads, got {n_kv}/{n_heads}")
    query_states = self.q_norm(self.q_proj(hidden_states).view(batch, seq_len, n_heads, head_dim))
    key_states = self.k_norm(self.k_proj(hidden_states).view(batch, seq_len, n_kv, head_dim))
    value_states = self.v_norm(self.v_proj(hidden_states).view(batch, seq_len, n_kv, head_dim))
    if position_embeddings is not None:
        cos, sin = position_embeddings
        query_states = _apply_multidimensional_rope_ane(
            query_states, cos, sin, position_ids, unsqueeze_dim=2
        )
        key_states = _apply_multidimensional_rope_ane(
            key_states, cos, sin, position_ids, unsqueeze_dim=2
        )
    scale = float(self.scaling)
    hidden = n_heads * head_dim
    attn_weights = query_states.new_zeros(batch, seq_len, seq_len)
    mask = None
    if attention_mask is not None:
        mask = attention_mask.reshape(batch, 1, -1, seq_len)
    q = query_states.transpose(1, 2)
    key_t = key_states.transpose(1, 2).transpose(-1, -2)
    v = value_states.transpose(1, 2)
    n_blocks = VISION_ATTN_QUERY_BLOCKS if seq_len % VISION_ATTN_QUERY_BLOCKS == 0 else 1
    n_groups = VISION_ATTN_HEAD_GROUPS if n_heads % VISION_ATTN_HEAD_GROUPS == 0 else 1
    rows = seq_len // n_blocks
    width = n_heads // n_groups
    groups = []
    for g in range(n_groups):
        heads = slice(g * width, (g + 1) * width)
        q_g, key_t_g, v_g = q[:, heads], key_t[:, heads], v[:, heads]
        parts = []
        for b in range(n_blocks):
            scores = torch.matmul(q_g[:, :, b * rows : (b + 1) * rows], key_t_g) * scale
            if mask is not None:
                scores = scores + mask
            parts.append(torch.matmul(softmax_unfused(scores), v_g))
        groups.append(torch.cat(parts, dim=2) if n_blocks > 1 else parts[0])
    out = torch.cat(groups, dim=1) if n_groups > 1 else groups[0]
    attn_output = out.transpose(1, 2).reshape(batch, seq_len, hidden)
    return self.o_proj(attn_output), attn_weights


def _rms_norm_fp16_safe(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """``x * (mean(x²) + eps)^-½`` computed on ``x / max|x|`` (forge ``rms_robust``).

    Vision activations reach ~900, so ``x²`` overflows fp16 (65504) and the
    norm returns 0. Squares of ``x / m`` stay in [0, 1]; ``eps / m²`` keeps
    eps exact, and zero rows stay zero (``m`` is clamped to 1e-3).
    """
    # reciprocal + multiply, not divide: the ANE rejects a divide whose
    # operands have different ranks (seen as 320x512 / 1x320x1 in text_embeds).
    inv = torch.reciprocal(hidden_states.abs().amax(-1, keepdim=True).clamp_min(1e-3))
    xs = hidden_states * inv
    return xs * torch.rsqrt((xs * xs).mean(-1, keepdim=True) + self.eps * inv * inv)


def apply_fp16_safe_rms_norm_patch() -> list[str]:
    """Swap ``_norm`` on every Gemma RMSNorm class an export can reach.

    ``embed_vision`` / ``embed_audio`` use the EmbeddingGemma2 class, the
    towers use Gemma4's; both have the same ``_norm``.
    """
    import transformers.models.embedding_gemma2.modeling_embedding_gemma2 as eg2
    import transformers.models.gemma4.modeling_gemma4 as g4

    classes = (g4.Gemma4RMSNorm, eg2.EmbeddingGemma2RMSNorm)
    for cls in classes:
        cls._norm = _rms_norm_fp16_safe
    return [cls.__name__ for cls in classes]


def apply_vision_ane_embed_patch() -> dict[str, Any]:
    """Pos one-hot + RoPE + heads-first attn without gather / concat_slice / transpose."""
    import transformers.models.gemma4.modeling_gemma4 as g4

    global _ORIG_APPLY_ROPE
    g4.Gemma4VisionPatchEmbedder._position_embeddings = _position_embeddings_ane
    if _ORIG_APPLY_ROPE is None:
        _ORIG_APPLY_ROPE = g4.apply_multidimensional_rope
    g4.Gemma4VisionRotaryEmbedding.recomposition_frequencies = _recomposition_frequencies_ane
    g4.apply_multidimensional_rope = _apply_multidimensional_rope_ane
    g4.rotate_half = rotate_half_matmul
    g4.Gemma4VisionAttention.forward = _vision_attn_forward
    norms = apply_fp16_safe_rms_norm_patch()
    return {
        "rms_norm": f"max_scaled_fp16_safe:{','.join(norms)}",
        "patched": 1,
        "pos_embed": "float_onehot_matmul",
        "rope": "expand_reshape_no_concat_slice",
        "rotate_half": "matmul",
        "attn": f"head_groups_{VISION_ATTN_HEAD_GROUPS}_query_blocks_{VISION_ATTN_QUERY_BLOCKS}",
    }
