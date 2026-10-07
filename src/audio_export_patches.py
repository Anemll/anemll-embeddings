"""Export-time Gemma4 audio patches for Core AI / ANE.

* ``unfold`` → ``index_select`` (convert rejects ``aten.unfold.default``).
* HF ``create_bidirectional_mask`` + ``_convert_4d_mask_to_blocked_5d`` emit
  ``i1`` (``memref<1x1x6x12x84xi1>``, ``memref<1x1x1x1x70xi1>``). ANE cannot
  reshape i1, so preferred-ANE falls through to GPU. Build a float additive
  5D blocked mask instead and add it (no ``masked_fill`` / ``logical_not``).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

# HF Gemma4 audio defaults (EmbeddingGemma 2).
AUDIO_CHUNK = 12
AUDIO_PAST = 12  # attention_context_left - 1
AUDIO_FUTURE = 0
AUDIO_LEFT_WINDOW = 12
AUDIO_RIGHT_WINDOW = 0
AUDIO_INVALID = -1.0e9  # config.attention_invalid_logits_value


def gather_seq_windows(x: torch.Tensor, window: int, step: int) -> torch.Tensor:
    """``x.unfold(1, window, step)`` then ``movedim(-1, 2)`` via ``index_select``.

    ``x`` is ``[B, S, ...]``. Result is ``[B, n_win, window, ...]``.
    Kept for tests; export uses ``slice_seq_windows`` (no i64 gather).
    """
    seq_len = int(x.shape[1])
    window = int(window)
    step = int(step)
    if window <= 0 or step <= 0:
        raise ValueError(f"window={window} step={step}")
    n_win = (seq_len - window) // step + 1
    if n_win <= 0:
        raise ValueError(f"n_win={n_win} for S={seq_len} window={window} step={step}")
    starts = torch.arange(n_win, device=x.device, dtype=torch.long) * step
    offsets = torch.arange(window, device=x.device, dtype=torch.long)
    idx = (starts.unsqueeze(1) + offsets.unsqueeze(0)).reshape(-1)
    gathered = x.index_select(1, idx)
    return gathered.reshape(x.shape[0], n_win, window, *x.shape[2:])


def slice_seq_windows(x: torch.Tensor, window: int, step: int) -> torch.Tensor:
    """Same layout as ``gather_seq_windows`` via static slices + ``stack``.

    No ``index_select`` / ``gather`` — ANE rejected i64 indices on the
    padded K/V window ``[1, 93, H, D]`` (window=24, step=12).
    """
    seq_len = int(x.shape[1])
    window = int(window)
    step = int(step)
    if window <= 0 or step <= 0:
        raise ValueError(f"window={window} step={step}")
    n_win = (seq_len - window) // step + 1
    if n_win <= 0:
        raise ValueError(f"n_win={n_win} for S={seq_len} window={window} step={step}")
    windows = [x[:, b * step : b * step + window] for b in range(n_win)]
    return torch.stack(windows, dim=1)


def _extract_block_context_slices(self, hidden_states: torch.Tensor) -> torch.Tensor:
    hidden_states = F.pad(
        hidden_states,
        (0, 0, 0, 0, self.max_past_horizon, self.max_future_horizon + self.chunk_size - 1),
    )
    hidden_states = slice_seq_windows(hidden_states, self.context_size, self.chunk_size)
    return hidden_states.contiguous()


def _float_sliding_window(
    seq_len: int,
    left_window: int,
    right_window: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """[S, S] float 1=keep. Integer distances via clamp — no i1 compare."""
    pos = torch.arange(int(seq_len), device=device, dtype=dtype)
    dist = pos[:, None] - pos[None, :]
    # Integer step via clamp: dist>=0 & dist<left; dist<0 & -dist<right.
    left_ok = torch.clamp(dist + 1, 0, 1) * torch.clamp(left_window - dist, 0, 1)
    right_ok = torch.clamp(-dist, 0, 1) * torch.clamp(dist + right_window, 0, 1)
    return torch.clamp(left_ok + right_ok, 0, 1)


def blocked_additive_attention_mask(
    keep: torch.Tensor,
    *,
    chunk_size: int = AUDIO_CHUNK,
    max_past_horizon: int = AUDIO_PAST,
    max_future_horizon: int = AUDIO_FUTURE,
    left_window: int = AUDIO_LEFT_WINDOW,
    right_window: int = AUDIO_RIGHT_WINDOW,
    invalid: float = AUDIO_INVALID,
) -> torch.Tensor:
    """2D keep ``[B, S]`` → 5D additive ``[B, 1, n_blocks, chunk, context]``.

    Matches HF ``create_bidirectional_mask`` (key pad + sliding window) then
    ``_convert_4d_mask_to_blocked_5d``, but stays float so ANE can reshape.
    """
    if keep.ndim != 2:
        raise ValueError(f"keep rank {keep.ndim} (want [B, S])")
    batch, seq_len = int(keep.shape[0]), int(keep.shape[1])
    dtype = keep.dtype if keep.is_floating_point() else torch.float32
    keep = keep.to(dtype=dtype)
    window = _float_sliding_window(
        seq_len, int(left_window), int(right_window), device=keep.device, dtype=dtype
    )
    # HF padding_mask_function is key-only.
    keep4 = window.view(1, 1, seq_len, seq_len) * keep[:, None, None, :]

    chunk_size = int(chunk_size)
    max_past_horizon = int(max_past_horizon)
    max_future_horizon = int(max_future_horizon)
    num_blocks = (seq_len + chunk_size - 1) // chunk_size
    padded_seq = num_blocks * chunk_size
    pad_amount = padded_seq - seq_len
    keep4 = F.pad(keep4, (0, pad_amount, 0, pad_amount), value=0.0)
    keep5 = keep4.reshape(batch, 1, num_blocks, chunk_size, padded_seq)
    keep5 = F.pad(keep5, (max_past_horizon, max_future_horizon), value=0.0)
    context = chunk_size + max_past_horizon + max_future_horizon
    # Static slices — no gather(-1) / i64 indices. Block b takes last-dim
    # [b*chunk : b*chunk+context] of the padded 84-wide key axis.
    windows = [
        keep5[:, :, b : b + 1, :, b * chunk_size : b * chunk_size + context]
        for b in range(num_blocks)
    ]
    keep5 = torch.cat(windows, dim=2)
    return (1.0 - keep5) * float(invalid)


def _audio_attn_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, seq_length, _ = hidden_states.shape
    hidden_shape = (batch_size, seq_length, self.num_heads, self.head_dim)

    query_states = self.q_proj(hidden_states).float().view(hidden_shape)
    key_states = self.k_proj(hidden_states).float().view(hidden_shape)
    value_states = self.v_proj(hidden_states).float().view(hidden_shape)

    query_states = query_states * self.q_scale * F.softplus(self.per_dim_scale)
    key_states = key_states * self.k_scale

    query_states = self._convert_to_block(query_states)
    key_states = self._extract_block_context(key_states)
    value_states = self._extract_block_context(value_states)
    num_blocks = query_states.shape[1]

    relative_key_states = self.relative_k_proj(position_embeddings)
    relative_key_states = relative_key_states.view(-1, self.num_heads, self.head_dim)
    relative_key_states = relative_key_states.to(dtype=query_states.dtype)

    queries = query_states.permute(0, 3, 1, 2, 4)
    matrix_ac = queries @ key_states.permute(0, 3, 1, 4, 2)

    queries_flat = queries.reshape(batch_size, self.num_heads, -1, self.head_dim)
    matrix_bd = queries_flat @ relative_key_states.permute(1, 2, 0)
    matrix_bd = matrix_bd.reshape(batch_size, self.num_heads, num_blocks, self.chunk_size, -1)
    matrix_bd = self._rel_shift(matrix_bd)

    attn_weights = matrix_ac + matrix_bd
    attn_weights = attn_weights / self.softcap
    attn_weights = torch.tanh(attn_weights)
    attn_weights = attn_weights * self.softcap

    if attention_mask is not None:
        # Float additive 5D (ANE-legal). Do not logical_not / masked_fill i1.
        attn_weights = attn_weights + attention_mask.to(dtype=attn_weights.dtype)

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value_states.dtype)
    attn_output = attn_weights @ value_states.permute(0, 3, 1, 2, 4)
    attn_output = attn_output.permute(0, 2, 3, 1, 4).reshape(
        batch_size, num_blocks * self.chunk_size, -1
    )
    attn_output = attn_output[:, :seq_length].contiguous()
    attn_output = self.post(attn_output.to(hidden_states.dtype))
    return attn_output, attn_weights


def _audio_model_forward(
    self,
    input_features: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    **kwargs: Any,
):
    import transformers.models.gemma4.modeling_gemma4 as g4

    hidden_states, output_mask = self.subsample_conv_projection(
        input_features, attention_mask
    )
    position_embeddings = self.rel_pos_enc(hidden_states)
    keep = output_mask
    if keep is None:
        keep = torch.ones(
            hidden_states.shape[0],
            hidden_states.shape[1],
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
    else:
        keep = keep.to(dtype=hidden_states.dtype)
    attn = blocked_additive_attention_mask(
        keep,
        chunk_size=int(self.config.attention_chunk_size),
        max_past_horizon=int(self.config.attention_context_left) - 1,
        max_future_horizon=int(self.config.attention_context_right),
        left_window=int(self.config.attention_context_left) - 1,
        right_window=int(self.config.attention_context_right),
        invalid=float(self.config.attention_invalid_logits_value),
    )
    for encoder_layer in self.layers[: self.config.num_hidden_layers]:
        hidden_states = encoder_layer(
            hidden_states,
            attention_mask=attn,
            position_embeddings=position_embeddings,
            **kwargs,
        )
    hidden_states = self.output_proj(hidden_states)
    # Drop output_mask live-out (constin_liveout_dummy_pool).
    return g4.Gemma4AudioModelOutput(last_hidden_state=hidden_states, attention_mask=None)


def apply_audio_unfold_patch() -> dict[str, int]:
    """Monkey-patch ``Gemma4AudioAttention._extract_block_context`` for export."""
    import transformers.models.gemma4.modeling_gemma4 as g4

    g4.Gemma4AudioAttention._extract_block_context = _extract_block_context_slices
    return {"patched": 1, "window_op": "static_slice"}


def apply_audio_ane_mask_patch() -> dict[str, Any]:
    """Replace i1 blocked mask + ``masked_fill`` with float additive 5D."""
    import transformers.models.gemma4.modeling_gemma4 as g4

    g4.Gemma4AudioAttention.forward = _audio_attn_forward
    g4.Gemma4AudioModel.forward = _audio_model_forward
    return {
        "patched": 1,
        "mask": "float_blocked_additive_5d",
        "chunk": AUDIO_CHUNK,
        "context": AUDIO_CHUNK + AUDIO_PAST + AUDIO_FUTURE,
    }


def apply_audio_export_patches() -> dict[str, Any]:
    """Unfold gather + float blocked mask (ANE-legal, no i1 reshape)."""
    unfold = apply_audio_unfold_patch()
    mask = apply_audio_ane_mask_patch()
    return {"unfold": unfold, "mask": mask}
