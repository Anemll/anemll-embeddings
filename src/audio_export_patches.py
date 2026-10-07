"""Export-time Gemma4 audio patches for Core AI / ANE.

* ``unfold`` → ``index_select`` (convert rejects ``aten.unfold.default``).
* HF ``create_bidirectional_mask`` + ``_convert_4d_mask_to_blocked_5d`` emit
  ``i1``. ANE cannot reshape i1, so preferred-ANE falls through to GPU.
  Build a float additive 4-D blocked mask instead (no ``masked_fill`` /
  ``logical_not`` / 5-D ``permute``). Package I/O is 4-D NCHW expandDims.
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


def window_onehot(
    seq_len: int,
    window: int,
    step: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """``[n_win, window, seq_len]`` one-hot for overlapping windows."""
    seq_len = int(seq_len)
    window = int(window)
    step = int(step)
    if window <= 0 or step <= 0:
        raise ValueError(f"window={window} step={step}")
    n_win = (seq_len - window) // step + 1
    if n_win <= 0:
        raise ValueError(f"n_win={n_win} for S={seq_len} window={window} step={step}")
    slots = torch.arange(seq_len, device=device, dtype=dtype)
    starts = torch.arange(n_win, device=device, dtype=dtype) * float(step)
    offsets = torch.arange(window, device=device, dtype=dtype)
    idx = starts.unsqueeze(1) + offsets.unsqueeze(0)
    return torch.clamp(1.0 - (idx.unsqueeze(-1) - slots).abs(), 0, 1)


def swap_mid_dims(x: torch.Tensor) -> torch.Tensor:
    """``[N, A, B, D] → [N, B, A, D]`` via left one-hot matmul (no transpose).

    Audio leftover ``GPU_region_0`` was ``permute(0,3,1,2,4)`` on
    ``[B, n, c, H, D] ↔ [B, H, n, c, D]``. Swap heads/seq in 4-D instead.
    Left-multiply so Core AI does not insert ``reshape→transpose``.
    """
    if x.ndim != 4:
        raise ValueError(f"swap_mid_dims rank {x.ndim} (want [N, A, B, D])")
    batch, dim_a, dim_b, last = (int(s) for s in x.shape)
    src_idx = [a * dim_b + b for b in range(dim_b) for a in range(dim_a)]
    src = torch.tensor(src_idx, device=x.device, dtype=x.dtype)
    slots = torch.arange(dim_a * dim_b, device=x.device, dtype=x.dtype)
    onehot = torch.clamp(1.0 - (src.unsqueeze(-1) - slots).abs(), 0, 1)
    flat = x.reshape(batch, dim_a * dim_b, last)
    return (onehot @ flat).reshape(batch, dim_b, dim_a, last)


def nchw_to_nhwc(x: torch.Tensor) -> torch.Tensor:
    """``[B, C, H, W] → [B, H, W, C]`` via one-hot swaps (no ``permute``)."""
    if x.ndim != 4:
        raise ValueError(f"nchw_to_nhwc rank {x.ndim}")
    batch, channels, height, width = (int(s) for s in x.shape)
    mid = swap_mid_dims(x)
    return swap_mid_dims(mid.reshape(batch * height, channels, width, 1)).reshape(
        batch, height, width, channels
    )


def nhwc_to_nchw(x: torch.Tensor) -> torch.Tensor:
    """``[B, H, W, C] → [B, C, H, W]`` via one-hot swaps (no ``permute``)."""
    if x.ndim != 4:
        raise ValueError(f"nhwc_to_nchw rank {x.ndim}")
    batch, height, width, channels = (int(s) for s in x.shape)
    mid = swap_mid_dims(x.reshape(batch * height, width, channels, 1)).reshape(
        batch, height, channels, width
    )
    return swap_mid_dims(mid)


def swap_last_two(x: torch.Tensor) -> torch.Tensor:
    """``[..., A, B] → [..., B, A]`` via one-hot (no ``transpose`` / ``permute``)."""
    if x.ndim < 2:
        raise ValueError(f"swap_last_two rank {x.ndim}")
    lead = x.shape[:-2]
    dim_a, dim_b = int(x.shape[-2]), int(x.shape[-1])
    flat = x.reshape(int(x.numel() // (dim_a * dim_b)), dim_a, dim_b, 1)
    return swap_mid_dims(flat).reshape(*lead, dim_b, dim_a)


def slice_seq_windows(x: torch.Tensor, window: int, step: int) -> torch.Tensor:
    """Same layout as ``gather_seq_windows`` via one-hot ``matmul``.

    Slice+``stack`` cleared ``index_select`` but left a GPU
    ``reshape→permute`` / ``strided_slice`` I/O (not squeeze/expandDims).
    """
    seq_len = int(x.shape[1])
    window = int(window)
    step = int(step)
    onehot = window_onehot(seq_len, window, step, device=x.device, dtype=x.dtype)
    n_win = int(onehot.shape[0])
    flat = x.reshape(x.shape[0], seq_len, -1)
    gathered = onehot.reshape(-1, seq_len) @ flat
    return gathered.reshape(x.shape[0], n_win, window, *x.shape[2:])


def _rel_shift_matmul(self, x: torch.Tensor) -> torch.Tensor:
    """Transformer-XL rel-shift without ``view``+``strided_slice`` (GPU I/O).

    HF pads last-dim to ``context+1``, flattens, keeps the first
    ``chunk*context`` values, then views back. That lowers as
    ``reshape->strided_slice->reshape``. Keep the prefix via matmul.
    """
    batch, heads, num_blocks, block_size, pos_len = x.shape
    context = int(self.context_size)
    x = F.pad(x, (0, context + 1 - int(pos_len)))
    flat = x.reshape(batch, heads, num_blocks, block_size * (context + 1))
    keep = block_size * context
    total = block_size * (context + 1)
    eye = torch.eye(keep, device=x.device, dtype=x.dtype)
    zeros = torch.zeros(total - keep, keep, device=x.device, dtype=x.dtype)
    select = torch.cat([eye, zeros], dim=0)
    return (flat @ select).reshape(batch, heads, num_blocks, block_size, context)


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
    """2D keep ``[B, S]`` → 4D additive ``[B, n_blocks, chunk, context]``.

    Matches HF ``create_bidirectional_mask`` (key pad + sliding window) then
    ``_convert_4d_mask_to_blocked_5d``, but stays float 4-D (no 5-D permute).
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
    keep4 = keep4.reshape(batch, num_blocks, chunk_size, padded_seq)
    keep4 = F.pad(keep4, (max_past_horizon, max_future_horizon), value=0.0)
    context = chunk_size + max_past_horizon + max_future_horizon
    # One-hot matmul on the key axis — no gather / 5-D layout I/O.
    padded_keys = int(keep4.shape[-1])
    left = keep4.reshape(batch * num_blocks, chunk_size, padded_keys)
    # [n, K, ctx] one-hot so left @ right needs no activation transpose.
    slots = torch.arange(padded_keys, device=keep4.device, dtype=keep4.dtype)
    starts = torch.arange(num_blocks, device=keep4.device, dtype=keep4.dtype) * float(
        chunk_size
    )
    offsets = torch.arange(context, device=keep4.device, dtype=keep4.dtype)
    idx = starts.unsqueeze(1) + offsets.unsqueeze(0)
    onehot_kt = torch.clamp(1.0 - (slots.view(1, -1, 1) - idx.unsqueeze(1)).abs(), 0, 1)
    right = onehot_kt.unsqueeze(0).expand(batch, -1, -1, -1).reshape(
        batch * num_blocks, padded_keys, context
    )
    keep4 = (left @ right).reshape(batch, num_blocks, chunk_size, context)
    return (1.0 - keep4) * float(invalid)


def _audio_attn_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, seq_length, _ = hidden_states.shape
    num_heads = int(self.num_heads)
    head_dim = int(self.head_dim)
    chunk = int(self.chunk_size)
    context = int(self.context_size)
    hidden_shape = (batch_size, seq_length, num_heads, head_dim)

    # Stay in hidden dtype. ``.float()`` made ANE I/O see f32 ``value``
    # (Incompatible element type: not fp16/si8/si16) at the GPU island.
    query_states = self.q_proj(hidden_states).view(hidden_shape)
    key_states = self.k_proj(hidden_states).view(hidden_shape)
    value_states = self.v_proj(hidden_states).view(hidden_shape)

    query_states = query_states * self.q_scale * F.softplus(self.per_dim_scale)
    key_states = key_states * self.k_scale

    # Heads-first 4-D only: [B, S, H, D] → [B, H, S, D] → [BH, n, c, D].
    # Never materialize [B, H, n, c, D] (that 5-D reshape became permute).
    query_states = swap_mid_dims(query_states)
    key_states = swap_mid_dims(key_states)
    value_states = swap_mid_dims(value_states)

    num_blocks = (seq_length + chunk - 1) // chunk
    q_pad = num_blocks * chunk - seq_length
    query_states = F.pad(query_states, (0, 0, 0, q_pad))
    q4 = query_states.reshape(batch_size * num_heads, num_blocks, chunk, head_dim)

    key_states = F.pad(
        key_states,
        (0, 0, self.max_past_horizon, self.max_future_horizon + chunk - 1),
    )
    value_states = F.pad(
        value_states,
        (0, 0, self.max_past_horizon, self.max_future_horizon + chunk - 1),
    )
    pad_len = int(key_states.shape[2])
    k4 = slice_seq_windows(
        key_states.reshape(batch_size * num_heads, pad_len, head_dim), context, chunk
    )
    v4 = slice_seq_windows(
        value_states.reshape(batch_size * num_heads, pad_len, head_dim), context, chunk
    )
    matrix_ac = q4 @ swap_last_two(k4)

    relative_key_states = self.relative_k_proj(position_embeddings)
    relative_key_states = relative_key_states.view(-1, num_heads, head_dim)
    relative_key_states = relative_key_states.to(dtype=q4.dtype)
    relative_key_states = swap_mid_dims(relative_key_states.unsqueeze(0)).squeeze(0)
    rel_bh = relative_key_states.reshape(1, num_heads, -1, head_dim).expand(
        batch_size, -1, -1, -1
    ).reshape(batch_size * num_heads, -1, head_dim)
    matrix_bd = q4.reshape(batch_size * num_heads, num_blocks * chunk, head_dim) @ (
        swap_last_two(rel_bh)
    )
    # HF _rel_shift is 5-D. Singleton head dim is expandDims, not a head/seq swap.
    # Do not replace _rel_shift (prefix-matmul dropped cosine).
    matrix_bd = self._rel_shift(
        matrix_bd.reshape(batch_size * num_heads, 1, num_blocks, chunk, -1)
    )
    matrix_bd = matrix_bd.reshape(batch_size * num_heads, num_blocks, chunk, context)

    attn_weights = matrix_ac + matrix_bd
    attn_weights = attn_weights / self.softcap
    attn_weights = torch.tanh(attn_weights)
    attn_weights = attn_weights * self.softcap

    if attention_mask is not None:
        mask = attention_mask.to(dtype=attn_weights.dtype)
        if mask.ndim == 5:
            mask = mask.reshape(batch_size, num_blocks, chunk, context)
        attn_weights = attn_weights.reshape(
            batch_size, num_heads, num_blocks * chunk, context
        ) + mask.reshape(batch_size, 1, num_blocks * chunk, context)
        attn_weights = attn_weights.reshape(
            batch_size * num_heads, num_blocks, chunk, context
        )

    attn_weights = F.softmax(attn_weights, dim=-1)
    attn_output = attn_weights @ v4
    attn_output = attn_output.reshape(batch_size, num_heads, num_blocks * chunk, head_dim)
    attn_output = swap_mid_dims(attn_output).reshape(
        batch_size, num_blocks * chunk, num_heads * head_dim
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


def _subsample_conv_forward(
    self,
    input_features: torch.Tensor,
    input_features_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Accept 4-D NCHW ``[B, 1, T, C]`` so package I/O is expandDims, not permute."""
    hidden_states = input_features
    if hidden_states.ndim == 3:
        hidden_states = hidden_states.unsqueeze(1)
    hidden_states, mask = self.layer0(hidden_states, input_features_mask)
    hidden_states, mask = self.layer1(hidden_states, mask)
    batch_size, _, seq_len, _ = hidden_states.shape
    hidden_states = nchw_to_nhwc(hidden_states).reshape(batch_size, seq_len, -1)
    return self.input_proj_linear(hidden_states), mask


def _subsample_layer_forward(
    self, hidden_states: torch.Tensor, mask: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Conv2d + LayerNorm without NCHW↔NHWC ``permute`` (GPU layout I/O)."""
    if mask is not None:
        mask = mask.to(device=hidden_states.device)
        hidden_states = hidden_states * mask[:, None, :, None]
    hidden_states = self.conv(hidden_states.to(self.conv.weight.dtype))
    hidden_states = self.act(self.norm(nchw_to_nhwc(hidden_states)))
    hidden_states = nhwc_to_nchw(hidden_states)
    if mask is not None:
        mask = mask[:, ::2]
    return hidden_states, mask


def apply_audio_unfold_patch() -> dict[str, int]:
    """Monkey-patch ``Gemma4AudioAttention._extract_block_context`` for export."""
    import transformers.models.gemma4.modeling_gemma4 as g4

    g4.Gemma4AudioAttention._extract_block_context = _extract_block_context_slices
    g4.Gemma4AudioSubSampleConvProjection.forward = _subsample_conv_forward
    g4.Gemma4AudioSubSampleConvProjectionLayer.forward = _subsample_layer_forward
    return {
        "patched": 1,
        "window_op": "onehot_matmul",
        "attn_layout": "heads_first_4d_no5d",
        "io": "nchw_expanddims",
        "subsample": "nchw_ln_no_permute",
        "value_dtype": "hidden_no_float_cast",
    }


def apply_audio_ane_mask_patch() -> dict[str, Any]:
    """Replace i1 blocked mask + ``masked_fill`` with float additive 5D."""
    import transformers.models.gemma4.modeling_gemma4 as g4

    g4.Gemma4AudioAttention.forward = _audio_attn_forward
    g4.Gemma4AudioModel.forward = _audio_model_forward
    return {
        "patched": 1,
        "mask": "float_blocked_additive_4d",
        "chunk": AUDIO_CHUNK,
        "context": AUDIO_CHUNK + AUDIO_PAST + AUDIO_FUTURE,
    }


def apply_audio_export_patches() -> dict[str, Any]:
    """Unfold gather + float blocked mask (ANE-legal, no i1 reshape)."""
    unfold = apply_audio_unfold_patch()
    mask = apply_audio_ane_mask_patch()
    return {"unfold": unfold, "mask": mask}
