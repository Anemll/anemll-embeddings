"""Export-time Gemma4 audio patches for Core AI / ANE.

* ``unfold`` → ``index_select`` (convert rejects ``aten.unfold.default``).
* HF ``create_bidirectional_mask`` + ``_convert_4d_mask_to_blocked_5d`` emit
  ``i1``. ANE cannot reshape i1, so preferred-ANE falls through to GPU.
  Build a float additive 4-D blocked mask instead (no ``masked_fill`` /
  ``logical_not`` / 5-D ``permute``). Package I/O is 4-D NCHW expandDims.
* Rel-pos ``position_ids`` is mid-graph int64 ``arange`` (not package I/O).
  ANE rejects that type. Keep it in the hidden float dtype.
* LightConv1d ``transpose(1,2)`` on ``[1,70,1024]`` is the leftover
  ``reshape→permute``. Depthwise conv stays channels-last.
"""

from __future__ import annotations

import functools
from typing import Any

import numpy as np

import torch
import torch.nn.functional as F

# HF Gemma4 audio defaults (EmbeddingGemma 2).
AUDIO_CHUNK = 12
AUDIO_PAST = 12  # attention_context_left - 1
AUDIO_FUTURE = 0
AUDIO_LEFT_WINDOW = 12
AUDIO_RIGHT_WINDOW = 0
# config.attention_invalid_logits_value is -1e9, which is -inf in fp16 (the
# mask then makes NaN). Logits are softcapped to ±50, so -1e4 already gives
# exp() == 0 for masked keys.
AUDIO_INVALID = -1.0e4
# Largest integer range fp16 holds exactly; one-hot index math must stay below it.
FP16_EXACT_INT = 2048
FP16_MAX = 65504.0


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
    return torch.relu(1.0 - (idx.unsqueeze(-1) - slots).abs())


@functools.lru_cache(maxsize=None)
def _swap_permutation(dim_a: int, dim_b: int) -> np.ndarray:
    """``[A*B, A*B]`` 0/1 matrix mapping row ``b*A + a`` to ``a*B + b``."""
    n = dim_a * dim_b
    perm = np.zeros((n, n), dtype=np.float32)
    src = [a * dim_b + b for b in range(dim_b) for a in range(dim_a)]
    perm[np.arange(n), src] = 1.0
    return perm


def swap_mid_dims(x: torch.Tensor) -> torch.Tensor:
    """``[N, A, B, D] → [N, B, A, D]`` via left one-hot matmul (no transpose).

    Audio leftover ``GPU_region_0`` was ``permute(0,3,1,2,4)`` on
    ``[B, n, c, H, D] ↔ [B, H, n, c, D]``. Swap heads/seq in 4-D instead.
    Left-multiply so Core AI does not insert ``reshape→transpose``.
    """
    if x.ndim != 4:
        raise ValueError(f"swap_mid_dims rank {x.ndim} (want [N, A, B, D])")
    batch, dim_a, dim_b, last = (int(s) for s in x.shape)
    if dim_a * dim_b > FP16_EXACT_INT:
        # fp16 indices past 2048 are inexact (the one-hot comes out wrong after
        # cast16) and the (A*B)^2 one-hot is huge: the conv stem's swap was
        # 17920^2. Use a plain transpose.
        return x.transpose(1, 2).contiguous()
    # Exact 0/1 permutation built on the host, so the graph sees a constant
    # weight. Computed in-graph (relu of index math) it stayed a computed
    # tensor, and the ANE rejects matmul with a computed left operand
    # ("Unsupported mps.matmul": text_embeds went 1 -> 38 ANE regions).
    onehot = torch.from_numpy(_swap_permutation(dim_a, dim_b)).to(device=x.device, dtype=x.dtype)
    flat = x.reshape(batch, dim_a * dim_b, last)
    return (onehot @ flat).reshape(batch, dim_b, dim_a, last)


def nchw_to_nhwc(x: torch.Tensor) -> torch.Tensor:
    """``[B, C, H, W] → [B, H, W, C]`` via one-hot swaps (no ``permute``)."""
    if x.ndim != 4:
        raise ValueError(f"nchw_to_nhwc rank {x.ndim}")
    # A plain permute: the one-hot swaps here were (C*H)^2 and a 1024^2
    # matrix-vector matmul the ANE rejects ("Unsupported mps.matmul").
    return x.permute(0, 2, 3, 1).contiguous()


def nhwc_to_nchw(x: torch.Tensor) -> torch.Tensor:
    """``[B, H, W, C] → [B, C, H, W]`` via one-hot swaps (no ``permute``)."""
    if x.ndim != 4:
        raise ValueError(f"nhwc_to_nchw rank {x.ndim}")
    return x.permute(0, 3, 1, 2).contiguous()


def swap_last_two(x: torch.Tensor) -> torch.Tensor:
    """``[..., A, B] → [..., B, A]`` via one-hot (no ``transpose`` / ``permute``)."""
    if x.ndim < 2:
        raise ValueError(f"swap_last_two rank {x.ndim}")
    lead = x.shape[:-2]
    dim_a, dim_b = int(x.shape[-2]), int(x.shape[-1])
    flat = x.reshape(int(x.numel() // (dim_a * dim_b)), dim_a, dim_b, 1)
    return swap_mid_dims(flat).reshape(*lead, dim_b, dim_a)


def stride_select(x: torch.Tensor, step: int) -> torch.Tensor:
    """``x[:, ::step]`` via one-hot matmul — no ``aten.slice``.

    Exact on dim 1. Used for the audio subsample mask ``(1,280)→(1,70)``.
    That slice was in the FX graph and never became a MIL use-site.
    """
    step = int(step)
    if step <= 0:
        raise ValueError(f"stride_select step {step}")
    seq = int(x.shape[1])
    n_keep = seq // step
    dtype = x.dtype if x.is_floating_point() else torch.float32
    slots = torch.arange(seq, device=x.device, dtype=dtype)
    idx = torch.arange(n_keep, device=x.device, dtype=dtype) * float(step)
    onehot = torch.relu(1.0 - (idx.unsqueeze(1) - slots).abs())
    flat = x.reshape(int(x.shape[0]), seq, -1).to(dtype=dtype)
    got = onehot.reshape(1, n_keep, seq) @ flat
    return got.reshape(int(x.shape[0]), n_keep, *x.shape[2:]).to(dtype=x.dtype)


def prefix_rows(x: torch.Tensor, keep: int) -> torch.Tensor:
    """``x[:, :keep]`` via one-hot matmul — no ``aten.slice``.

    Audio attention unpad is ``[1,72,1024] → [1,70,1024]`` (block pad 72,
    real seq 70). That slice was in FX and never became a MIL use-site.
    """
    seq = int(x.shape[1])
    keep = int(keep)
    if keep == seq:
        return x
    if keep < 0 or keep > seq:
        raise ValueError(f"prefix_rows keep {keep} seq {seq}")
    dtype = x.dtype if x.is_floating_point() else torch.float32
    slots = torch.arange(seq, device=x.device, dtype=dtype)
    idx = torch.arange(keep, device=x.device, dtype=dtype)
    onehot = torch.relu(1.0 - (idx.unsqueeze(1) - slots).abs())
    flat = x.reshape(int(x.shape[0]), seq, -1).to(dtype=dtype)
    got = onehot.reshape(1, keep, seq) @ flat
    return got.reshape(int(x.shape[0]), keep, *x.shape[2:]).to(dtype=x.dtype)


def bind_glu_half_weights(module: torch.nn.Module) -> int:
    """Split LightConv GLU weights once, outside the traced graph.

    ``torch.split`` on ``linear_start.weight`` stayed in FX as
    ``aten.split`` and constant-folded out of MIL. Bind the halves as
    buffers so export has two parameters and no split op.
    """
    bound = 0
    for mod in module.modules():
        start = getattr(mod, "linear_start", None)
        if start is None or getattr(mod, "_glu_w_lo", None) is not None:
            continue
        inner = getattr(start, "linear", start)
        weight = getattr(inner, "weight", None)
        if weight is None:
            continue
        half = int(weight.shape[0]) // 2
        mod.register_buffer(
            "_glu_w_lo", weight.detach()[:half].contiguous(), persistent=False
        )
        mod.register_buffer(
            "_glu_w_hi", weight.detach()[half:].contiguous(), persistent=False
        )
        bias = getattr(inner, "bias", None)
        if bias is not None:
            mod.register_buffer(
                "_glu_b_lo", bias.detach()[:half].contiguous(), persistent=False
            )
            mod.register_buffer(
                "_glu_b_hi", bias.detach()[half:].contiguous(), persistent=False
            )
        bound += 1
    return bound


def bind_fp16_audio_constants(module: torch.nn.Module) -> dict[str, int]:
    """Make the audio tower's fixed constants fp16-legal before export.

    * ``gradient_clipping`` is 1e10 (not representable in fp16), so cast16
      kept those clamps in f32 and the ANE refused them. Activations stay far
      below 65504, so clamping at the fp16 max is the same no-op.
    * The rel-pos sinusoid depends only on ``context_size``; bake it as a
      buffer so ``relative_k_proj`` reads a constant, not GPU sin/cos.
    """
    import transformers.models.gemma4.modeling_gemma4 as g4

    clipped = 0
    baked = 0
    for mod in module.modules():
        if hasattr(mod, "gradient_clipping"):
            mod.gradient_clipping = min(float(mod.gradient_clipping), FP16_MAX)
            clipped += 1
        if isinstance(mod, g4.Gemma4AudioRelPositionalEncoding):
            mod._buffers.pop("_pos_embed_const", None)
            probe = torch.zeros(1, dtype=torch.float32)
            with torch.no_grad():
                pos = _rel_pos_forward(mod, probe).detach().clone()
            mod.register_buffer("_pos_embed_const", pos, persistent=False)
            baked += 1
    # The rel-pos keys depend only on weights and that constant. Computed in
    # the graph they were two GPU regions: relative_k_proj on a constant input
    # ("Input cannot run on ANE") and a 1664^2 one-hot matrix-vector transpose.
    pos = next(
        (m._pos_embed_const for m in module.modules() if isinstance(m, g4.Gemma4AudioRelPositionalEncoding)),
        None,
    )
    rel_keys = 0
    for mod in module.modules():
        if isinstance(mod, g4.Gemma4AudioAttention) and pos is not None:
            mod._buffers.pop("_rel_kt_const", None)
            with torch.no_grad():
                kt = _rel_keys_transposed(mod, pos, int(mod.num_heads), int(mod.head_dim))
            mod.register_buffer("_rel_kt_const", kt.detach().clone(), persistent=False)
            rel_keys += 1
    return {"gradient_clipping": clipped, "rel_pos": baked, "rel_keys": rel_keys}


def glu_from_bound_halves(mod: Any, x: torch.Tensor) -> torch.Tensor:
    """GLU from buffers written by ``bind_glu_half_weights``."""
    linear = mod.linear_start
    if getattr(linear, "use_clipped_linears", False):
        x = torch.clamp(x, linear.input_min, linear.input_max)
    b_lo = getattr(mod, "_glu_b_lo", None)
    b_hi = getattr(mod, "_glu_b_hi", None)
    lo = F.linear(x, mod._glu_w_lo, b_lo)
    hi = F.linear(x, mod._glu_w_hi, b_hi)
    if getattr(linear, "use_clipped_linears", False):
        lo = torch.clamp(lo, linear.output_min, linear.output_max)
        hi = torch.clamp(hi, linear.output_min, linear.output_max)
    return lo * torch.sigmoid(hi)


def glu_split_last(x: torch.Tensor) -> torch.Tensor:
    """``F.glu(x, dim=-1)`` via ``torch.split`` — first half × sigmoid(second).

    Do not use a constant half-take (drifted mm_audio 0.87080 → 0.88296).
    """
    last = int(x.shape[-1])
    if last % 2:
        raise ValueError(f"glu_split_last last dim {last} (want even)")
    lo, hi = torch.split(x, last // 2, dim=-1)
    return lo * torch.sigmoid(hi)


def _inner_linear(linear: Any) -> Any:
    """``Gemma4ClippableLinear`` stores ``nn.Linear`` at ``.linear`` (no ``.weight``)."""
    inner = getattr(linear, "linear", None)
    if inner is not None and hasattr(inner, "weight"):
        return inner
    return linear


def glu_from_linear_halves(linear: Any, x: torch.Tensor) -> torch.Tensor:
    """``linear`` then GLU as two half-width GEMMs — no ``[1,70,2048]`` slice.

    Live leftover ``reshape→strided_slice`` was ``F.glu`` on the 2048-wide
    ``linear_start`` (affine ``d0*143360 + d1*2048 + d2 + 1024``). Split the
    inner weight into two half-width tensors and write two GEMMs. Keep the
    ClippableLinear input/output clamps so this matches ``linear`` then
    first×sigmoid(second) in f32. Do not retry a constant half-take.
    """
    if getattr(linear, "use_clipped_linears", False):
        x = torch.clamp(x, linear.input_min, linear.input_max)
    inner = _inner_linear(linear)
    weight = inner.weight
    bias = getattr(inner, "bias", None)
    out_f = int(weight.shape[0])
    if out_f % 2:
        raise ValueError(f"glu_from_linear_halves out {out_f} (want even)")
    half = out_f // 2
    w_lo, w_hi = torch.split(weight, half, dim=0)
    if bias is None:
        lo = F.linear(x, w_lo)
        hi = F.linear(x, w_hi)
    else:
        b_lo, b_hi = torch.split(bias, half, dim=0)
        lo = F.linear(x, w_lo, b_lo)
        hi = F.linear(x, w_hi, b_hi)
    if getattr(linear, "use_clipped_linears", False):
        lo = torch.clamp(lo, linear.output_min, linear.output_max)
        hi = torch.clamp(hi, linear.output_min, linear.output_max)
    return lo * torch.sigmoid(hi)


def depthwise_conv1d_channels_last(conv: Any, x: torch.Tensor) -> torch.Tensor:
    """Causal depthwise ``conv1d`` on ``[B, S, C]`` without ``transpose(1, 2)``.

    HF ``Gemma4AudioLightConv1d`` does ``conv(x.transpose(1, 2)).transpose(1, 2)``
    on ``[1, 70, 1024]``. That is the leftover ``reshape→permute``. Window the
    sequence dim with a one-hot and multiply the depthwise taps in place.
    """
    left_pad = int(conv.left_pad)
    kernel = conv.kernel_size
    kernel_size = int(kernel[0] if isinstance(kernel, tuple) else kernel)
    padded = F.pad(x, (0, 0, left_pad, 0))
    windows = slice_seq_windows(padded, kernel_size, 1)
    # weight is [C, 1, K]; taps must be [1, 1, K, C] = weight[c,0,k].
    taps = swap_last_two(conv.weight[:, 0, :]).reshape(1, 1, kernel_size, -1)
    out = (windows * taps).sum(dim=2)
    if conv.bias is not None:
        out = out + conv.bias.reshape(1, 1, -1)
    return out


def rel_pos_ids_float(n_pos: int, *, device: torch.device | str, dtype: torch.dtype) -> torch.Tensor:
    """``arange(n_pos-1, -1, -1)`` in ``dtype`` — no mid-graph int64."""
    n_pos = int(n_pos)
    return torch.arange(n_pos - 1, -1, -1, device=device, dtype=dtype)


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


def _rel_shift_static(self, x: torch.Tensor) -> torch.Tensor:
    """Transformer-XL rel-shift via one static prefix slice (no per-row loop).

    Live leftover after this was ``reshape→strided_slice→reshape``
    (affine ``12×25=300``). Kept for tests. Export uses ``_rel_shift_baked``.
    """
    if x.ndim < 2:
        raise ValueError(f"_rel_shift_static rank {x.ndim}")
    *lead, block_size, pos_len = x.shape
    context = int(self.context_size)
    block_size = int(block_size)
    x = F.pad(x, (0, context + 1 - int(pos_len)))
    flat = x.reshape(*lead, block_size * (context + 1))
    keep = block_size * context
    return flat[..., :keep].reshape(*lead, block_size, context)


def rel_shift_onehot(
    block_size: int,
    pos_len: int,
    context: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """``[chunk*context, chunk*pos]`` one-hot for XL rel-shift without pad/slice.

    Output ``p`` reads ``scores[r, c]`` where ``r, c = divmod(p, context+1)``
    when ``c < pos``. Pad slots stay zero. Not the banned 300-wide
    prefix-matmul (no pad to ``context+1``, no ``[..., :288]``).
    """
    block_size = int(block_size)
    pos_len = int(pos_len)
    context = int(context)
    keep = block_size * context
    src = block_size * pos_len
    slots = torch.arange(keep, device=device, dtype=dtype)
    stride = float(context + 1)
    row = torch.floor(slots / stride)
    col = slots - row * stride
    valid = torch.clamp(float(pos_len) - col, 0, 1)
    src_idx = row * float(pos_len) + col
    dest = torch.arange(src, device=device, dtype=dtype)
    return torch.relu(1.0 - (src_idx.unsqueeze(1) - dest).abs()) * valid.unsqueeze(1)


def _rel_shift_baked(self, x: torch.Tensor) -> torch.Tensor:
    """XL rel-shift via a constant one-hot — no ``strided_slice``.

    GPU leftover on ``62e3011`` was the prefix slice itself
    (``reshape→strided_slice→reshape``). Do not retry prefix-matmul or
    the f16 island cast.
    """
    if x.ndim < 2:
        raise ValueError(f"_rel_shift_baked rank {x.ndim}")
    *lead, block_size, pos_len = x.shape
    context = int(self.context_size)
    block_size = int(block_size)
    pos_len = int(pos_len)
    select = rel_shift_onehot(
        block_size, pos_len, context, device=x.device, dtype=x.dtype
    )
    flat = x.reshape(*lead, block_size * pos_len)
    return (flat @ select.T).reshape(*lead, block_size, context)


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
    onehot_kt = torch.relu(1.0 - (slots.view(1, -1, 1) - idx.unsqueeze(1)).abs())
    right = onehot_kt.unsqueeze(0).expand(batch, -1, -1, -1).reshape(
        batch * num_blocks, padded_keys, context
    )
    keep4 = (left @ right).reshape(batch, num_blocks, chunk_size, context)
    return (1.0 - keep4) * float(invalid)


def _rel_keys_transposed(
    attn: Any, position_embeddings: torch.Tensor, num_heads: int, head_dim: int
) -> torch.Tensor:
    """Relative-position keys as ``[H, D, P]`` (ready for ``q @ k^T``)."""
    rel = attn.relative_k_proj(position_embeddings).view(-1, num_heads, head_dim)
    rel = swap_mid_dims(rel.unsqueeze(0)).squeeze(0)  # [H, P, D]
    return swap_last_two(rel)


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

    rel_kt = getattr(self, "_rel_kt_const", None)
    if rel_kt is None:
        rel_kt = _rel_keys_transposed(self, position_embeddings, num_heads, head_dim)
    rel_kt = rel_kt.to(dtype=q4.dtype).unsqueeze(0).expand(batch_size, -1, -1, -1)
    matrix_bd = q4.reshape(batch_size * num_heads, num_blocks * chunk, head_dim) @ (
        rel_kt.reshape(batch_size * num_heads, head_dim, -1)
    )
    # Baked XL shift on ``[BH, n, chunk, pos]`` — constant one-hot, no slice.
    matrix_bd = self._rel_shift(
        matrix_bd.reshape(batch_size * num_heads, num_blocks, chunk, -1)
    )

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
    attn_output = prefix_rows(attn_output, seq_length).contiguous()
    attn_output = self.post(attn_output.to(hidden_states.dtype))
    return attn_output, attn_weights


def _rel_pos_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """Sinusoidal rel-pos with float ``position_ids`` (ANE-legal mid-graph).

    HF ``torch.arange(..., device=)`` is int64. That tensor is the
    ``Incompatible element type for ANE`` leftover (not fp16/si8/si16).
    It is mid-graph, not package I/O — do not I/O-cast it.
    """
    const = getattr(self, "_pos_embed_const", None)
    if const is not None:
        return const.to(dtype=hidden_states.dtype)
    n_pos = int(self.context_size) // 2 + 1
    dtype = hidden_states.dtype
    position_ids = rel_pos_ids_float(n_pos, device=hidden_states.device, dtype=dtype)
    position_ids = position_ids[..., None]
    scaled_time = position_ids * self.inv_timescales.to(device=hidden_states.device, dtype=dtype)
    pos_embed = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=-1)
    return pos_embed.to(dtype=dtype)


def _light_conv1d_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    residual = hidden_states
    hidden_states = self.pre_layer_norm(hidden_states)
    if getattr(self, "_glu_w_lo", None) is not None:
        hidden_states = glu_from_bound_halves(self, hidden_states)
    else:
        hidden_states = glu_from_linear_halves(self.linear_start, hidden_states)
    hidden_states = depthwise_conv1d_channels_last(self.depthwise_conv1d, hidden_states)
    gradient_clipping = min(self.gradient_clipping, torch.finfo(hidden_states.dtype).max)
    hidden_states = torch.clamp(hidden_states, -gradient_clipping, gradient_clipping)
    hidden_states = self.conv_norm(hidden_states)
    hidden_states = self.act_fn(hidden_states)
    hidden_states = self.linear_end(hidden_states)
    hidden_states = hidden_states + residual
    return hidden_states


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
        # Not config.attention_invalid_logits_value (-1e9): see AUDIO_INVALID.
        invalid=AUDIO_INVALID,
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


def layer_norm_fp16_safe(x: torch.Tensor, norm: nn.LayerNorm) -> torch.Tensor:
    """Last-dim ``nn.LayerNorm`` computed on ``x / max|x|``.

    The conv stem output is large enough that the variance overflows fp16.
    LayerNorm is scale-invariant, so with ``inv = 1 / max|x|`` and eps as
    ``eps * inv²`` this is exact; zero rows stay zero (max clamped to 1e-3).
    """
    inv = torch.reciprocal(x.abs().amax(-1, keepdim=True).clamp_min(1e-3))
    xs = x * inv
    centered = xs - xs.mean(-1, keepdim=True)
    var = (centered * centered).mean(-1, keepdim=True)
    out = centered * torch.rsqrt(var + norm.eps * inv * inv)
    if norm.weight is not None:
        out = out * norm.weight
    if norm.bias is not None:
        out = out + norm.bias
    return out


def _subsample_layer_forward(
    self, hidden_states: torch.Tensor, mask: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Conv2d + LayerNorm without NCHW↔NHWC ``permute`` (GPU layout I/O)."""
    if mask is not None:
        mask = mask.to(device=hidden_states.device)
        hidden_states = hidden_states * mask[:, None, :, None]
    hidden_states = self.conv(hidden_states.to(self.conv.weight.dtype))
    hidden_states = self.act(layer_norm_fp16_safe(nchw_to_nhwc(hidden_states), self.norm))
    hidden_states = nhwc_to_nchw(hidden_states)
    if mask is not None:
        mask = stride_select(mask, 2)
    return hidden_states, mask


def apply_audio_unfold_patch() -> dict[str, int]:
    """Monkey-patch ``Gemma4AudioAttention._extract_block_context`` for export."""
    import transformers.models.gemma4.modeling_gemma4 as g4

    g4.Gemma4AudioAttention._extract_block_context = _extract_block_context_slices
    g4.Gemma4AudioSubSampleConvProjection.forward = _subsample_conv_forward
    g4.Gemma4AudioSubSampleConvProjectionLayer.forward = _subsample_layer_forward
    g4.Gemma4AudioLightConv1d.forward = _light_conv1d_forward
    g4.Gemma4AudioRelPositionalEncoding.forward = _rel_pos_forward
    g4.Gemma4AudioAttention._rel_shift = _rel_shift_baked
    return {
        "patched": 1,
        "window_op": "onehot_matmul",
        "attn_layout": "heads_first_4d_no5d",
        "io": "nchw_expanddims",
        "subsample": "nchw_ln_no_permute",
        "value_dtype": "hidden_no_float_cast",
        "rel_pos": "float_arange",
        "lconv1d": "channels_last_onehot",
        "glu": "bound_halves",
        "mask_stride": "onehot_stride2",
        "attn_unpad": "onehot_prefix",
        "rel_shift": "baked_onehot",
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
