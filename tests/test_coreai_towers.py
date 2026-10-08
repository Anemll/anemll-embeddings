#!/usr/bin/env python3
"""Unit checks for Core AI tower I/O specs (no checkpoint)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.audio_export_patches import (  # noqa: E402
    AUDIO_CHUNK,
    AUDIO_FUTURE,
    AUDIO_INVALID,
    AUDIO_PAST,
    _rel_shift_baked,
    _rel_shift_matmul,
    _rel_shift_static,
    blocked_additive_attention_mask,
    depthwise_conv1d_channels_last,
    gather_seq_windows,
    bind_glu_half_weights,
    glu_from_bound_halves,
    glu_from_linear_halves,
    glu_split_last,
    prefix_rows,
    stride_select,
    nchw_to_nhwc,
    nhwc_to_nchw,
    rel_pos_ids_float,
    slice_seq_windows,
    swap_last_two,
    swap_mid_dims,
)
from src.trace_patches import (  # noqa: E402
    _make_attention_forward,
    _take_ple_layer,
    bake_k_layout,
    bind_used_weight_layout,
)
from src.vision_export_patches import (  # noqa: E402
    _apply_multidimensional_rope_ane,
    _recomposition_frequencies_ane,
    _vision_attn_forward,
    embedding_from_int_indices,
    rotate_half_matmul,
)
from src.coreai_towers import (  # noqa: E402
    AUDIO_FEAT,
    AUDIO_FRAMES,
    AUDIO_SOFT_TOKENS,
    VISION_PATCH_DIM,
    VISION_PATCHES,
    VISION_SOFT_TOKENS,
    audio_example,
    tower_io_spec,
    vision_example,
)


def _fail(msg: str) -> None:
    raise AssertionError(msg)


def test_vision_example_shape() -> None:
    pixels, pos = vision_example()
    if tuple(pixels.shape) != (1, VISION_PATCHES, VISION_PATCH_DIM):
        _fail(str(pixels.shape))
    if tuple(pos.shape) != (1, VISION_PATCHES, 2):
        _fail(str(pos.shape))
    if pos.dtype != torch.float16:
        _fail(f"pos dtype {pos.dtype} (want float16)")
    if pos.min() < 0:
        _fail("example grid should have no pad (-1)")


def test_audio_example_shape() -> None:
    feat, mask = audio_example()
    if tuple(feat.shape) != (1, 1, AUDIO_FRAMES, AUDIO_FEAT):
        _fail(str(feat.shape))
    if tuple(mask.shape) != (1, 1, AUDIO_FRAMES, 1) or mask.dtype != torch.float16:
        _fail(f"{tuple(mask.shape)} {mask.dtype}")
    if int((mask != 0).sum()) != AUDIO_FRAMES - 8:
        _fail("expected trailing-zero mask (avoid dummy_pool const-all-ones)")


def test_gather_seq_windows_matches_unfold() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 70, 8, 16)
    window, step = 24, 12
    ref = torch.movedim(x.unfold(1, window, step), -1, 2)
    got = gather_seq_windows(x, window, step)
    if got.shape != ref.shape:
        _fail(f"shape {tuple(got.shape)} != {tuple(ref.shape)}")
    if not torch.equal(got, ref):
        _fail("gather_seq_windows != unfold+movedim")


def test_slice_seq_windows_matches_unfold() -> None:
    torch.manual_seed(1)
    x = torch.randn(1, 93, 8, 16)
    window, step = 24, 12
    ref = torch.movedim(x.unfold(1, window, step), -1, 2)
    got = slice_seq_windows(x, window, step)
    if got.shape != ref.shape:
        _fail(f"shape {tuple(got.shape)} != {tuple(ref.shape)}")
    if not torch.equal(got, ref):
        _fail("slice_seq_windows != unfold+movedim")
    if not torch.equal(got, gather_seq_windows(x, window, step)):
        _fail("slice_seq_windows != gather_seq_windows")


def test_slice_seq_windows_export_has_no_gather() -> None:
    class _Win(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return slice_seq_windows(x, 24, 12)

    ep = torch.export.export(_Win(), (torch.randn(1, 93, 4, 8),), strict=False)
    targets = [str(n.target) for n in ep.graph.nodes]
    if any(("gather" in t or "index_select" in t or "unfold" in t) for t in targets):
        _fail(f"export still contains gather/index_select/unfold: {targets}")


def test_gather_seq_windows_export_has_no_unfold() -> None:
    class _Win(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return gather_seq_windows(x, 24, 12)

    ep = torch.export.export(_Win(), (torch.randn(1, 70, 4, 8),), strict=False)
    targets = [str(n.target) for n in ep.graph.nodes]
    if any("unfold" in t for t in targets):
        _fail(f"export still contains unfold: {targets}")


def test_blocked_additive_mask_shape_and_pad() -> None:
    keep = torch.ones(1, AUDIO_SOFT_TOKENS, dtype=torch.float32)
    keep[:, -8:] = 0
    mask = blocked_additive_attention_mask(keep)
    n_blocks = (AUDIO_SOFT_TOKENS + AUDIO_CHUNK - 1) // AUDIO_CHUNK
    context = AUDIO_CHUNK + AUDIO_PAST + AUDIO_FUTURE
    if tuple(mask.shape) != (1, n_blocks, AUDIO_CHUNK, context):
        _fail(str(tuple(mask.shape)))
    if mask.dtype != torch.float32:
        _fail(f"dtype {mask.dtype}")
    # Last 8 keys of seq 70 land in the final block's current-chunk columns.
    # Those positions must be masked (large negative), earlier keys not.
    # AUDIO_INVALID is -1e4 (fp16-safe); logits are softcapped to ±50.
    if not bool((mask[0, -1, :, AUDIO_PAST:] <= AUDIO_INVALID).any()):
        _fail("expected padded keys to be invalid in last block")
    if float(mask[0, 0, 0, AUDIO_PAST]) < -1.0:
        _fail(f"first valid key of block 0 should be keep, got {float(mask[0, 0, 0, AUDIO_PAST])}")


def test_blocked_additive_mask_export_has_no_gather() -> None:
    class _Mask(torch.nn.Module):
        def forward(self, keep: torch.Tensor) -> torch.Tensor:
            return blocked_additive_attention_mask(keep)

    ep = torch.export.export(
        _Mask(), (torch.ones(1, AUDIO_SOFT_TOKENS, dtype=torch.float32),), strict=False
    )
    targets = [str(n.target) for n in ep.graph.nodes]
    if any("gather" in t for t in targets):
        _fail(f"export still contains gather: {targets}")


def test_embedding_from_int_indices_matches_embedding() -> None:
    torch.manual_seed(0)
    table = torch.randn(8, 4)
    idx = torch.tensor([[0, 3, 7, 1]], dtype=torch.int16)
    ref = torch.nn.functional.embedding(idx.to(torch.long), table)
    got = embedding_from_int_indices(idx, table)
    if got.shape != ref.shape:
        _fail(str(tuple(got.shape)))
    if not torch.allclose(got, ref, atol=1e-6):
        _fail("int16 one-hot embed != F.embedding")


def test_embedding_from_int_indices_export_has_no_i64() -> None:
    class _Emb(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.table = torch.nn.Parameter(torch.randn(6, 3))

        def forward(self, idx: torch.Tensor) -> torch.Tensor:
            return embedding_from_int_indices(idx, self.table)

    ep = torch.export.export(_Emb(), (torch.zeros(1, 4, dtype=torch.int16),), strict=False)
    for node in ep.graph.nodes:
        blob = f"{node.target} {node.meta.get('val', '')}"
        if "embedding" in str(node.target) or "int64" in blob or "torch.long" in blob:
            _fail(f"export still widens/embeds with i64: {blob}")


def test_blocked_additive_mask_export_stays_float() -> None:
    class _Mask(torch.nn.Module):
        def forward(self, keep: torch.Tensor) -> torch.Tensor:
            return blocked_additive_attention_mask(keep)

    ep = torch.export.export(
        _Mask(), (torch.ones(1, AUDIO_SOFT_TOKENS, dtype=torch.float32),), strict=False
    )
    for node in ep.graph.nodes:
        blob = f"{node.target} {node.meta.get('val', '')}"
        if "i1" in blob or "torch.bool" in blob:
            _fail(f"export still contains bool/i1: {blob}")


def test_swap_mid_dims_matches_permute() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 70, 8, 16)
    ref = x.permute(0, 2, 1, 3)
    got = swap_mid_dims(x)
    if got.shape != ref.shape or not torch.allclose(got, ref, atol=1e-5):
        _fail("swap_mid_dims != permute(0,2,1,3)")
    q = torch.randn(1, 6, 12, 8, 16)
    k = torch.randn(1, 6, 24, 8, 16)
    ref_ac = q.permute(0, 3, 1, 2, 4) @ k.permute(0, 3, 1, 4, 2)
    q4 = swap_mid_dims(q.reshape(6, 12, 8, 16))
    k4 = swap_mid_dims(k.reshape(6, 24, 8, 16))
    got_ac = (q4 @ k4.transpose(-1, -2)).reshape(1, 6, 8, 12, 24).permute(0, 2, 1, 3, 4)
    if not torch.allclose(got_ac, ref_ac, atol=1e-5):
        _fail("heads-first 4D matmul != 5D permute matmul")


def test_swap_mid_dims_export_has_no_5d_permute() -> None:
    class _Swap(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return swap_mid_dims(x)

    ep = torch.export.export(_Swap(), (torch.randn(1, 70, 8, 16),), strict=False)
    blob = " ".join(str(n.target) for n in ep.graph.nodes)
    if "permute" in blob or "transpose" in blob and "5" in blob:
        # 3-D transpose for the matmul is OK; reject aten.permute of 5-D.
        perms = [str(n.target) for n in ep.graph.nodes if "permute" in str(n.target)]
        if perms:
            _fail(f"export still permutes: {perms}")


def test_nchw_nhwc_matches_permute() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 4, 5, 6)
    got = nchw_to_nhwc(x)
    ref = x.permute(0, 2, 3, 1)
    if got.shape != ref.shape or not torch.allclose(got, ref, atol=1e-5):
        _fail("nchw_to_nhwc != permute(0,2,3,1)")
    back = nhwc_to_nchw(got)
    if not torch.allclose(back, x, atol=1e-5):
        _fail("nhwc_to_nchw roundtrip")
    kt = swap_last_two(x)
    if not torch.allclose(kt, x.transpose(-1, -2), atol=1e-5):
        _fail("swap_last_two != transpose(-1,-2)")


def test_vision_attn_matches_transpose_matmul() -> None:
    class _Cfg:
        num_attention_heads = 2
        num_key_value_heads = 2

    class _Attn:
        head_dim = 4
        scaling = 1.0
        config = _Cfg()

        def __init__(self) -> None:
            self.q_proj = torch.nn.Linear(8, 8, bias=False)
            self.k_proj = torch.nn.Linear(8, 8, bias=False)
            self.v_proj = torch.nn.Linear(8, 8, bias=False)
            self.o_proj = torch.nn.Linear(8, 8, bias=False)
            self.q_norm = torch.nn.Identity()
            self.k_norm = torch.nn.Identity()
            self.v_norm = torch.nn.Identity()

    torch.manual_seed(0)
    attn = _Attn()
    x = torch.randn(1, 5, 8)
    mask = torch.zeros(1, 1, 1, 5)
    got, _ = _vision_attn_forward(attn, x, position_embeddings=None, attention_mask=mask)
    q = attn.q_proj(x).view(1, 5, 2, 4).transpose(1, 2)
    k = attn.k_proj(x).view(1, 5, 2, 4).transpose(1, 2)
    v = attn.v_proj(x).view(1, 5, 2, 4).transpose(1, 2)
    w = torch.softmax((q @ k.transpose(-1, -2)) + mask, dim=-1)
    ref = attn.o_proj((w @ v).transpose(1, 2).reshape(1, 5, 8))
    if got.shape != ref.shape or not torch.allclose(got, ref, atol=1e-5):
        _fail("vision heads-first != transpose+matmul")


def test_vision_attn_export_has_no_sdpa() -> None:
    class _Cfg:
        num_attention_heads = 2
        num_key_value_heads = 2

    class _Attn(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head_dim = 4
            self.scaling = 1.0
            self.config = _Cfg()
            self.q_proj = torch.nn.Linear(8, 8, bias=False)
            self.k_proj = torch.nn.Linear(8, 8, bias=False)
            self.v_proj = torch.nn.Linear(8, 8, bias=False)
            self.o_proj = torch.nn.Linear(8, 8, bias=False)
            self.q_norm = torch.nn.Identity()
            self.k_norm = torch.nn.Identity()
            self.v_norm = torch.nn.Identity()

        def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            out, _ = _vision_attn_forward(
                self, x, position_embeddings=None, attention_mask=mask
            )
            return out

    mod = _Attn().eval()
    ep = torch.export.export(
        mod, (torch.randn(1, 5, 8), torch.zeros(1, 1, 1, 5)), strict=False
    )
    bad = [
        str(n.target)
        for n in ep.graph.nodes
        if "scaled_dot_product_attention" in str(n.target)
    ]
    if bad:
        _fail(f"fused sdpa still in vision attn export: {bad}")


def test_rel_shift_matmul_matches_hf() -> None:
    class _Attn:
        context_size = 24

        def hf(self, x: torch.Tensor) -> torch.Tensor:
            batch, heads, blocks, block, pos = x.shape
            context = self.context_size
            x = torch.nn.functional.pad(x, (0, context + 1 - pos))
            x = x.view(batch, heads, blocks, block * (context + 1))
            x = x[..., : block * context]
            return x.view(batch, heads, blocks, block, context)

    torch.manual_seed(0)
    x = torch.randn(1, 2, 6, 12, 24)
    attn = _Attn()
    got = _rel_shift_matmul(attn, x)
    ref = attn.hf(x)
    if got.shape != ref.shape or not torch.allclose(got, ref):
        _fail(f"rel_shift {tuple(got.shape)} != {tuple(ref.shape)}")


def test_rel_shift_static_matches_hf() -> None:
    class _Attn:
        context_size = 24

        def hf(self, x: torch.Tensor) -> torch.Tensor:
            *lead, block, pos = x.shape
            context = self.context_size
            x = torch.nn.functional.pad(x, (0, context + 1 - pos))
            x = x.reshape(*lead, block * (context + 1))
            x = x[..., : block * context]
            return x.reshape(*lead, block, context)

    torch.manual_seed(0)
    attn = _Attn()
    for shape in ((1, 2, 6, 12, 24), (8, 1, 6, 12, 13), (8, 6, 12, 13)):
        x = torch.randn(*shape)
        got = _rel_shift_static(attn, x)
        ref = attn.hf(x)
        if got.shape != ref.shape or not torch.allclose(got, ref):
            _fail(f"static rel_shift {shape} {tuple(got.shape)} != {tuple(ref.shape)}")


def test_rel_shift_baked_matches_hf() -> None:
    class _Attn:
        context_size = 24

        def hf(self, x: torch.Tensor) -> torch.Tensor:
            *lead, block, pos = x.shape
            context = self.context_size
            x = torch.nn.functional.pad(x, (0, context + 1 - pos))
            x = x.reshape(*lead, block * (context + 1))
            x = x[..., : block * context]
            return x.reshape(*lead, block, context)

    torch.manual_seed(0)
    attn = _Attn()
    for shape in ((1, 2, 6, 12, 24), (8, 1, 6, 12, 13), (8, 6, 12, 13)):
        x = torch.randn(*shape)
        got = _rel_shift_baked(attn, x)
        ref = attn.hf(x)
        if got.shape != ref.shape or not torch.allclose(got, ref, atol=1e-5):
            _fail(f"baked rel_shift {shape} {tuple(got.shape)} != {tuple(ref.shape)}")


def test_rel_shift_baked_export_has_no_slice() -> None:
    class _Shift(torch.nn.Module):
        context_size = 24

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return _rel_shift_baked(self, x)

    ep = torch.export.export(_Shift(), (torch.randn(1, 6, 12, 13),), strict=False)
    targets = [str(n.target) for n in ep.graph.nodes]
    if any("slice" in t or "cat" in t or "stack" in t or "pad" in t for t in targets):
        _fail(f"slice/pad still in baked rel-shift export: {targets}")


def test_stride_select_matches_slice() -> None:
    torch.manual_seed(0)
    x = torch.randn(1, 280)
    got = stride_select(x, 2)
    ref = x[:, ::2]
    if got.shape != ref.shape or not torch.allclose(got, ref):
        _fail("stride_select != [:, ::2]")
    got2 = stride_select(got, 2)
    ref2 = ref[:, ::2]
    if got2.shape != (1, 70) or not torch.allclose(got2, ref2):
        _fail("stride_select twice != (1,70)")


def test_stride_select_export_has_no_slice() -> None:
    class _S(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return stride_select(stride_select(x, 2), 2)

    ep = torch.export.export(_S(), (torch.randn(1, 280),), strict=False)
    bad = [str(n.target) for n in ep.graph.nodes if "slice" in str(n.target)]
    if bad:
        _fail(f"slice still in stride_select export: {bad}")


def test_prefix_rows_matches_slice() -> None:
    torch.manual_seed(0)
    x = torch.randn(1, 72, 16)
    got = prefix_rows(x, 70)
    ref = x[:, :70]
    if got.shape != ref.shape or not torch.allclose(got, ref):
        _fail("prefix_rows != [:, :70]")


def test_prefix_rows_export_has_no_slice() -> None:
    class _P(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return prefix_rows(x, 5)

    ep = torch.export.export(_P(), (torch.randn(1, 8, 4),), strict=False)
    bad = [str(n.target) for n in ep.graph.nodes if "slice" in str(n.target)]
    if bad:
        _fail(f"slice still in prefix_rows export: {bad}")


def test_bound_glu_halves_export_has_no_split() -> None:
    class _M(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear_start = torch.nn.Linear(8, 16, bias=False)
            self.pre_layer_norm = torch.nn.Identity()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return glu_from_bound_halves(self, x)

    mod = _M()
    if bind_glu_half_weights(mod) != 1:
        _fail("bind_glu_half_weights")
    x = torch.randn(1, 4, 8)
    got = mod(x)
    ref = torch.nn.functional.glu(mod.linear_start(x), dim=-1)
    if not torch.allclose(got, ref, atol=1e-5):
        _fail("bound halves != linear then F.glu")
    ep = torch.export.export(mod, (x,), strict=False)
    bad = [str(n.target) for n in ep.graph.nodes if "split" in str(n.target) or "slice" in str(n.target)]
    if bad:
        _fail(f"split/slice still in bound GLU export: {bad}")


def test_glu_split_last_matches_glu() -> None:
    torch.manual_seed(0)
    x = torch.randn(1, 70, 2048)
    got = glu_split_last(x)
    ref = torch.nn.functional.glu(x, dim=-1)
    if got.shape != ref.shape or not torch.allclose(got, ref):
        _fail("glu_split_last != F.glu")


def test_glu_from_linear_halves_matches_linear_glu() -> None:
    torch.manual_seed(0)
    lin = torch.nn.Linear(16, 32)
    x = torch.randn(1, 70, 16)
    got = glu_from_linear_halves(lin, x)
    ref = torch.nn.functional.glu(lin(x), dim=-1)
    if got.shape != ref.shape or not torch.allclose(got, ref, atol=1e-5):
        _fail("glu_from_linear_halves != linear then F.glu")


def test_glu_from_linear_halves_export_has_no_activation_slice() -> None:
    class _Glu(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = torch.nn.Linear(8, 16)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return glu_from_linear_halves(self.lin, x)

    ep = torch.export.export(_Glu(), (torch.randn(1, 4, 8),), strict=False)
    targets = [str(n.target) for n in ep.graph.nodes]
    if any("slice" in t and "linear" not in t for t in targets):
        _fail(f"activation slice still in halves GLU export: {targets}")


def test_glu_from_linear_halves_matches_clippable() -> None:
    """Gemma4ClippableLinear has no ``.weight``; clips wrap an inner Linear."""

    class _Clippable(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.use_clipped_linears = True
            self.linear = torch.nn.Linear(16, 32, bias=False)
            self.input_min = torch.tensor(-2.0)
            self.input_max = torch.tensor(2.0)
            self.output_min = torch.tensor(-4.0)
            self.output_max = torch.tensor(4.0)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = torch.clamp(x, self.input_min, self.input_max)
            y = self.linear(x)
            return torch.clamp(y, self.output_min, self.output_max)

    torch.manual_seed(1)
    lin = _Clippable()
    x = torch.randn(1, 70, 16) * 3
    got = glu_from_linear_halves(lin, x)
    ref = torch.nn.functional.glu(lin(x), dim=-1)
    if got.shape != ref.shape or not torch.allclose(got, ref, atol=1e-5):
        _fail("glu_from_linear_halves != clipped linear then F.glu")


def test_rotate_half_matmul_matches_cat() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 4, 8)
    half = x.shape[-1] // 2
    ref = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    got = rotate_half_matmul(x)
    if got.shape != ref.shape or not torch.allclose(got, ref):
        _fail("rotate_half_matmul != cat((-x2, x1))")


def test_apply_rope_matmul_matches_hf() -> None:
    import transformers.models.gemma4.modeling_gemma4 as g4

    torch.manual_seed(0)
    x = torch.randn(1, 5, 3, 8)
    cos = torch.randn(1, 5, 8)
    sin = torch.randn(1, 5, 8)
    pos = torch.zeros(1, 5, 2)
    ref = g4.apply_multidimensional_rope(x, cos, sin, pos, unsqueeze_dim=2)
    got = _apply_multidimensional_rope_ane(x, cos, sin, pos, unsqueeze_dim=2)
    if got.shape != ref.shape or not torch.allclose(got, ref, atol=1e-5):
        _fail("2-D RoPE matmul halves != HF split+cat")


def test_recomposition_matches_cat() -> None:
    torch.manual_seed(0)
    freq = torch.randn(1, 8, 2, 4)
    ref = torch.cat([freq[:, :, 0], freq[:, :, 0], freq[:, :, 1], freq[:, :, 1]], dim=-1)
    got = _recomposition_frequencies_ane(None, freq)
    if got.shape != ref.shape or not torch.equal(got, ref):
        _fail(f"recomposition {tuple(got.shape)} != {tuple(ref.shape)}")


def test_rel_pos_ids_float_matches_int64() -> None:
    n_pos = 13
    ref = torch.arange(n_pos - 1, -1, -1, dtype=torch.int64).to(torch.float32)
    got = rel_pos_ids_float(n_pos, device="cpu", dtype=torch.float32)
    if got.dtype != torch.float32:
        _fail(f"dtype {got.dtype}")
    if not torch.equal(got, ref):
        _fail(f"{got} != {ref}")


def test_rel_pos_ids_export_has_no_i64() -> None:
    class _Pos(torch.nn.Module):
        def forward(self, hidden: torch.Tensor) -> torch.Tensor:
            ids = rel_pos_ids_float(13, device=hidden.device, dtype=hidden.dtype)
            return ids[..., None] * hidden[:1, :1, :1]

    ep = torch.export.export(_Pos(), (torch.randn(1, 4, 8),), strict=False)
    for node in ep.graph.nodes:
        val = node.meta.get("val")
        dt = getattr(val, "dtype", None)
        if dt in (torch.int64, torch.int32) or (dt is not None and "int64" in str(dt)):
            _fail(f"export still has integer ids: {node.name} {dt}")


def test_depthwise_conv1d_channels_last_matches_transpose() -> None:
    torch.manual_seed(0)
    channels, kernel, seq = 8, 5, 16
    conv = torch.nn.Conv1d(channels, channels, kernel, groups=channels, bias=False)
    conv.left_pad = kernel - 1
    x = torch.randn(2, seq, channels)
    padded = torch.nn.functional.pad(x.transpose(1, 2), (conv.left_pad, 0))
    ref = conv(padded).transpose(1, 2)
    got = depthwise_conv1d_channels_last(conv, x)
    if got.shape != ref.shape or not torch.allclose(got, ref, atol=1e-5):
        _fail("channels-last depthwise != transpose+conv1d")


def test_depthwise_conv1d_export_has_no_transpose() -> None:
    class _Dw(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv1d(4, 4, 5, groups=4, bias=False)
            self.conv.left_pad = 4

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return depthwise_conv1d_channels_last(self.conv, x)

    ep = torch.export.export(_Dw(), (torch.randn(1, 16, 4),), strict=False)
    bad = [
        str(n.target)
        for n in ep.graph.nodes
        if "permute" in str(n.target) or "transpose" in str(n.target)
    ]
    if bad:
        _fail(f"export still permutes: {bad}")


def test_take_ple_layer_matches_index() -> None:
    torch.manual_seed(0)
    ple = torch.randn(1, 5, 4, 6)
    for layer in range(4):
        got = _take_ple_layer(ple, layer)
        ref = ple[:, :, layer, :]
        if got.shape != ref.shape or not torch.allclose(got, ref, atol=1e-5):
            _fail(f"ple layer {layer}")


def test_take_ple_layer_export_has_no_slice() -> None:
    class _M(torch.nn.Module):
        def forward(self, ple: torch.Tensor) -> torch.Tensor:
            return _take_ple_layer(ple, 2)

    ep = torch.export.export(_M().eval(), (torch.randn(1, 5, 4, 6),), strict=False)
    bad = [str(n.target) for n in ep.graph.nodes if "slice" in str(n.target)]
    if bad:
        _fail(f"slice still in ple take: {bad}")


def test_text_attn_swap_matches_transpose() -> None:
    class _Attn:
        head_dim = 4
        scaling = 1.0

        def __init__(self) -> None:
            self.q_proj = torch.nn.Linear(8, 8, bias=False)
            self.k_proj = torch.nn.Linear(8, 4, bias=False)
            self.v_proj = torch.nn.Linear(8, 4, bias=False)
            self.o_proj = torch.nn.Linear(8, 8, bias=False)
            self.q_norm = torch.nn.Identity()
            self.k_norm = torch.nn.Identity()
            self.v_norm = torch.nn.Identity()

    torch.manual_seed(0)
    attn = _Attn()
    x = torch.randn(1, 5, 8)
    cos = torch.ones(1, 5, 4)
    sin = torch.zeros(1, 5, 4)
    mask = torch.zeros(1, 1, 5, 5)
    forward = _make_attention_forward(1, 5)
    got, _ = forward(attn, x, attention_mask=mask, position_embeddings=(cos, sin))
    q = attn.q_proj(x).view(1, 5, 2, 4).transpose(1, 2)
    k = attn.k_proj(x).view(1, 5, 1, 4).transpose(1, 2)
    v = attn.v_proj(x).view(1, 5, 1, 4).transpose(1, 2)
    # n_rep = 2. Match eager repeat.
    k = k.repeat_interleave(2, dim=1)
    v = v.repeat_interleave(2, dim=1)
    w = torch.softmax(q @ k.transpose(2, 3) + mask, dim=-1)
    ref = attn.o_proj((w @ v).transpose(1, 2).reshape(1, 5, 8))
    if got.shape != ref.shape or not torch.allclose(got, ref, atol=1e-5):
        _fail("text swap_mid attn != transpose+matmul")


def test_used_weight_layout_matches_linear_and_has_no_transpose() -> None:
    class _M(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.square = torch.nn.Linear(8, 8, bias=False)
            self.rect = torch.nn.Linear(8, 4, bias=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.rect(self.square(x))

    torch.manual_seed(0)
    mod = _M().eval()
    x = torch.randn(1, 5, 8)
    with torch.no_grad():
        ref = mod(x).clone()
    bound = bind_used_weight_layout(mod)
    if bound != 2:
        _fail(f"bound {bound}")
    with torch.no_grad():
        got = mod(x)
    if not torch.allclose(got, ref, atol=1e-5):
        _fail("used layout != linear")
    ep = torch.export.export(mod, (x,), strict=False).run_decompositions()
    bad = [
        str(n.target)
        for n in ep.graph.nodes
        if "transpose" in str(n.target) or "permute" in str(n.target)
    ]
    if bad:
        _fail(f"constant transpose still in used-layout export: {bad}")


def test_bake_k_layout_matches_transpose() -> None:
    torch.manual_seed(0)
    for shape in ((1, 4, 320, 256), (1, 4, 320, 512), (1, 2, 5, 4)):
        x = torch.randn(*shape)
        got = bake_k_layout(x)
        ref = x.transpose(-1, -2)
        if got.shape != ref.shape or not torch.equal(got, ref):
            _fail(f"bake_k_layout != transpose {shape}")


def test_bake_k_layout_export_has_no_transpose() -> None:
    class _M(torch.nn.Module):
        def forward(self, key: torch.Tensor) -> torch.Tensor:
            return bake_k_layout(key)

    ep = torch.export.export(_M().eval(), (torch.randn(1, 4, 320, 256),), strict=False)
    bad = [
        str(n.target)
        for n in ep.graph.nodes
        if "transpose" in str(n.target) or "permute" in str(n.target)
    ]
    if bad:
        _fail(f"transpose still in baked K export: {bad}")


def test_text_attn_export_has_no_head_transpose() -> None:
    class _Attn(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head_dim = 4
            self.scaling = 1.0
            self.q_proj = torch.nn.Linear(8, 8, bias=False)
            self.k_proj = torch.nn.Linear(8, 4, bias=False)
            self.v_proj = torch.nn.Linear(8, 4, bias=False)
            self.o_proj = torch.nn.Linear(8, 8, bias=False)
            self.q_norm = torch.nn.Identity()
            self.k_norm = torch.nn.Identity()
            self.v_norm = torch.nn.Identity()
            self._fn = _make_attention_forward(1, 5)

        def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask: torch.Tensor):
            out, _ = self._fn(self, x, attention_mask=mask, position_embeddings=(cos, sin))
            return out

    torch.manual_seed(0)
    mod = _Attn().eval()
    args = (
        torch.randn(1, 5, 8),
        torch.ones(1, 5, 4),
        torch.zeros(1, 5, 4),
        torch.zeros(1, 1, 5, 5),
    )
    ep = torch.export.export(mod, args, strict=False)
    head_transpose = []
    for n in ep.graph.nodes:
        name = str(n.target)
        if "permute" in name:
            head_transpose.append(name)
            continue
        if "transpose" not in name:
            continue
        dims = [a for a in n.args if isinstance(a, int)]
        if dims in ([1, 2], [2, 1]):
            head_transpose.append(f"{name}{dims}")
    if head_transpose:
        _fail(f"head transpose still in text attn export: {head_transpose}")


def test_io_specs() -> None:
    v = tower_io_spec("vision")
    if v["outputs"]["soft_tokens"] != [1, VISION_SOFT_TOKENS, 512]:
        _fail(str(v))
    if v.get("output_dtypes", {}).get("soft_tokens") != "float16":
        _fail("vision out dtype")
    if v.get("input_dtypes", {}).get("pixel_position_ids") != "float16":
        _fail("vision pos dtype")
    if tower_io_spec("audio")["outputs"]["soft_tokens"] != [1, 1, AUDIO_SOFT_TOKENS, 512]:
        _fail("audio tokens")
    if tower_io_spec("audio").get("output_dtypes", {}).get("soft_tokens") != "float16":
        _fail("audio out dtype")
    if tower_io_spec("text")["outputs"]["embedding"] != [1, 768]:
        _fail("text embed")


def main() -> int:
    tests = [
        test_vision_example_shape,
        test_audio_example_shape,
        test_gather_seq_windows_matches_unfold,
        test_gather_seq_windows_export_has_no_unfold,
        test_slice_seq_windows_matches_unfold,
        test_slice_seq_windows_export_has_no_gather,
        test_blocked_additive_mask_shape_and_pad,
        test_blocked_additive_mask_export_has_no_gather,
        test_blocked_additive_mask_export_stays_float,
        test_embedding_from_int_indices_matches_embedding,
        test_embedding_from_int_indices_export_has_no_i64,
        test_swap_mid_dims_matches_permute,
        test_swap_mid_dims_export_has_no_5d_permute,
        test_nchw_nhwc_matches_permute,
        test_rel_pos_ids_float_matches_int64,
        test_rel_pos_ids_export_has_no_i64,
        test_depthwise_conv1d_channels_last_matches_transpose,
        test_depthwise_conv1d_export_has_no_transpose,
        test_vision_attn_matches_transpose_matmul,
        test_vision_attn_export_has_no_sdpa,
        test_rel_shift_matmul_matches_hf,
        test_rel_shift_static_matches_hf,
        test_rel_shift_baked_matches_hf,
        test_rel_shift_baked_export_has_no_slice,
        test_stride_select_matches_slice,
        test_stride_select_export_has_no_slice,
        test_prefix_rows_matches_slice,
        test_prefix_rows_export_has_no_slice,
        test_bound_glu_halves_export_has_no_split,
        test_glu_split_last_matches_glu,
        test_glu_from_linear_halves_matches_linear_glu,
        test_glu_from_linear_halves_export_has_no_activation_slice,
        test_glu_from_linear_halves_matches_clippable,
        test_rotate_half_matmul_matches_cat,
        test_apply_rope_matmul_matches_hf,
        test_recomposition_matches_cat,
        test_take_ple_layer_matches_index,
        test_take_ple_layer_export_has_no_slice,
        test_text_attn_swap_matches_transpose,
        test_used_weight_layout_matches_linear_and_has_no_transpose,
        test_bake_k_layout_matches_transpose,
        test_bake_k_layout_export_has_no_transpose,
        test_text_attn_export_has_no_head_transpose,
        test_io_specs,
    ]
    for fn in tests:
        fn()
        print(f"OK {fn.__name__}")
    print(f"OK: {len(tests)} coreai-tower checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
