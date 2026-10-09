"""Unit checks for ``softmax_unfused`` (export-time attention softmax).

The helper must match ``torch.softmax`` on the last dim, stay finite in fp16
for attention-sized scores (up to ~20), and trace without a ``softmax`` or
``div`` op: either one lets MPSGraph re-fuse the attention into
``mps_spi.sdpa``, which the macOS 27.2 ANE pre-check rejects.
See ``docs/M5_ANE_SOFTMAX_FIX.md``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from model.trace_patches import softmax_unfused  # noqa: E402


@pytest.mark.parametrize("shape", [(7,), (3, 5), (1, 2, 16, 32), (1, 4, 320, 320)])
def test_shape_and_rows_sum_to_one(shape: tuple[int, ...]) -> None:
    torch.manual_seed(0)
    x = torch.randn(shape) * 4
    y = softmax_unfused(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert torch.all(y >= 0)
    torch.testing.assert_close(y.sum(-1), torch.ones(shape[:-1]), atol=1e-5, rtol=0)


@pytest.mark.parametrize("scale", [1.0, 8.0, 20.0, 60.0])
def test_matches_torch_softmax_fp32(scale: float) -> None:
    torch.manual_seed(1)
    x = torch.randn(2, 4, 64, 320) * scale
    torch.testing.assert_close(
        softmax_unfused(x), torch.softmax(x, dim=-1), atol=1e-6, rtol=1e-5
    )


def test_masked_scores_fp32() -> None:
    torch.manual_seed(2)
    x = torch.randn(1, 2, 32, 64) * 10
    x[..., 40:] = -1e9  # additive key mask
    y = softmax_unfused(x)
    torch.testing.assert_close(y, torch.softmax(x, dim=-1), atol=1e-6, rtol=1e-5)
    assert torch.all(y[..., 40:] == 0)


@pytest.mark.parametrize("scale", [1.0, 8.0, 20.0])
def test_matches_torch_softmax_fp16(scale: float) -> None:
    torch.manual_seed(3)
    x32 = torch.randn(2, 4, 64, 320) * scale
    # Real attention rows peak at 12.8-20.7; force a row max around 20.
    x32[..., 0] = 20.0
    x = x32.to(torch.float16)
    y = softmax_unfused(x)
    assert y.dtype == torch.float16
    assert torch.isfinite(y).all()
    ref = torch.softmax(x.to(torch.float32), dim=-1)
    torch.testing.assert_close(y.to(torch.float32), ref, atol=2e-3, rtol=1e-2)


def test_row_max_subtract_needed_in_fp16() -> None:
    """Without the row max, fp16 ``exp`` overflows above ~11 (exp(20) > 65504)."""
    x = torch.tensor([[20.0, 19.0, 0.0]], dtype=torch.float16)
    naive = torch.exp(x) * torch.reciprocal(torch.exp(x).sum(-1, keepdim=True))
    assert not torch.isfinite(naive).all()
    y = softmax_unfused(x)
    assert torch.isfinite(y).all()
    torch.testing.assert_close(
        y.to(torch.float32), torch.softmax(x.to(torch.float32), -1), atol=2e-3, rtol=1e-2
    )


def _traced_kinds(fn, *args) -> set[str]:
    traced = torch.jit.trace(fn, args, check_trace=False)
    return {node.kind() for node in traced.inlined_graph.nodes()}


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_traced_graph_has_no_softmax_or_divide() -> None:
    kinds = _traced_kinds(softmax_unfused, torch.randn(1, 2, 8, 16))
    forbidden = {k for k in kinds if "softmax" in k or k.startswith("aten::div")}
    assert not forbidden, f"re-fusable ops in trace: {sorted(forbidden)}"
    assert "aten::reciprocal" in kinds
    assert "aten::exp" in kinds
    assert "aten::amax" in kinds or "aten::max" in kinds


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_attention_chain_traces_without_softmax() -> None:
    """``q @ K^T -> softmax_unfused -> @ V`` keeps the softmax spelled out."""

    def attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return torch.matmul(softmax_unfused(torch.matmul(q, k.transpose(-1, -2))), v)

    q, k, v = (torch.randn(1, 2, 8, 16) for _ in range(3))
    kinds = _traced_kinds(attn, q, k, v)
    assert not {k for k in kinds if "softmax" in k or "scaled_dot_product" in k}
    assert not {k for k in kinds if k.startswith("aten::div")}
    ref = torch.matmul(torch.softmax(torch.matmul(q, k.transpose(-1, -2)), -1), v)
    torch.testing.assert_close(attn(q, k, v), ref, atol=1e-6, rtol=1e-5)
