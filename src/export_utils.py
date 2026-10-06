"""Shared helpers for T4 TorchScript export and T5 Core ML convert.

Mask construction is tensor-only (no HF `create_*_mask` Python factories) so
`torch.jit.trace` at a fixed S records a real pad+window graph instead of the
`None` skip path HF takes on an all-ones mask.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy
import torch
import torch.nn.functional as F

try:
    import coremltools as _coremltools
except ImportError:
    _coremltools = None
try:
    import sentence_transformers as _sentence_transformers
except ImportError:
    _sentence_transformers = None
try:
    import transformers as _transformers
except ImportError:
    _transformers = None

# Finite additive bias (ANE lesson: -inf / finfo.min is a placement/NaN hazard).
# -1e4 is enough to zero softmax while staying in a later FP16 range.
MASK_NEG = -1.0e4
DEFAULT_SEQ_LEN = 512
DEFAULT_ARTIFACTS = Path("/Volumes/Models/anemll-embeddings/artifacts")


def artifacts_root() -> Path:
    raw = os.environ.get("ANEMLL_EMBEDDINGS_ARTIFACTS")
    return Path(raw) if raw else DEFAULT_ARTIFACTS


def package_stem(seq_len: int) -> str:
    return f"embeddinggemma2-text-s{int(seq_len)}"


def artifact_subdir(seq_len: int, root: Path | None = None) -> Path:
    base = root if root is not None else artifacts_root()
    return base / package_stem(seq_len)


def git_sha(repo_root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def host_versions() -> dict[str, Any]:
    packages = {
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "transformers": (
            _transformers.__version__ if _transformers is not None else "missing"
        ),
        "sentence_transformers": (
            _sentence_transformers.__version__
            if _sentence_transformers is not None
            else "missing"
        ),
        "coremltools": (
            _coremltools.__version__ if _coremltools is not None else "missing"
        ),
    }
    return {
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "packages": packages,
    }


def pad_to_seq_len(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    seq_len: int,
    pad_token_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad or truncate a batch to a fixed S. Mask pads are 0."""
    if input_ids.ndim != 2 or attention_mask.ndim != 2:
        raise ValueError(
            f"expected [B, S] ids/mask; got {tuple(input_ids.shape)} / "
            f"{tuple(attention_mask.shape)}"
        )
    if input_ids.shape != attention_mask.shape:
        raise ValueError(
            f"ids/mask shape mismatch: {tuple(input_ids.shape)} vs "
            f"{tuple(attention_mask.shape)}"
        )
    cur = int(input_ids.shape[-1])
    if cur == seq_len:
        return input_ids, attention_mask
    if cur > seq_len:
        return input_ids[:, :seq_len], attention_mask[:, :seq_len]
    pad = seq_len - cur
    return (
        F.pad(input_ids, (0, pad), value=int(pad_token_id)),
        F.pad(attention_mask, (0, pad), value=0),
    )


def example_trace_inputs(
    seq_len: int,
    *,
    batch: int = 1,
    pad_last: int = 8,
    vocab_hi: int = 1024,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed-S example with trailing pads so the mask graph is not skipped."""
    if pad_last < 1:
        raise ValueError("pad_last must be >= 1 so tracing records a pad mask")
    if pad_last >= seq_len:
        raise ValueError(f"pad_last={pad_last} must be < seq_len={seq_len}")
    valid = seq_len - pad_last
    ids = torch.randint(2, vocab_hi, (batch, seq_len), dtype=torch.int32, device=device)
    mask = torch.zeros(batch, seq_len, dtype=torch.int32, device=device)
    mask[:, :valid] = 1
    ids[:, valid:] = 0
    return ids, mask


def additive_full_attention_bias(
    attention_mask: torch.Tensor,
    dtype: torch.dtype,
    *,
    neg: float = MASK_NEG,
) -> torch.Tensor:
    """2D pad mask [B, S] → additive 4D bias [B, 1, S, S] (keys only)."""
    batch, seq_len = attention_mask.shape
    key_ok = attention_mask.to(dtype=dtype)[:, None, None, :]
    return ((1.0 - key_ok) * float(neg)).expand(batch, 1, seq_len, seq_len).contiguous()


def additive_sliding_attention_bias(
    attention_mask: torch.Tensor,
    sliding_window: int,
    dtype: torch.dtype,
    *,
    neg: float = MASK_NEG,
) -> torch.Tensor:
    """Pad + inclusive |q-k| <= window, matching HF bidirectional overlay."""
    _batch, seq_len = attention_mask.shape
    pos = torch.arange(seq_len, device=attention_mask.device)
    window_ok = (pos[:, None] - pos[None, :]).abs() <= int(sliding_window)
    key_ok = attention_mask.to(dtype=torch.bool)[:, None, :]
    keep = window_ok.view(1, seq_len, seq_len) & key_ok
    return (~keep).to(dtype=dtype).unsqueeze(1) * float(neg)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def force_eager_attention(module: torch.nn.Module) -> None:
    """Disable SDPA/flash so the traced graph is the eager matmul softmax path."""
    cfg = getattr(module, "config", None)
    if cfg is not None:
        if hasattr(cfg, "_attn_implementation"):
            cfg._attn_implementation = "eager"
        if hasattr(cfg, "output_attentions"):
            cfg.output_attentions = False
        if hasattr(cfg, "output_hidden_states"):
            cfg.output_hidden_states = False
    for child in module.children():
        force_eager_attention(child)
