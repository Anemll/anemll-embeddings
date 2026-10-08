"""T2: Text-only EmbeddingGemma 2 loader (Sentence-Transformers).

Loads the local checkpoint with vision/audio configs disabled and a
BF16-or-FP32 dtype policy. FP16 is refused — upstream EmbeddingGemma 2
produces NaNs / silent trash under float16.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from sentence_transformers import SentenceTransformer

def _embeddings_home() -> Path:
    raw = os.environ.get("ANEMLL_EMBEDDINGS_HOME")
    return Path(raw).expanduser() if raw else Path.home() / ".anemll-embeddings"


# Full checkpoint written by scripts/download_export_assets.py.
DEFAULT_MODEL = _embeddings_home() / "embeddinggemma-2-full"
TEXT_ONLY_CONFIG_KWARGS: dict[str, Any] = {
    "vision_config": None,
    "audio_config": None,
}


def ensure_hf_cache_env() -> None:
    """Point the HF caches at ``$ANEMLL_HF_CACHE`` when that is set.

    Unset leaves Hugging Face's own defaults (``HF_HOME`` or
    ``~/.cache/huggingface``) alone. Either way nothing lands in the checkout.
    """
    cache = os.environ.get("ANEMLL_HF_CACHE")
    if cache:
        os.environ.setdefault("HF_HOME", cache)
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", cache)


def resolve_dtype_device(
    prefer_bf16: bool = True,
) -> tuple[torch.dtype, str]:
    """Pick BF16 when usable, else FP32. Never returns float16."""
    if prefer_bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16, "cuda"
    if prefer_bf16 and torch.backends.mps.is_available():
        try:
            t = torch.tensor([1.0], dtype=torch.bfloat16, device="mps")
            _ = (t * 2).item()
            return torch.bfloat16, "mps"
        except Exception:
            return torch.float32, "mps"
    if torch.cuda.is_available():
        return torch.float32, "cuda"
    if torch.backends.mps.is_available():
        return torch.float32, "mps"
    return torch.float32, "cpu"


def _validate_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.float16:
        raise ValueError(
            "FP16 is forbidden for EmbeddingGemma 2 (NaNs / silent trash). "
            "Use torch.bfloat16 or torch.float32."
        )
    if dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(
            f"Unsupported dtype {dtype}; allowed: bfloat16, float32 (never float16)."
        )
    return dtype


def default_model_path() -> Path:
    return Path(os.environ.get("ANEMLL_EMBEDDINGS_MODEL", DEFAULT_MODEL))


def load_sentence_transformer(
    model_path: Path | str | None = None,
    *,
    dtype: torch.dtype | None = None,
    device: str | None = None,
    trust_remote_code: bool = False,
) -> tuple[SentenceTransformer, dict[str, Any]]:
    """Load text-only SentenceTransformer; return (model, load_meta).

    Always passes ``config_kwargs`` with vision/audio disabled so only the
    ~270M text tower is constructed.
    """
    ensure_hf_cache_env()
    path = Path(model_path) if model_path is not None else default_model_path()
    if not path.is_dir():
        raise FileNotFoundError(f"Model path missing: {path}")

    if dtype is None or device is None:
        auto_dtype, auto_device = resolve_dtype_device()
        if dtype is None:
            dtype = auto_dtype
        if device is None:
            device = auto_device
    dtype = _validate_dtype(dtype)

    model = SentenceTransformer(
        str(path),
        trust_remote_code=trust_remote_code,
        device=device,
        config_kwargs=dict(TEXT_ONLY_CONFIG_KWARGS),
        model_kwargs={"torch_dtype": dtype},
    )
    meta: dict[str, Any] = {
        "model_path": str(path.resolve()),
        "torch_dtype": str(dtype).replace("torch.", ""),
        "device": device,
        "text_only": True,
        "config_kwargs": dict(TEXT_ONLY_CONFIG_KWARGS),
    }
    return model, meta


def get_text_backbone(st_model: SentenceTransformer):
    """Return ``EmbeddingGemma2TextModel`` (language_model) from an ST stack."""
    transformer = st_model[0]
    auto = getattr(transformer, "auto_model", None) or getattr(transformer, "model")
    if not hasattr(auto, "language_model"):
        raise TypeError(
            f"Expected EmbeddingGemma2Model with language_model; got {type(auto)}"
        )
    return auto.language_model
