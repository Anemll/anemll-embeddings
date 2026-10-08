"""Full 740M EmbeddingGemma 2 loader (text + vision + audio).

Text-only T2 stays in ``load_text_model.py``. This loader leaves both
modality configs enabled unless the caller drops one on purpose.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from sentence_transformers import SentenceTransformer

from .load_text_model import (
    _validate_dtype,
    default_model_path,
    ensure_hf_cache_env,
    get_text_backbone,
    resolve_dtype_device,
)


def modality_config_kwargs(*, vision: bool = True, audio: bool = True) -> dict[str, Any]:
    """HF ``config_kwargs`` for selective encoder loading.

    Full multimodal is ``{}`` (740M). Omitting vision and audio matches T2.
    """
    kwargs: dict[str, Any] = {}
    if not vision:
        kwargs["vision_config"] = None
    if not audio:
        kwargs["audio_config"] = None
    return kwargs


def load_multimodal_sentence_transformer(
    model_path: Path | str | None = None,
    *,
    dtype: torch.dtype | None = None,
    device: str | None = None,
    vision: bool = True,
    audio: bool = True,
    trust_remote_code: bool = True,
) -> tuple[SentenceTransformer, dict[str, Any]]:
    """Load EmbeddingGemma 2 with vision/audio towers unless disabled."""
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
    config_kwargs = modality_config_kwargs(vision=vision, audio=audio)

    model = SentenceTransformer(
        str(path),
        trust_remote_code=trust_remote_code,
        device=device,
        config_kwargs=dict(config_kwargs),
        model_kwargs={"torch_dtype": dtype},
    )
    meta: dict[str, Any] = {
        "model_path": str(path.resolve()),
        "torch_dtype": str(dtype).replace("torch.", ""),
        "device": device,
        "text_only": not vision and not audio,
        "vision": bool(vision),
        "audio": bool(audio),
        "effective_params": (
            "740M"
            if vision and audio
            else "440M"
            if vision
            else "570M"
            if audio
            else "270M"
        ),
        "config_kwargs": dict(config_kwargs),
        "target": "coreai",
    }
    return model, meta


def get_auto_model(st_model: SentenceTransformer):
    """Return the HF ``EmbeddingGemma2Model`` from an ST stack."""
    transformer = st_model[0]
    auto = getattr(transformer, "auto_model", None) or getattr(transformer, "model")
    if auto is None:
        raise TypeError(f"ST module 0 has no auto_model: {type(transformer)}")
    return auto


def get_language_model(st_model: SentenceTransformer):
    """Same as T2 ``get_text_backbone`` (language_model)."""
    return get_text_backbone(st_model)
