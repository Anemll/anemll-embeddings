"""Export / convert EmbeddingGemma 2 towers to Core AI / ANE.

Heavy stacks (sentence-transformers, the text wrapper) load on first use so
``api.coreai_host`` can be imported with torch alone.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "DEFAULT_MODEL",
    "TEXT_ONLY_CONFIG_KWARGS",
    "load_sentence_transformer",
    "resolve_dtype_device",
    "EmbeddingGemma2Wrapper",
    "masked_mean_pool",
]


def __getattr__(name: str) -> Any:
    if name in {
        "DEFAULT_MODEL",
        "TEXT_ONLY_CONFIG_KWARGS",
        "load_sentence_transformer",
        "resolve_dtype_device",
    }:
        from . import load_text_model as mod

        return getattr(mod, name)
    if name in {"EmbeddingGemma2Wrapper", "masked_mean_pool"}:
        from . import embed_wrapper as mod

        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
