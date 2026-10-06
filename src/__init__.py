"""anemll-embeddings — EmbeddingGemma 2 → Core ML / ANE helpers."""

from .load_text_model import (
    DEFAULT_MODEL,
    TEXT_ONLY_CONFIG_KWARGS,
    load_sentence_transformer,
    resolve_dtype_device,
)
from .embed_wrapper import EmbeddingGemma2Wrapper, masked_mean_pool

__all__ = [
    "DEFAULT_MODEL",
    "TEXT_ONLY_CONFIG_KWARGS",
    "load_sentence_transformer",
    "resolve_dtype_device",
    "EmbeddingGemma2Wrapper",
    "masked_mean_pool",
]
