"""Importable EmbeddingGemma 2 runtime.

Load the Core AI ``.aimodel`` towers and embed text, images, or audio:

    from api import Embedder
    embedder = Embedder(artifacts=..., model=..., compute="ane")
    vec = embedder.embed_text("a red fox")
"""

from __future__ import annotations

from typing import Any

__all__ = ["DIM", "Embedder", "EmbedResult"]


def __getattr__(name: str) -> Any:
    if name == "Embedder":
        from .embedder import Embedder

        return Embedder
    if name in {"DIM", "EmbedResult"}:
        from . import types as mod

        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
