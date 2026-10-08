"""Public runtime for EmbeddingGemma 2 on Core AI / ANE.

Load the three ``.aimodel`` towers and embed text, images, or audio. Each
call returns one L2-normalized 768-d vector. Compare vectors with
:func:`cosine`.

    from api import Embedder, cosine

    embedder = Embedder(compute="ane")
    query = embedder.embed_text("a red fox")
    doc = embedder.embed_text("a red fox in snow", role="document")
    score = cosine(query, doc)   # float in [-1, 1]
    embedder.close()

Copy-paste examples: `api/README.md`. The demo pages are the same API
wired to a browser (`demo/server.py`, `demo/alert_routes.py`).
"""

from __future__ import annotations

from typing import Any

from api.similarity import cosine, similarity
from api.types import DIM

__all__ = ["DIM", "Embedder", "cosine", "similarity"]


def __getattr__(name: str) -> Any:
    # Embedder pulls the Core AI host (torch). Keep ``from api import cosine`` light.
    if name == "Embedder":
        from api.embedder import Embedder

        return Embedder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
