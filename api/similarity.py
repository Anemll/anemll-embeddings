"""Compare L2-normalized 768-d embeddings."""

from __future__ import annotations

import numpy as np

from api.types import DIM


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    """Cosine similarity of two embeddings.

    ``Embedder`` returns L2-normalized ``(768,)`` float32 vectors, so this is
    a dot product in ``[-1, 1]``. Inputs are re-normalized so the score stays
    a true cosine if a caller passes a raw vector.

    Rough guide (Core AI / ANE, not the mock backend):

    - ``1.0`` — same input
    - ``0.85–1.0`` — same thing, different wording or viewpoint
    - ``0.60–0.85`` — related (UPS photo vs ``a brown UPS delivery truck`` 0.727, bark vs ``a dog barking`` 0.721)
    - near ``0`` — unrelated
    - negative — opposite directions (rare for this model)

    Raises:
        ValueError: if a vector is not 768-d, not finite, or zero-norm.
    """
    return float(np.dot(_unit(left), _unit(right)))


def similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Alias for :func:`cosine`."""
    return cosine(left, right)


def _unit(vector: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    if arr.shape != (DIM,):
        raise ValueError(f"expected a {DIM}-d vector, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("embedding is not finite")
    norm = float(np.linalg.norm(arr))
    if norm < 1e-8:
        raise ValueError("embedding norm is zero")
    return (arr / norm).astype(np.float32, copy=False)
