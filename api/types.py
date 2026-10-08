"""Result metadata from the last ``Embedder.embed_*`` call.

Most callers only need the ``(768,)`` vector that ``embed_*`` returns.
``EmbedResult`` is the extra timing/placement the demo badges show.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

DIM = 768


@dataclass
class EmbedResult:
    """One L2-normalized embedding plus the timing the UI badge shows."""

    vector: np.ndarray
    latency_ms: float
    backend: str
    modality: str
    placement: str | None = None
    slices: int = 1
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        vec = np.asarray(self.vector, dtype=np.float32).reshape(-1)
        payload: dict[str, Any] = {
            "modality": self.modality,
            "dim": int(vec.shape[0]),
            "vector": vec.tolist(),
            "latency_ms": round(float(self.latency_ms), 3),
            "backend": self.backend,
            "normalized": True,
            "slices": int(self.slices),
        }
        if self.placement:
            payload["placement"] = self.placement
        if self.extra:
            payload["detail"] = self.extra
        return payload


@dataclass
class TowerHealth:
    name: str
    loaded: bool
    warmup_ms: float | None = None
    placement: str | None = None
    simulated: bool = False

    def as_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "name": self.name,
            "loaded": bool(self.loaded),
            "simulated": bool(self.simulated),
        }
        if self.warmup_ms is not None:
            row["warmup_ms"] = round(float(self.warmup_ms), 3)
        if self.placement:
            row["placement"] = self.placement
        return row
