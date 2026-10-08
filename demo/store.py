"""Small cosine store: one ``index.npz`` matrix plus ``index.json`` metadata.

Vectors are L2-normalized, so cosine is a dot product. Media files live
beside the index under ``media/`` and are served back to the pages.
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .types import DIM


def _l2(vector: np.ndarray, dim: int) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    if arr.shape != (dim,):
        raise ValueError(f"expected a {dim}-d vector, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("embedding is not finite")
    norm = float(np.linalg.norm(arr))
    if norm < 1e-8:
        raise ValueError("embedding norm is zero")
    return (arr / norm).astype(np.float32, copy=False)


def _now() -> str:
    return datetime.now(UTC).isoformat()


class VectorStore:
    def __init__(self, data_dir: Path, *, dim: int = DIM) -> None:
        self.dir = Path(data_dir)
        self.media_dir = self.dir / "media"
        self.dim = int(dim)
        self._lock = threading.Lock()
        self.items: list[dict[str, Any]] = []
        self.vectors = np.zeros((0, self.dim), dtype=np.float32)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.media_dir.mkdir(parents=True, exist_ok=True)
        self.load()

    def load(self) -> None:
        meta_path = self.dir / "index.json"
        vec_path = self.dir / "index.npz"
        if not meta_path.is_file() and not vec_path.is_file():
            return
        if not meta_path.is_file() or not vec_path.is_file():
            raise RuntimeError(f"incomplete index in {self.dir} (need index.json and index.npz)")
        items = json.loads(meta_path.read_text())
        vectors = np.load(vec_path)["vectors"].astype(np.float32, copy=False)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise RuntimeError(f"index vectors shape {vectors.shape} != [N, {self.dim}]")
        if len(items) != int(vectors.shape[0]):
            raise RuntimeError("index.json rows and index.npz vectors disagree")
        self.items = list(items)
        self.vectors = vectors

    def save(self) -> None:
        meta_path = self.dir / "index.json"
        vec_path = self.dir / "index.npz"
        meta_path.write_text(json.dumps(self.items, indent=2))
        np.savez(vec_path, vectors=self.vectors)

    def __len__(self) -> int:
        return len(self.items)

    def add(
        self,
        vector: np.ndarray,
        *,
        modality: str,
        label: str | None = None,
        text: str | None = None,
        session: str | None = None,
        t_start: float | None = None,
        t_end: float | None = None,
        credit: str | None = None,
        source: str | None = None,
        filename: str | None = None,
    ) -> dict[str, Any]:
        vec = _l2(vector, self.dim)
        item: dict[str, Any] = {
            "id": uuid.uuid4().hex[:12],
            "modality": modality,
            "label": label or None,
            "text": text or None,
            "created_at": _now(),
            "session": session or None,
            "t_start": None if t_start is None else float(t_start),
            "t_end": None if t_end is None else float(t_end),
            "credit": credit or None,
            "source": source or None,
            "file": filename,
        }
        with self._lock:
            if self.vectors.shape[0] == 0:
                self.vectors = vec.reshape(1, self.dim)
            else:
                self.vectors = np.concatenate([self.vectors, vec.reshape(1, self.dim)], axis=0)
            self.items.append(item)
            self.save()
        return self.public(item)

    def delete(self, item_id: str) -> bool:
        with self._lock:
            idx = next((i for i, row in enumerate(self.items) if row["id"] == item_id), None)
            if idx is None:
                return False
            row = self.items.pop(idx)
            self.vectors = np.delete(self.vectors, idx, axis=0)
            self.save()
        name = row.get("file")
        if name:
            path = self.media_dir / name
            if path.is_file():
                path.unlink()
        return True

    def get(self, item_id: str) -> dict[str, Any] | None:
        for row in self.items:
            if row["id"] == item_id:
                return row
        return None

    def list_items(
        self,
        *,
        modality: str | None = None,
        session: str | None = None,
    ) -> list[dict[str, Any]]:
        rows = []
        for item in self.items:
            if modality and item.get("modality") != modality:
                continue
            if session and item.get("session") != session:
                continue
            rows.append(self.public(item))
        return rows

    def search(
        self,
        vector: np.ndarray,
        k: int,
        *,
        modality: str | None = None,
        session: str | None = None,
    ) -> list[dict[str, Any]]:
        query = _l2(vector, self.dim)
        with self._lock:
            if self.vectors.shape[0] == 0:
                return []
            scores = self.vectors @ query
            order = np.argsort(-scores)
            hits: list[dict[str, Any]] = []
            for index in order:
                item = self.items[int(index)]
                if modality and item.get("modality") != modality:
                    continue
                if session and item.get("session") != session:
                    continue
                row = self.public(item)
                row["score"] = float(scores[int(index)])
                hits.append(row)
                if len(hits) >= int(k):
                    break
        return hits

    def public(self, item: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": item["id"],
            "modality": item["modality"],
            "label": item.get("label"),
            "text": item.get("text"),
            "created_at": item.get("created_at"),
            "session": item.get("session"),
            "t_start": item.get("t_start"),
            "t_end": item.get("t_end"),
            "credit": item.get("credit"),
            "source": item.get("source"),
        }
        if item.get("file"):
            out["media_url"] = f"/media/{item['id']}"
        return out

    def media_path(self, item_id: str) -> Path | None:
        item = self.get(item_id)
        if not item or not item.get("file"):
            return None
        path = self.media_dir / str(item["file"])
        if not path.is_file():
            return None
        return path
