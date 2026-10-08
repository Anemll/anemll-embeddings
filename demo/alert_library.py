"""On-disk alert frames, sounds, Sparky's reference, and an embedding cache.

Fetched files live under the alert directory (default
``~/.anemll-embeddings/alert``). Overrides from the page (a dropped photo,
replacement reference shots) live in ``overrides/`` and win over the fetch.
Embeddings are reused by SHA-256 so a new text query does not re-run vision.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import threading
import wave
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from api.types import EmbedResult
from demo.alert_catalog import REFERENCES, catalog_items
from demo.alert_score import l2
from demo.media_io import AUDIO_SR, load_audio_bytes, read_wav_bytes, resample_linear, save_jpeg

_ID = re.compile(r"^[a-z0-9-]{1,40}$")


def valid_id(value: str) -> str:
    text = str(value or "").strip()
    if not _ID.fullmatch(text):
        raise ValueError(f"invalid id {value!r}")
    return text


class AlertLibrary:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._lock = threading.Lock()
        self._vectors: dict[str, np.ndarray] = {}
        self._load_cache()

    def manifest_by_id(self) -> dict[str, dict[str, Any]]:
        path = self.root / "manifest.json"
        if not path.is_file():
            return {}
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}
        rows = payload.get("items") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            return {}
        found: dict[str, dict[str, Any]] = {}
        for row in rows:
            if isinstance(row, dict) and row.get("id"):
                found[str(row["id"])] = row
        return found

    def media_path(self, item_id: str) -> Path | None:
        """Resolved image/audio for a catalog id, override first."""
        item_id = valid_id(item_id)
        override = self._override_media(item_id)
        if override is not None:
            return override
        row = self.manifest_by_id().get(item_id)
        if row and row.get("file"):
            candidate = (self.root / str(row["file"])).resolve()
            if self._inside(candidate) and candidate.is_file():
                return candidate
        for item in catalog_items():
            if item["id"] != item_id:
                continue
            candidate = (self.root / item["file"]).resolve()
            if self._inside(candidate) and candidate.is_file():
                return candidate
        return None

    def reference_paths(self, rule_id: str) -> list[Path]:
        """1–3 reference photos. Custom uploads replace the stock Sparky shot."""
        rule_id = valid_id(rule_id)
        custom_dir = self.root / "overrides" / "references" / rule_id
        if custom_dir.is_dir():
            files = [
                path
                for path in sorted(custom_dir.iterdir())
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            ]
            if files:
                return files[:3]
        if rule_id == "sparky":
            stock = self.media_path("sparky-ref")
            if stock is not None:
                return [stock]
        for ref in REFERENCES:
            if ref.get("rule_id") == rule_id:
                path = self.media_path(ref["id"])
                if path is not None:
                    return [path]
        return []

    def credit_for(self, item_id: str) -> dict[str, Any]:
        row = self.manifest_by_id().get(item_id) or {}
        return {
            "credit": row.get("credit"),
            "license": row.get("license"),
            "source": row.get("source"),
            "creator": row.get("creator"),
        }

    def save_frame(self, item_id: str, image: Image.Image) -> Path:
        item_id = valid_id(item_id)
        path = self.root / "overrides" / "frames" / f"{item_id}.jpg"
        save_jpeg(image, path, max_edge=1280)
        return path

    def save_sound(self, item_id: str, samples: np.ndarray, rate: int) -> Path:
        item_id = valid_id(item_id)
        path = self.root / "overrides" / "sounds" / f"{item_id}.wav"
        _write_wav(path, samples, rate)
        return path

    def save_references(self, rule_id: str, images: list[Image.Image]) -> list[Path]:
        rule_id = valid_id(rule_id)
        if not images or len(images) > 3:
            raise ValueError("upload 1 to 3 reference photos")
        dest = self.root / "overrides" / "references" / rule_id
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True, exist_ok=True)
        saved = []
        for index, image in enumerate(images, start=1):
            path = dest / f"{index}.jpg"
            save_jpeg(image, path, max_edge=1280)
            saved.append(path)
        return saved

    def clear_overrides(self) -> None:
        folder = self.root / "overrides"
        if folder.exists():
            shutil.rmtree(folder)

    def is_cached_file(self, path: Path) -> bool:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with self._lock:
            return digest in self._vectors

    def embed_file(
        self,
        path: Path,
        modality: str,
        embed: Callable[[Path, str], tuple[np.ndarray, float]],
        *,
        fresh: bool = False,
    ) -> tuple[np.ndarray, float]:
        """Return a unit vector and the embed time. A cache hit reports 0 ms.

        ``fresh`` skips the cache lookup (a click re-runs the model) and then
        refreshes the cached vector.
        """
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        with self._lock:
            hit = None if fresh else self._vectors.get(digest)
        if hit is not None:
            return hit, 0.0
        vector, latency_ms = embed(path, modality)
        unit = l2(vector)
        with self._lock:
            self._vectors[digest] = unit
            self._save_cache_locked()
        return unit, float(latency_ms)

    def embed_text(self, text: str, role: str, embed: Callable[[str, str], tuple[np.ndarray, float]]) -> tuple[np.ndarray, float]:
        digest = hashlib.sha256(f"{role}\n{text}".encode("utf-8")).hexdigest()
        key = f"text:{digest}"
        with self._lock:
            hit = self._vectors.get(key)
        if hit is not None:
            return hit, 0.0
        vector, latency_ms = embed(text, role)
        unit = l2(vector)
        with self._lock:
            self._vectors[key] = unit
            self._save_cache_locked()
        return unit, float(latency_ms)

    def _override_media(self, item_id: str) -> Path | None:
        frames = self.root / "overrides" / "frames" / f"{item_id}.jpg"
        sounds = self.root / "overrides" / "sounds" / f"{item_id}.wav"
        if frames.is_file():
            return frames
        if sounds.is_file():
            return sounds
        return None

    def _inside(self, path: Path) -> bool:
        try:
            path.relative_to(self.root.resolve())
        except ValueError:
            return False
        return True

    def _load_cache(self) -> None:
        vec_path = self.root / "embed-cache.npz"
        meta_path = self.root / "embed-cache.json"
        if not vec_path.is_file() or not meta_path.is_file():
            return
        try:
            keys = json.loads(meta_path.read_text())
            vectors = np.load(vec_path)["vectors"].astype(np.float32, copy=False)
        except (OSError, json.JSONDecodeError, ValueError, KeyError):
            return
        if not isinstance(keys, list) or len(keys) != int(vectors.shape[0]):
            return
        self._vectors = {str(key): l2(row) for key, row in zip(keys, vectors)}

    def _save_cache_locked(self) -> None:
        if not self._vectors:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        keys = list(self._vectors)
        matrix = np.stack([self._vectors[key] for key in keys], axis=0)
        np.savez(self.root / "embed-cache.npz", vectors=matrix)
        (self.root / "embed-cache.json").write_text(json.dumps(keys))


def _from_embed(backend: Any, value: Any, timing: dict[str, Any] | None) -> tuple[np.ndarray, float]:
    """Read a public ``Embedder`` vector (and ``.last``) or a raw ``EmbedResult``."""
    if isinstance(value, EmbedResult):
        if timing is not None:
            timing.update(value.extra or {})
        return np.asarray(value.vector, dtype=np.float32), float(value.latency_ms)
    last = getattr(backend, "last", None)
    if isinstance(last, EmbedResult) and timing is not None:
        timing.update(last.extra or {})
    latency = float(last.latency_ms) if isinstance(last, EmbedResult) else 0.0
    return np.asarray(value, dtype=np.float32), latency


def embed_path(
    path: Path, modality: str, backend: Any, timing: dict[str, Any] | None = None
) -> tuple[np.ndarray, float]:
    """Embed one file through the public ``Embedder`` methods."""
    if modality == "image":
        image = Image.open(path).convert("RGB")
        return _from_embed(backend, backend.embed_image(image), timing)
    if modality == "audio":
        samples, rate = read_wav_bytes(path.read_bytes())
        if samples.ndim > 1:
            samples = samples.mean(axis=-1)
        if rate != AUDIO_SR:
            samples = resample_linear(samples, rate, AUDIO_SR)
            rate = AUDIO_SR
        return _from_embed(
            backend,
            backend.embed_audio(np.ascontiguousarray(samples, dtype=np.float32), rate),
            timing,
        )
    raise ValueError(f"unknown modality {modality}")


def decode_audio_upload(data: bytes, filename: str) -> np.ndarray:
    suffix = Path(filename or "clip.wav").suffix.lower() or ".wav"
    return load_audio_bytes(data, suffix)


def _write_wav(path: Path, samples: np.ndarray, rate: int) -> None:
    pcm = np.clip(np.asarray(samples, dtype=np.float32).reshape(-1), -1.0, 1.0)
    ints = (pcm * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(rate))
        handle.writeframes(ints.tobytes())
