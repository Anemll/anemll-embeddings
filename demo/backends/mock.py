"""Deterministic stand-in embeddings.

Same input always yields the same unit vector. Text uses token hashing, so
shared words rank together. Images use a 16×16 RGB layout (exactly 768
values). Audio uses a fixed-length log spectrum. Nothing here is semantic
across modalities — that is what the Core AI backend is for.
"""

from __future__ import annotations

import hashlib
import re
import time

import numpy as np
from PIL import Image

from demo.types import DIM, EmbedResult, TowerHealth

_TOKEN = re.compile(r"[a-z0-9]+")


def _unit(vector: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    if arr.shape != (DIM,):
        raise RuntimeError(f"mock vector {arr.shape}")
    norm = float(np.linalg.norm(arr))
    if norm < 1e-8:
        out = np.zeros(DIM, dtype=np.float32)
        out[0] = 1.0
        return out
    return (arr / norm).astype(np.float32, copy=False)


def _token_vector(text: str) -> np.ndarray:
    vec = np.zeros(DIM, dtype=np.float64)
    tokens = _TOKEN.findall(text.lower()) or ["empty"]
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        for i in range(0, 16, 2):
            index = int.from_bytes(digest[i : i + 2], "little") % DIM
            sign = 1.0 if digest[i] % 2 == 0 else -1.0
            vec[index] += sign
    return _unit(vec)


class MockBackend:
    """CPU stand-in. Reports three simulated towers so /health has the same shape."""

    name = "mock"
    dim = DIM

    def __init__(self) -> None:
        self.placement = "mock"
        self._towers: list[TowerHealth] = []
        self._ready = False

    def warmup(self) -> dict:
        if self._ready:
            return self.health()
        timings: list[tuple[str, float]] = []
        samples = (
            ("text_embeds_s320", lambda: self.embed_text("warmup", role="document")),
            ("vision_s280", lambda: self.embed_image(Image.new("RGB", (8, 8), (40, 80, 120)))),
            ("audio_s280", lambda: self.embed_audio(np.zeros(1600, dtype=np.float32), 16000)),
        )
        for name, fn in samples:
            t0 = time.perf_counter()
            fn()
            timings.append((name, (time.perf_counter() - t0) * 1000.0))
        self._towers = [
            TowerHealth(name=name, loaded=True, warmup_ms=ms, placement="mock", simulated=True)
            for name, ms in timings
        ]
        self._ready = True
        return self.health()

    def health(self) -> dict:
        return {
            "backend": self.name,
            "placement": self.placement,
            "dim": self.dim,
            "towers": [row.as_dict() for row in self._towers],
        }

    def embed_text(self, text: str, *, role: str) -> EmbedResult:
        t0 = time.perf_counter()
        # A role token keeps query and document prefixes from being identical,
        # the way SearchQuery / Document prompts do on the real model.
        vector = _token_vector(f"{role} {text}")
        return EmbedResult(
            vector=vector,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            backend=self.name,
            modality="text",
            placement=self.placement,
        )

    def embed_image(self, image: Image.Image) -> EmbedResult:
        t0 = time.perf_counter()
        small = np.asarray(image.convert("RGB").resize((16, 16)), dtype=np.float32)
        vector = _unit(small.reshape(-1) / 255.0)
        return EmbedResult(
            vector=vector,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            backend=self.name,
            modality="image",
            placement=self.placement,
        )

    def embed_audio(self, wav: np.ndarray, sample_rate: int) -> EmbedResult:
        t0 = time.perf_counter()
        samples = np.asarray(wav, dtype=np.float32).reshape(-1)
        if samples.size < 32:
            samples = np.pad(samples, (0, 32 - samples.size))
        target = 4096
        src_x = np.linspace(0.0, 1.0, samples.size, endpoint=False)
        grid = np.linspace(0.0, 1.0, target, endpoint=False)
        window = np.interp(grid, src_x, samples).astype(np.float32) * np.hanning(target)
        mag = np.abs(np.fft.rfft(window)).astype(np.float32)
        edges = np.linspace(0, mag.size, DIM + 1).astype(int)
        pooled = np.empty(DIM, dtype=np.float32)
        for i in range(DIM):
            pooled[i] = float(mag[edges[i] : max(edges[i + 1], edges[i] + 1)].mean())
        vector = _unit(np.log1p(pooled))
        _ = sample_rate  # already resampled by the server
        return EmbedResult(
            vector=vector,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            backend=self.name,
            modality="audio",
            placement=self.placement,
            slices=1,
        )

    def close(self) -> None:
        return None
