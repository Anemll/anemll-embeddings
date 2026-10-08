"""PyTorch reference: the full EmbeddingGemma 2 checkpoint on CPU.

Same encode path as ``model/gen_multimodal_fixtures.py``. This is the
parity target for the Core AI towers, not an ANE placement. It needs the
local checkpoint (``ANEMLL_EMBEDDINGS_MODEL``) and sentence-transformers.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

from demo.media_io import AUDIO_SR, resample_linear
from demo.types import DIM, EmbedResult, TowerHealth


def _unit(vector: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if arr.shape != (DIM,) or not np.isfinite(arr).all() or norm < 1e-8:
        raise RuntimeError(f"reference embedding shape={arr.shape} norm={norm}")
    return (arr / norm).astype(np.float32, copy=False)


class ReferenceBackend:
    name = "reference"
    dim = DIM

    def __init__(self, *, model_path: Path | None) -> None:
        self.model_path = model_path
        self.placement = "pytorch-cpu"
        self._model = None
        self._towers: list[TowerHealth] = []
        self._ready = False

    def warmup(self) -> dict:
        if self._ready:
            return self.health()
        if self.model_path is None or not Path(self.model_path).is_dir():
            raise FileNotFoundError(
                "reference backend needs ANEMLL_EMBEDDINGS_MODEL "
                "(full EmbeddingGemma 2 checkpoint, vision + audio enabled)"
            )
        import torch

        from model.load_multimodal_model import load_multimodal_sentence_transformer

        model, _meta = load_multimodal_sentence_transformer(
            self.model_path, dtype=torch.float32, device="cpu"
        )
        self._model = model
        timings = []
        for name, fn in (
            ("text_embeds_s320", lambda: self.embed_text("warmup", role="query")),
            ("vision_s280", lambda: self.embed_image(Image.new("RGB", (32, 32), (32, 160, 80)))),
            (
                "audio_s280",
                lambda: self.embed_audio(np.zeros(AUDIO_SR, dtype=np.float32), AUDIO_SR),
            ),
        ):
            t0 = time.perf_counter()
            fn()
            timings.append((name, (time.perf_counter() - t0) * 1000.0))
        self._towers = [
            TowerHealth(
                name=name,
                loaded=True,
                warmup_ms=ms,
                placement=self.placement,
                simulated=False,
            )
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
        if role == "query":
            prompt = "SearchQuery"
        elif role == "document":
            prompt = "Document"
        else:
            raise ValueError(f"unknown text role {role!r}")
        vec = self._model.encode(text, prompt_name=prompt)
        return EmbedResult(
            vector=_unit(vec),
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            backend=self.name,
            modality="text",
            placement=self.placement,
        )

    def embed_image(self, image: Image.Image) -> EmbedResult:
        t0 = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="anemll-ref-") as tmp:
            path = Path(tmp) / "image.png"
            image.save(path)
            vec = self._model.encode({"text": "<|image|>", "image": [str(path)]})
        return EmbedResult(
            vector=_unit(vec),
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            backend=self.name,
            modality="image",
            placement=self.placement,
        )

    def embed_audio(self, wav: np.ndarray, sample_rate: int) -> EmbedResult:
        t0 = time.perf_counter()
        samples = np.asarray(wav, dtype=np.float32).reshape(-1)
        if int(sample_rate) != AUDIO_SR:
            samples = resample_linear(samples, int(sample_rate), AUDIO_SR)
        vec = self._model.encode({"text": "<|audio|>", "audio": samples})
        return EmbedResult(
            vector=_unit(vec),
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            backend=self.name,
            modality="audio",
            placement=self.placement,
        )

    def close(self) -> None:
        self._model = None
