#!/usr/bin/env python3
"""Public ``from api import Embedder`` surface (mock backend, no Core AI)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from api import DIM, Embedder, cosine, similarity  # noqa: E402


def _fail(msg: str) -> None:
    raise AssertionError(msg)


def test_embed_text_image_audio() -> None:
    embedder = Embedder(backend="mock")
    text = embedder.embed_text("a red fox", role="query")
    image = embedder.embed_image(Image.new("RGB", (16, 16), (12, 34, 56)))
    audio = embedder.embed_audio(np.zeros(1600, dtype=np.float32), 16000)
    embedder.close()
    for name, vec in (("text", text), ("image", image), ("audio", audio)):
        if vec.shape != (DIM,):
            _fail(f"{name} shape {vec.shape}")
        if abs(float(np.linalg.norm(vec)) - 1.0) > 1e-5:
            _fail(f"{name} not unit")
    again = Embedder(backend="mock").embed_text("a red fox", role="query")
    if not np.allclose(text, again, atol=1e-6):
        _fail("mock text embed is not deterministic")
    same = cosine(text, again)
    if abs(same - 1.0) > 1e-5:
        _fail(f"cosine(self) {same}")
    if cosine(text, image) == same and cosine(text, audio) == same:
        _fail("mock modalities should not all match")
    if similarity(text, text) != cosine(text, text):
        _fail("similarity should alias cosine")
    if embedder.last is None or embedder.last.vector.shape != (DIM,):
        _fail("Embedder.last should record the last embed")


def test_cosine_rejects_bad_shape() -> None:
    try:
        cosine(np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32))
    except ValueError:
        return
    _fail("expected ValueError for a non-768 vector")


def test_unknown_backend() -> None:
    try:
        Embedder(backend="nope")
    except ValueError:
        return
    _fail("expected ValueError for unknown backend")


def main() -> int:
    test_embed_text_image_audio()
    test_cosine_rejects_bad_shape()
    test_unknown_backend()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
