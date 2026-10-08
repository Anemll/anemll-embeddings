#!/usr/bin/env python3
"""Public ``from api import Embedder`` surface (mock backend, no Core AI)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from api import DIM, Embedder, cosine, similarity  # noqa: E402

_ENV_KEYS = (
    "ANEMLL_EMBEDDINGS_ARTIFACTS",
    "ANEMLL_EMBEDDINGS_MODEL",
    "ANEMLL_COREAI_PYTHON",
)


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


def _restore_env(saved: dict[str, str | None]) -> None:
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


class _StubBackend:
    def __init__(
        self,
        *,
        artifacts: Path | None,
        model_path: Path | None,
        coreai_python: Path | None,
        compute: str = "ane",
        **kwargs: object,
    ) -> None:
        self.artifacts = artifacts
        self.model_path = model_path
        self.coreai_python = coreai_python
        self.compute = compute
        self.kwargs = kwargs

    def close(self) -> None:
        return None


def test_embedder_reads_env_when_args_are_none() -> None:
    saved = {key: os.environ.get(key) for key in _ENV_KEYS}
    artifacts = Path("/tmp/anemll-test-artifacts")
    model = Path("/tmp/anemll-test-model")
    coreai = Path("/tmp/anemll-test-coreai-python")
    try:
        os.environ["ANEMLL_EMBEDDINGS_ARTIFACTS"] = str(artifacts)
        os.environ["ANEMLL_EMBEDDINGS_MODEL"] = str(model)
        os.environ["ANEMLL_COREAI_PYTHON"] = str(coreai)
        with patch("api.embedder.CoreAIBackend", _StubBackend):
            embedder = Embedder(compute="ane")
        impl = embedder._impl
        if not isinstance(impl, _StubBackend):
            _fail("expected stubbed CoreAIBackend")
        if impl.artifacts != artifacts:
            _fail(f"artifacts {impl.artifacts!r} != {artifacts}")
        if impl.model_path != model:
            _fail(f"model {impl.model_path!r} != {model}")
        if impl.coreai_python != coreai:
            _fail(f"coreai_python {impl.coreai_python!r} != {coreai}")
        if impl.compute != "ane":
            _fail(f"compute {impl.compute!r}")
    finally:
        _restore_env(saved)


def test_embedder_unset_env_passes_none() -> None:
    saved = {key: os.environ.get(key) for key in _ENV_KEYS}
    try:
        for key in _ENV_KEYS:
            os.environ.pop(key, None)
        with patch("api.embedder.CoreAIBackend", _StubBackend):
            embedder = Embedder(compute="ane")
        impl = embedder._impl
        if not isinstance(impl, _StubBackend):
            _fail("expected stubbed CoreAIBackend")
        if impl.artifacts is not None:
            _fail(f"artifacts should be None, got {impl.artifacts!r}")
        if impl.model_path is not None:
            _fail(f"model should be None, got {impl.model_path!r}")
        if impl.coreai_python is not None:
            _fail(f"coreai_python should be None, got {impl.coreai_python!r}")
    finally:
        _restore_env(saved)


def test_embedder_explicit_paths_win_over_env() -> None:
    saved = {key: os.environ.get(key) for key in _ENV_KEYS}
    env_art = Path("/tmp/anemll-env-artifacts")
    explicit_art = Path("/tmp/anemll-explicit-artifacts")
    explicit_model = Path("/tmp/anemll-explicit-model")
    explicit_py = Path("/tmp/anemll-explicit-coreai-python")
    try:
        os.environ["ANEMLL_EMBEDDINGS_ARTIFACTS"] = str(env_art)
        os.environ["ANEMLL_EMBEDDINGS_MODEL"] = "/tmp/anemll-env-model"
        os.environ["ANEMLL_COREAI_PYTHON"] = "/tmp/anemll-env-coreai-python"
        with patch("api.embedder.CoreAIBackend", _StubBackend):
            embedder = Embedder(
                artifacts=explicit_art,
                model=explicit_model,
                coreai_python=explicit_py,
                compute="cpu",
            )
        impl = embedder._impl
        if not isinstance(impl, _StubBackend):
            _fail("expected stubbed CoreAIBackend")
        if impl.artifacts != explicit_art:
            _fail(f"explicit artifacts lost: {impl.artifacts!r}")
        if impl.model_path != explicit_model:
            _fail(f"explicit model lost: {impl.model_path!r}")
        if impl.coreai_python != explicit_py:
            _fail(f"explicit coreai_python lost: {impl.coreai_python!r}")
        if impl.compute != "cpu":
            _fail(f"compute {impl.compute!r}")
    finally:
        _restore_env(saved)


def main() -> int:
    test_embed_text_image_audio()
    test_cosine_rejects_bad_shape()
    test_unknown_backend()
    test_embedder_reads_env_when_args_are_none()
    test_embedder_unset_env_passes_none()
    test_embedder_explicit_paths_win_over_env()
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
