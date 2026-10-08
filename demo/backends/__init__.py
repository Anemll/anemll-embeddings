"""Pluggable embedders: ``mock``, ``reference``, and ``coreai``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .mock import MockBackend


def open_backend(
    name: str,
    *,
    artifacts: Path | None = None,
    model: Path | None = None,
    coreai_python: Path | None = None,
    compute: str = "ane",
) -> Any:
    key = (name or "mock").strip().lower()
    if key == "mock":
        return MockBackend()
    if key == "reference":
        # Sentence-Transformers is optional and heavy; import only when selected.
        from .reference import ReferenceBackend

        return ReferenceBackend(model_path=model)
    if key == "coreai":
        from .coreai import CoreAIBackend

        return CoreAIBackend(
            artifacts=artifacts,
            model_path=model,
            coreai_python=coreai_python,
            compute=compute,
        )
    raise ValueError(f"unknown backend {name!r} (expected mock, reference, or coreai)")
