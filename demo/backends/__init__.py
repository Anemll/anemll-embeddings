"""Pluggable embedders: ``mock``, ``reference``, and ``coreai``.

The showcase talks to :class:`api.embedder.Embedder` — the same public
runtime a script would use. ``open_backend`` is the factory the server
calls at startup.
"""

from __future__ import annotations

from pathlib import Path

from api import Embedder


def open_backend(
    name: str,
    *,
    artifacts: Path | None = None,
    model: Path | None = None,
    coreai_python: Path | None = None,
    compute: str = "ane",
) -> Embedder:
    key = (name or "mock").strip().lower()
    if key not in {"mock", "reference", "coreai"}:
        raise ValueError(f"unknown backend {name!r} (expected mock, reference, or coreai)")
    return Embedder(
        artifacts=artifacts,
        model=model,
        coreai_python=coreai_python,
        compute=compute,
        backend=key,
    )
