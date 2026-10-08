"""Showcase process settings.

Parity (``scripts/_coreai_run_npy.py``) defaults ``ANEMLL_COREAI_COMPUTE`` to
CPU. This server defaults to the Neural Engine. The worker process is the
only place that sets ``ANEMLL_COREAI_COMPUTE``, so a shell used for CPU
parity is not silently switched.
"""

from __future__ import annotations

import os
from pathlib import Path


def server_compute(explicit: str | None = None) -> str:
    """``ane`` unless ``--compute`` or ``ANEMLL_DEMO_COMPUTE`` says ``cpu``."""
    if explicit is not None and str(explicit).strip():
        raw = str(explicit)
    else:
        raw = os.environ.get("ANEMLL_DEMO_COMPUTE", "ane")
    key = raw.strip().lower()
    if key not in {"ane", "cpu"}:
        raise ValueError(f"compute must be ane or cpu, got {raw!r}")
    return key


def env_path(name: str) -> Path | None:
    raw = os.environ.get(name)
    if not raw:
        return None
    return Path(raw)


def default_data_dir() -> Path:
    raw = os.environ.get("ANEMLL_DEMO_DATA")
    if raw:
        return Path(raw)
    return Path.home() / ".anemll-embeddings" / "demo"


def default_corpus_dir() -> Path:
    raw = os.environ.get("ANEMLL_DEMO_CORPUS")
    if raw:
        return Path(raw)
    return Path.home() / ".anemll-embeddings" / "corpus"
