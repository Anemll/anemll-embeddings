"""Showcase process settings.

Parity (``model/_coreai_run_npy.py``) defaults ``ANEMLL_COREAI_COMPUTE`` to
CPU. This server defaults to the Neural Engine. The worker process is the
only place that sets ``ANEMLL_COREAI_COMPUTE``, so a shell used for CPU
parity is not silently switched.
"""

from __future__ import annotations

import os
from pathlib import Path

# 8765 is AnemllAgentHost on the Mac. Override with --port or ANEMLL_DEMO_PORT.
DEFAULT_PORT = 8766


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


def default_alert_dir() -> Path:
    raw = os.environ.get("ANEMLL_DEMO_ALERT")
    if raw:
        return Path(raw)
    return Path.home() / ".anemll-embeddings" / "alert"


def assert_outside_artifacts(path: Path, artifacts: Path | None) -> Path:
    """Refuse demo writes that would land in the Core AI artifacts tree.

    The artifacts directory holds ``.aimodel`` packages. The index, corpus,
    and any log the server is asked to keep belong somewhere else.
    """
    target = Path(path).expanduser().resolve()
    if artifacts is None:
        return target
    root = Path(artifacts).expanduser().resolve()
    if target == root or root in target.parents:
        raise ValueError(
            f"{target} is inside the Core AI artifacts directory ({root}). "
            "Keep demo data and logs outside it, for example "
            "$HOME/.anemll-embeddings/demo."
        )
    return target
