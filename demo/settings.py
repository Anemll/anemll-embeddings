"""Showcase process settings.

Parity (``model/_coreai_run_npy.py``) defaults ``ANEMLL_COREAI_COMPUTE`` to
CPU. This server defaults to the Neural Engine. The worker process is the
only place that sets ``ANEMLL_COREAI_COMPUTE``, so a shell used for CPU
parity is not silently switched.
"""

from __future__ import annotations

import ipaddress
import os
from pathlib import Path

# 8765 is AnemllAgentHost on the Mac. Override with --port or ANEMLL_DEMO_PORT.
DEFAULT_PORT = 8766
# Loopback unless asked. LAN testing: --host 0.0.0.0 or ANEMLL_DEMO_HOST=0.0.0.0.
DEFAULT_HOST = "127.0.0.1"
# Whole-request caps (checked on Content-Length and while streaming).
DEFAULT_MAX_UPLOAD_MB = 40.0  # multipart uploads (largest file cap is 30 MB audio)
DEFAULT_MAX_JSON_KB = 256.0  # JSON / other bodies (text queries, alert rules)


def server_host(explicit: str | None = None) -> str:
    """``--host``, else ``ANEMLL_DEMO_HOST``, else 127.0.0.1."""
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()
    return (os.environ.get("ANEMLL_DEMO_HOST") or DEFAULT_HOST).strip()


def is_loopback(host: str) -> bool:
    name = str(host or "").strip().strip("[]").lower()
    if name in {"localhost", "localhost.localdomain"}:
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


def bind_warning(host: str, port: int) -> str | None:
    """Warning text for a non-loopback bind, else ``None``."""
    if is_loopback(host):
        return None
    return (
        f"WARNING: the demo is listening on {host}:{port}, reachable from other machines. "
        "It has no authentication: anyone who can reach this port can add, delete, and "
        "read indexed items and uploads. Use it only on a network you trust, or bind "
        "--host 127.0.0.1. (The microphone page needs http://127.0.0.1 or HTTPS.)"
    )


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {raw!r}")
    return value


def max_upload_bytes() -> int:
    """Multipart request cap: ``ANEMLL_DEMO_MAX_UPLOAD_MB`` (default 40)."""
    return int(_env_float("ANEMLL_DEMO_MAX_UPLOAD_MB", DEFAULT_MAX_UPLOAD_MB) * 1024 * 1024)


def max_json_bytes() -> int:
    """Non-multipart request cap: ``ANEMLL_DEMO_MAX_JSON_KB`` (default 256)."""
    return int(_env_float("ANEMLL_DEMO_MAX_JSON_KB", DEFAULT_MAX_JSON_KB) * 1024)


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
