"""Bounded, checksummed HTTP downloads for the sample fetch scripts.

Every file is streamed with a byte cap (so a wrong or hostile URL cannot
fill the disk), written atomically, and described by a provenance record:
requested URL, final URL after redirects, byte count, SHA-256, and UTC
fetch time. The fetchers store that record in their ``manifest.json``.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MAX_IMAGE_BYTES = 25 * 1024 * 1024
MAX_AUDIO_BYTES = 60 * 1024 * 1024
MAX_JSON_BYTES = 8 * 1024 * 1024
_CHUNK = 1024 * 1024


class DownloadTooLarge(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    """SHA-256 and size of a file as stored (after any conversion)."""
    path = Path(path)
    return {"sha256": sha256_file(path), "bytes": path.stat().st_size}


def _read_capped(response, max_bytes: int, sink) -> tuple[int, str]:
    declared = response.headers.get("Content-Length")
    if declared is not None and declared.isdigit() and int(declared) > max_bytes:
        raise DownloadTooLarge(f"{declared} bytes declared, limit {max_bytes}")
    digest = hashlib.sha256()
    total = 0
    while True:
        chunk = response.read(_CHUNK)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise DownloadTooLarge(f"more than {max_bytes} bytes")
        digest.update(chunk)
        sink(chunk)
    return total, digest.hexdigest()


def download(url: str, dest: Path, *, user_agent: str, max_bytes: int, timeout: float = 90) -> dict[str, Any]:
    """Stream ``url`` to ``dest`` (atomic). Return the provenance record."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    fd, tmp_name = tempfile.mkstemp(prefix=dest.name + ".", suffix=".part", dir=dest.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle, urllib.request.urlopen(req, timeout=timeout) as response:
            final_url = response.geturl()
            total, digest = _read_capped(response, max_bytes, handle.write)
        os.replace(tmp, dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return {
        "url": url,
        "final_url": final_url,
        "bytes": total,
        "sha256": digest,
        "fetched_at": utc_now(),
    }


def get_json(url: str, *, user_agent: str, timeout: float = 60, max_bytes: int = MAX_JSON_BYTES) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": user_agent, "Accept": "application/json"})
    chunks: list[bytes] = []
    with urllib.request.urlopen(req, timeout=timeout) as response:
        _read_capped(response, max_bytes, chunks.append)
    return json.loads(b"".join(chunks).decode("utf-8"))
