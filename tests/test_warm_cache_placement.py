"""Warm-cache placement lookup: a binary main.hash with a whitespace edge byte."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from api import coreai_worker  # noqa: E402
from api.coreai_worker import (  # noqa: E402
    manifest_label_for_package,
    package_main_hash_hex,
)

# vision_s280's real main.hash ends in 0x0c (form feed). bytes.strip() eats it.
VISION_HASH = bytes.fromhex(
    "5ebb5342a5c78ab2790c82358e3ad90092ea3bbdde6a9dc85a47d4ecc8d37c0c"
)


def _write_pkg(root: Path, name: str, main_hash: bytes) -> Path:
    pkg = root / f"{name}.aimodel"
    pkg.mkdir(parents=True)
    (pkg / "main.hash").write_bytes(main_hash)
    return pkg


def test_binary_hash_keeps_whitespace_edge_bytes(tmp_path: Path) -> None:
    for index, raw in enumerate((
        VISION_HASH,  # ends with 0x0c
        b"\x09" + VISION_HASH[1:],  # starts with a tab
        VISION_HASH[:-1] + b"\x20",  # ends with a space
        VISION_HASH[:-1] + b"\x0a",  # ends with a newline byte
    )):
        pkg = _write_pkg(tmp_path / str(index), "vision_s280", raw)
        assert package_main_hash_hex(pkg) == raw.hex()
        assert len(package_main_hash_hex(pkg) or "") == 64


def test_hex_text_hash_still_accepts_trailing_newline(tmp_path: Path) -> None:
    pkg = _write_pkg(tmp_path, "t", b"AB" * 32 + b"\n")
    assert package_main_hash_hex(pkg) == ("ab" * 32)


def test_warm_manifest_found_for_whitespace_edged_hash(tmp_path: Path) -> None:
    pkg = _write_pkg(tmp_path, "vision_s280", VISION_HASH)
    cache = tmp_path / "coreai-cache"
    manifest = (
        cache / "26A5425a" / "org.python.python" / VISION_HASH.hex() / "ABCDEF" / "model.aimodelx"
        / "main-this-delegates" / "manifest.plist"
    )
    manifest.parent.mkdir(parents=True)
    manifest.write_bytes(b"mps.fullyPlacedOnANE")
    other = cache / "26A5425a" / "org.python.python" / ("cd" * 32) / "X" / "model.aimodelx" / "m" / "manifest.plist"
    other.parent.mkdir(parents=True)
    other.write_bytes(b"gpu only")
    os.utime(manifest, (time.time() - 9999,) * 2)  # warm start: old file, other newer
    assert manifest_label_for_package(pkg, cache) == "fullyOnANE"
    # The stripped (31 byte) name must not be what the lookup relies on.
    assert VISION_HASH.strip().hex() != package_main_hash_hex(pkg)
    assert coreai_worker.package_main_hash_hex is package_main_hash_hex
