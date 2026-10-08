"""Offline checks for scripts/release_manifest.py (no Hub access)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import release_manifest as rm  # noqa: E402
from scripts.download_common import HOST_SHA256, TOWER_MLIRB_SHA256  # noqa: E402


def test_repo_files_cover_download_path() -> None:
    files = rm.repo_files()
    for rel in (
        "scripts/download_common.py",
        "scripts/download_models.py",
        "scripts/warmup.py",
        "scripts/release_manifest.py",
        "api/embedder.py",
        "pyproject.toml",
        "constraints.txt",
    ):
        assert rel in files, rel


def test_committed_manifest_matches_pin() -> None:
    path = rm.manifest_path("v0.1.0")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema"] == rm.SCHEMA
    assert data["version"] == "v0.1.0"
    # v0.1.0 shipped at HF 90d2ab4; later pins (ANE_REVISION) only add files
    # (root config.json), so its tower/host digests still match the tables.
    assert data["huggingface"]["revision"] == "90d2ab497d423bba4ee29947b274c787bb4a1f0a"
    hub = data["huggingface"]["files"]
    for tower in ("vision_s280", "audio_s280", "text_embeds_s320"):
        assert f"{tower}/{tower}.aimodel/main.mlirb" in hub
    assert "host/embed_tokens.safetensors" in hub
    for tower, digest in TOWER_MLIRB_SHA256.items():
        assert hub[f"{tower}/{tower}.aimodel/main.mlirb"]["sha256"] == digest
    for name, digest in HOST_SHA256.items():
        assert hub[f"host/{name}"]["sha256"] == digest


def test_verify_and_sums(tmp_path: Path, capsys) -> None:
    manifest = {
        "schema": rm.SCHEMA,
        "version": "vtest",
        "huggingface": {"repo": "x/y", "revision": "r", "files": {"a/b.bin": {"sha256": "0" * 64, "size": 1}}},
        "repo": {"files": rm.hash_repo()},
    }
    path = tmp_path / "m.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    # No downloaded package under tmp_path/ane: only repo files are checked.
    assert rm.main(["verify", "--version", "vtest", "--manifest", str(path), "--dest", str(tmp_path)]) == 0
    sums = rm.sha256sums(manifest)
    assert sums.startswith("0" * 64 + "  hf/a/b.bin\n")
    assert "  repo/scripts/release_manifest.py\n" in sums

    # A tampered downloaded file is reported.
    (tmp_path / "ane" / "a").mkdir(parents=True)
    (tmp_path / "ane" / "a" / "b.bin").write_bytes(b"x")
    assert rm.main(["verify", "--version", "vtest", "--manifest", str(path), "--dest", str(tmp_path)]) == 1
    assert "MISMATCH hf/a/b.bin" in capsys.readouterr().out
