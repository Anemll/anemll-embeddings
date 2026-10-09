"""Staged text_buckets download path and descriptors (no network, no Core AI)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts import download_common as common  # noqa: E402
from scripts import download_models as dm  # noqa: E402


def _write_text_buckets(ane: Path, content: bytes = b"x") -> None:
    root = common.bundle_path(ane, common.TEXT_BUCKETS)
    root.mkdir(parents=True, exist_ok=True)
    for name in common.BUNDLE_FILES:
        (root / name).write_bytes(content)


def test_text_buckets_is_unpublished_placeholder_now() -> None:
    assert common.TEXT_BUCKETS_REVISION is None
    assert not common.text_buckets_published()
    assert all(v == common.TEXT_BUCKETS_PLACEHOLDER for v in common.TEXT_BUCKETS_SHA256.values())
    assert common.text_buckets_download_bytes() == 0
    assert common.text_buckets_checksums(Path("/nowhere")) == {}


def test_download_skips_unpublished_text_buckets(tmp_path: Path) -> None:
    calls: list[dict] = []
    with patch("scripts.download_common._hf_snapshot_download", lambda **kw: calls.append(kw)):
        assert dm.download_text_buckets(tmp_path, force=True) == "not-published"
        assert dm.download_text_buckets(tmp_path, force=True, enabled=False) == "disabled"
    assert calls == []
    assert dm.verify_text_buckets(tmp_path) == "not-published"


def test_published_text_buckets_download_and_verify(tmp_path: Path) -> None:
    payload = {"main.mlirb": b"mlirb-bytes", "main.hash": b"\x01" * 32, "metadata.json": b"{}"}
    digests = {k: hashlib.sha256(payload[k]).hexdigest() for k in ("main.mlirb", "main.hash")}
    calls: list[dict] = []

    def fake(**kwargs):
        calls.append(kwargs)
        root = common.bundle_path(Path(kwargs["local_dir"]), common.TEXT_BUCKETS)
        root.mkdir(parents=True, exist_ok=True)
        for name, data in payload.items():
            (root / name).write_bytes(data)

    with (
        patch.object(common, "TEXT_BUCKETS_REVISION", "a" * 40),
        patch.object(common, "TEXT_BUCKETS_SHA256", digests),
        patch("scripts.download_common._hf_snapshot_download", fake),
    ):
        assert common.text_buckets_published()
        assert dm.download_text_buckets(tmp_path, force=False) == "downloaded"
        assert calls[0]["allow_patterns"] == list(common.TEXT_BUCKETS_ALLOW)
        assert calls[0]["revision"] == "a" * 40
        assert dm.download_text_buckets(tmp_path, force=False) == "skipped"
        assert dm.verify_text_buckets(tmp_path) == "ok"
        (common.bundle_path(tmp_path, common.TEXT_BUCKETS) / "main.mlirb").write_bytes(b"tampered")
        try:
            dm.verify_text_buckets(tmp_path)
        except SystemExit as exc:
            assert "checksum" in str(exc)
        else:
            raise AssertionError("tampered main.mlirb must fail verification")


def test_link_coreai_adds_text_buckets_only_when_present(tmp_path: Path) -> None:
    ane = tmp_path / "ane"
    for name in common.TOWERS:
        root = common.bundle_path(ane, name)
        root.mkdir(parents=True)
        for f in common.BUNDLE_FILES:
            (root / f).write_text("ok\n")
    assert set(common.link_coreai(tmp_path / "artifacts", ane)) == set(common.TOWERS)
    _write_text_buckets(ane)
    status = common.link_coreai(tmp_path / "artifacts", ane)
    assert status[common.TEXT_BUCKETS] in {"created", "ok"}
    assert (tmp_path / "artifacts" / "coreai" / "text_buckets.aimodel").is_symlink()


def test_inference_allow_does_not_pull_text_buckets() -> None:
    assert not common.matches_hf_patterns(
        "text_buckets/text_buckets.aimodel/main.mlirb", common.ANE_INFERENCE_ALLOW, None
    )
    assert common.matches_hf_patterns(
        "text_buckets/text_buckets.aimodel/main.mlirb", common.TEXT_BUCKETS_ALLOW, None
    )


def test_staged_hf_config_describes_text_buckets() -> None:
    raw = (REPO_ROOT / "hf" / "config.json").read_bytes()
    staged = json.loads(raw)
    pkg = staged["optional_packages"]["text_buckets"]
    assert pkg["optional"] is True
    assert pkg["revision"] == common.TEXT_BUCKETS_PLACEHOLDER
    assert set(pkg["sha256"]) == set(common.TEXT_BUCKETS_SHA256)
    assert tuple(pkg["functions"]) == common.TEXT_BUCKETS_FUNCTIONS
    # the published towers table is unchanged and text_buckets is not listed in it
    assert set(staged["towers"]) == set(common.TOWERS)
    # staged digest/size pair matches the file until the pin is bumped
    assert hashlib.sha256(raw).hexdigest() == common.ROOT_CONFIG_STAGED_SHA256
    assert len(raw) == common.ROOT_CONFIG_STAGED_BYTES
    if common.text_buckets_published():
        assert common.ROOT_CONFIG_SHA256 == common.ROOT_CONFIG_STAGED_SHA256


def test_staged_towers_yaml_marks_placeholders() -> None:
    text = (REPO_ROOT / "hf" / "towers.yaml").read_text(encoding="utf-8")
    assert "text_buckets:" in text
    assert common.TEXT_BUCKETS_PLACEHOLDER in text
    for name in common.TEXT_BUCKETS_FUNCTIONS:
        assert name in text
