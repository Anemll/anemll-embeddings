#!/usr/bin/env python3
"""Unit checks for scripts/download_models.py (no network, no Core AI)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.download_models import (  # noqa: E402
    ANE_REVISION,
    TOWERS,
    all_bundles_complete,
    bundle_complete,
    env_exports,
    ensure_symlink,
    link_coreai,
    write_revision,
    revision_matches,
)


def _fail(msg: str) -> None:
    raise AssertionError(msg)


def _write_bundle(ane: Path, name: str) -> None:
    root = ane / name / f"{name}.aimodel"
    root.mkdir(parents=True, exist_ok=True)
    for filename in ("metadata.json", "main.hash", "main.mlirb"):
        (root / filename).write_text("ok\n", encoding="utf-8")


def test_bundle_complete(tmp: Path) -> None:
    ane = tmp / "ane"
    if all_bundles_complete(ane):
        _fail("empty dir should be incomplete")
    _write_bundle(ane, "vision_s280")
    if not bundle_complete(ane, "vision_s280"):
        _fail("vision bundle should be complete")
    if all_bundles_complete(ane):
        _fail("two towers still missing")
    for name in TOWERS:
        _write_bundle(ane, name)
    if not all_bundles_complete(ane):
        _fail("all three bundles should be complete")


def test_symlink_idempotent(tmp: Path) -> None:
    ane = tmp / "ane"
    artifacts = tmp / "artifacts"
    for name in TOWERS:
        _write_bundle(ane, name)
    first = link_coreai(artifacts, ane)
    if set(first) != set(TOWERS) or any(state != "created" for state in first.values()):
        _fail(f"first link {first}")
    again = link_coreai(artifacts, ane)
    if any(state != "ok" for state in again.values()):
        _fail(f"second link {again}")
    other = tmp / "other"
    _write_bundle(other, "vision_s280")
    link = artifacts / "coreai" / "vision_s280.aimodel"
    if ensure_symlink(link, other / "vision_s280" / "vision_s280.aimodel") != "replaced":
        _fail("expected replaced")
    if link.resolve() != (other / "vision_s280" / "vision_s280.aimodel").resolve():
        _fail("symlink did not retarget")


def test_revision_marker(tmp: Path) -> None:
    folder = tmp / "ane"
    if revision_matches(folder, ANE_REVISION):
        _fail("missing marker should not match")
    write_revision(folder, ANE_REVISION)
    if not revision_matches(folder, ANE_REVISION):
        _fail("marker should match")
    if revision_matches(folder, "deadbeef"):
        _fail("wrong revision should not match")


def test_env_exports() -> None:
    text = env_exports(
        artifacts=Path("/tmp/artifacts"),
        model=Path("/tmp/model"),
        coreai_python="/tmp/coreai/bin/python",
    )
    want = (
        "export ANEMLL_EMBEDDINGS_ARTIFACTS=/tmp/artifacts\n"
        "export ANEMLL_EMBEDDINGS_MODEL=/tmp/model\n"
        "export ANEMLL_COREAI_PYTHON=/tmp/coreai/bin/python"
    )
    if text != want:
        _fail(f"exports {text!r}")


def test_existing_dir_is_not_replaced(tmp: Path) -> None:
    target = tmp / "real.aimodel"
    target.mkdir(parents=True)
    (target / "main.hash").write_text("x\n", encoding="utf-8")
    link = tmp / "coreai" / "vision_s280.aimodel"
    link.parent.mkdir()
    link.mkdir()
    try:
        ensure_symlink(link, target)
    except FileExistsError:
        return
    _fail("expected FileExistsError for a real directory")


def main() -> int:
    import tempfile

    with tempfile.TemporaryDirectory(prefix="anemll-dl-") as raw:
        tmp = Path(raw)
        test_bundle_complete(tmp / "bundles")
        test_symlink_idempotent(tmp / "links")
        test_revision_marker(tmp / "rev")
        test_env_exports()
        test_existing_dir_is_not_replaced(tmp / "clash")
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
