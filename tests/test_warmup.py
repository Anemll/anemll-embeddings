#!/usr/bin/env python3
"""Unit checks for warmup cache/python discovery (no Core AI)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.download_common import (  # noqa: E402
    cache_unwritable_hint,
    coreai_cache_dir,
    discover_coreai_python,
    fixed_user_home_for_cache,
)


def _fail(msg: str) -> None:
    raise AssertionError(msg)


def test_cache_dir_home(monkey_home: Path) -> None:
    saved_home = os.environ.get("CFFIXED_USER_HOME")
    try:
        os.environ.pop("CFFIXED_USER_HOME", None)
        explicit = monkey_home / "custom" / "coreai-cache"
        if coreai_cache_dir(explicit) != explicit:
            _fail("explicit --cache-dir should win")
        os.environ["CFFIXED_USER_HOME"] = str(monkey_home / "fixed")
        want = monkey_home / "fixed" / "Library" / "Caches" / "coreai-cache"
        if coreai_cache_dir() != want:
            _fail(f"CFFIXED_USER_HOME cache {coreai_cache_dir()} != {want}")
    finally:
        if saved_home is None:
            os.environ.pop("CFFIXED_USER_HOME", None)
        else:
            os.environ["CFFIXED_USER_HOME"] = saved_home


def test_fixed_user_home_for_cache(tmp: Path) -> None:
    cache = tmp / "home" / "Library" / "Caches" / "coreai-cache"
    home = fixed_user_home_for_cache(cache)
    if home != tmp / "home":
        _fail(f"home {home}")
    if fixed_user_home_for_cache(tmp / "other") is not None:
        _fail("nonstandard cache should not invent a home")


def test_discover_coreai_python(tmp: Path) -> None:
    missing = tmp / "nope" / "python"
    try:
        discover_coreai_python(missing, required=True)
    except SystemExit as exc:
        if "not found" not in str(exc):
            _fail(f"message {exc}")
    else:
        _fail("expected SystemExit for missing explicit python")
    if discover_coreai_python(missing, required=False) is not None:
        _fail("optional miss should be None")
    found = tmp / "venv" / "bin" / "python"
    found.parent.mkdir(parents=True)
    found.write_text("#!/bin/sh\n", encoding="utf-8")
    found.chmod(0o755)
    got = discover_coreai_python(found, required=True)
    if got != found:
        _fail(f"explicit {got}")


def test_cache_hint_mentions_cffixed(tmp: Path) -> None:
    cache = tmp / "Library" / "Caches" / "coreai-cache"
    text = cache_unwritable_hint(cache)
    if "CFFIXED_USER_HOME" not in text:
        _fail("hint should mention CFFIXED_USER_HOME")
    if str(cache) not in text:
        _fail("hint should mention the cache path")
    if "broken symlink" not in text and "unwritable" not in text and "worker died" not in text:
        _fail(f"hint {text}")


def main() -> int:
    import tempfile

    with tempfile.TemporaryDirectory(prefix="anemll-warmup-") as raw:
        tmp = Path(raw)
        test_cache_dir_home(tmp)
        test_fixed_user_home_for_cache(tmp / "layout")
        test_discover_coreai_python(tmp / "py")
        test_cache_hint_mentions_cffixed(tmp / "hint")
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
