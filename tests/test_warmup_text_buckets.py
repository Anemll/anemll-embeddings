"""warmup.py gating over the text bucket / pack functions (no Core AI)."""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.warmup import towers_not_on_ane, use_text_buckets  # noqa: E402


def _row(place: str, extra: bool = False, loaded: bool = True) -> dict:
    row = {"placement": place, "loaded": loaded}
    if extra:
        row["extra"] = True
    return row


def test_require_ane_covers_extra_text_functions() -> None:
    good = {
        "towers": {
            "vision_s280": _row("fullyOnANE"),
            "text_embeds_s320": _row("fullyOnANE"),
            "text_embeds_s32": _row("fullyOnANE", extra=True),
            "text_pack_256x16": _row("fullyOnANE", extra=True),
        }
    }
    assert towers_not_on_ane(good, expect_extras=True) == []
    gpu = {"towers": dict(good["towers"], text_pack_256x16=_row("GPU", extra=True))}
    assert towers_not_on_ane(gpu, expect_extras=True) == ["text_pack_256x16"]
    unknown = {"towers": dict(good["towers"], text_embeds_s64=_row("unknown", extra=True))}
    assert towers_not_on_ane(unknown, expect_extras=True) == ["text_embeds_s64"]
    failed = {"towers": dict(good["towers"], **{"text_buckets.aimodel": {"loaded": False, "extra": True}})}
    assert towers_not_on_ane(failed, expect_extras=True) == ["text_buckets.aimodel"]


def test_require_ane_fails_when_extras_expected_but_none_loaded() -> None:
    only_main = {"towers": {"text_embeds_s320": _row("fullyOnANE")}}
    assert towers_not_on_ane(only_main) == []
    assert towers_not_on_ane(only_main, expect_extras=True) == ["<text_buckets: no function loaded>"]


def test_use_text_buckets_switches(monkeypatch) -> None:
    monkeypatch.delenv("ANEMLL_TEXT_BUCKETS", raising=False)
    assert use_text_buckets(Namespace(no_text_buckets=False))
    assert not use_text_buckets(Namespace(no_text_buckets=True))
    monkeypatch.setenv("ANEMLL_TEXT_BUCKETS", "0")
    assert not use_text_buckets(Namespace(no_text_buckets=False))
