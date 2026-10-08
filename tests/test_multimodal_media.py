#!/usr/bin/env python3
"""Unit checks for multimodal loader kwargs + synthetic media (no checkpoint)."""

from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from model.load_multimodal_model import modality_config_kwargs  # noqa: E402
from model.multimodal_media import (  # noqa: E402
    AUDIO_SR,
    write_default_media,
    write_tone_wav,
)


def _fail(msg: str) -> None:
    raise AssertionError(msg)


def test_modality_kwargs() -> None:
    if modality_config_kwargs(vision=True, audio=True) != {}:
        _fail(str(modality_config_kwargs(vision=True, audio=True)))
    if modality_config_kwargs(vision=False, audio=False) != {
        "vision_config": None,
        "audio_config": None,
    }:
        _fail("text-only kwargs")
    if modality_config_kwargs(vision=True, audio=False) != {"audio_config": None}:
        _fail("vision-only kwargs")


def test_prompts_schema() -> None:
    pack = json.loads((REPO_ROOT / "tests" / "fixtures" / "multimodal_prompts.json").read_text())
    ids = [p["id"] for p in pack["prompts"]]
    if len(ids) != len(set(ids)):
        _fail(f"duplicate ids {ids}")
    kinds = {p["kind"] for p in pack["prompts"]}
    if not {"text", "image", "audio", "video", "interleaved"} <= kinds:
        _fail(f"kinds {kinds}")


def test_write_media(tmp_path: Path | None = None) -> None:
    root = tmp_path or (REPO_ROOT / ".tmp-mm-media")
    media = root / "media" if tmp_path is None else root
    if tmp_path is None:
        if root.exists():
            for p in root.rglob("*"):
                if p.is_file():
                    p.unlink()
        media.mkdir(parents=True, exist_ok=True)
    wrote = write_default_media(media)
    img = Image.open(wrote["aurora.png"])
    if img.size != (64, 64) or img.mode != "RGB":
        _fail(f"png {img.size} {img.mode}")
    with wave.open(str(wrote["tone_a4.wav"]), "r") as w:
        if w.getnchannels() != 1 or w.getframerate() != AUDIO_SR:
            _fail(f"wav ch={w.getnchannels()} sr={w.getframerate()}")
        if w.getnframes() != AUDIO_SR:
            _fail(f"wav frames {w.getnframes()}")
    frames = wrote["video_frames"]
    if len(frames) != 3:
        _fail(str(frames))
    write_tone_wav(media / "extra.wav", seconds=0.25)
    if tmp_path is None and root.exists():
        for p in sorted(root.rglob("*"), reverse=True):
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                p.rmdir()


def main() -> int:
    tests = [test_modality_kwargs, test_prompts_schema, test_write_media]
    for fn in tests:
        fn()
        print(f"OK {fn.__name__}")
    print(f"OK: {len(tests)} multimodal-media checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
