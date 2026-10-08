#!/usr/bin/env python3
"""Download a small CC0 / CC BY demo corpus outside git.

Images and short sounds come from Openverse (Wikimedia, Flickr, Freesound,
and similar). Each file's license, creator, and source page are written to
``manifest.json``. Audio is stored as 16 kHz mono WAV, trimmed to at most
8 seconds. ``ffmpeg`` does that in one step. On a Mac without ffmpeg,
``afconvert`` writes the WAV and the ``wave`` module trims it. If neither
tool exists, audio topics are skipped with a warning. Nothing is committed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from demo.settings import assert_outside_artifacts, default_corpus_dir, env_path  # noqa: E402

UA = "anemll-embeddings-demo/1.0 (local educational corpus; contact local)"
IMAGE_TOPICS = [
    "tabby cat",
    "dog portrait",
    "songbird",
    "red fox",
    "horse",
    "penguin",
    "butterfly",
    "sunflower",
    "mountain lake",
    "ocean wave",
    "forest path",
    "desert dune",
    "stone bridge",
    "bicycle",
    "coffee cup",
    "red apple",
    "violin",
    "acoustic guitar",
    "lighthouse",
    "steam train",
    "sailboat",
    "city street night",
    "waterfall",
    "snow mountain",
    "tulip field",
    "bread loaf",
    "old camera",
    "hot air balloon",
]
AUDIO_TOPICS = [
    "cat meow",
    "dog bark",
    "bird song",
    "rain",
    "ocean waves",
    "piano",
    "bell",
    "footsteps",
    "thunder",
    "train whistle",
]
ALLOWED = {"cc0", "by", "pdm", "publicdomain"}


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=45) as response:
        return json.loads(response.read().decode("utf-8"))


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as response:
        dest.write_bytes(response.read())


def _license_ok(row: dict) -> bool:
    name = str(row.get("license") or "").lower().replace("_", "").replace("-", "")
    if name not in ALLOWED and not name.startswith("by"):
        return False
    if "nc" in name or "nd" in name or "sa" in name:
        return False
    return name in ALLOWED or name.startswith("by")


def _search(kind: str, query: str) -> list[dict]:
    params = {
        "q": query,
        "license": "cc0,by,pdm",
        "page_size": "8",
    }
    if kind == "images":
        params["category"] = "photograph"
    url = "https://api.openverse.org/v1/" + kind + "/?" + urllib.parse.urlencode(params)
    data = _get_json(url)
    return list(data.get("results") or [])


def _pick_image(rows: list[dict]) -> dict | None:
    for row in rows:
        if not _license_ok(row):
            continue
        url = row.get("url") or ""
        if not url.lower().split("?")[0].endswith((".jpg", ".jpeg", ".png")):
            continue
        return row
    return None


def _pick_audio(rows: list[dict]) -> dict | None:
    ranked = []
    for row in rows:
        if not _license_ok(row) or not row.get("url"):
            continue
        duration = row.get("duration")
        try:
            millis = float(duration) if duration is not None else 10_000
        except (TypeError, ValueError):
            millis = 10_000
        # Openverse durations are milliseconds.
        if millis > 60_000:
            continue
        ranked.append((millis, row))
    if not ranked:
        return None
    ranked.sort(key=lambda item: item[0])
    return ranked[0][1]


MAX_AUDIO_SECONDS = 8


def _audio_tool() -> str | None:
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    if shutil.which("afconvert"):
        return "afconvert"
    return None


def audio_skip_reason() -> str | None:
    """Warning text when this machine cannot decode corpus audio."""
    if _audio_tool() is not None:
        return None
    return "skipping audio: neither ffmpeg nor afconvert is available"


def _trim_with_wave(src: Path, dest: Path, *, seconds: float = MAX_AUDIO_SECONDS) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(src), "rb") as reader:
        rate = int(reader.getframerate())
        channels = int(reader.getnchannels())
        width = int(reader.getsampwidth())
        keep = min(int(reader.getnframes()), int(rate * seconds))
        frames = reader.readframes(keep)
    with wave.open(str(dest), "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(width)
        writer.setframerate(rate)
        writer.writeframes(frames)


def _run_checked(cmd: list[str], label: str) -> None:
    proc = subprocess.run(cmd, check=False, capture_output=True)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(err or f"{label} rc={proc.returncode}")


def _trim_wav(src: Path, dest: Path) -> None:
    tool = _audio_tool()
    if tool is None:
        raise RuntimeError(audio_skip_reason() or "no audio converter")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if tool == "ffmpeg":
        _run_checked(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(src),
                "-t",
                str(MAX_AUDIO_SECONDS),
                "-ac",
                "1",
                "-ar",
                "16000",
                str(dest),
            ],
            "ffmpeg",
        )
        if not dest.is_file():
            raise RuntimeError("ffmpeg wrote no wav")
        return
    converted = dest.with_name(dest.stem + ".afconvert.wav")
    _run_checked(
        [
            "afconvert",
            "-f",
            "WAVE",
            "-d",
            "LEI16@16000",
            "-c",
            "1",
            str(src),
            str(converted),
        ],
        "afconvert",
    )
    try:
        if not converted.is_file():
            raise RuntimeError("afconvert wrote no wav")
        _trim_with_wave(converted, dest)
    finally:
        converted.unlink(missing_ok=True)


def _credit(row: dict) -> str:
    creator = row.get("creator") or "unknown"
    version = row.get("license_version") or ""
    license_name = row.get("license") or ""
    pretty = f"{license_name} {version}".strip().upper()
    return f"{creator} · {pretty}"


def fetch(dest: Path, *, limit_images: int | None, limit_audio: int | None) -> dict:
    dest = assert_outside_artifacts(dest, env_path("ANEMLL_EMBEDDINGS_ARTIFACTS"))
    dest.mkdir(parents=True, exist_ok=True)
    items: list[dict] = []
    seen: set[str] = set()
    image_topics = IMAGE_TOPICS if limit_images is None else IMAGE_TOPICS[:limit_images]
    audio_topics = AUDIO_TOPICS if limit_audio is None else AUDIO_TOPICS[:limit_audio]
    for index, topic in enumerate(image_topics, start=1):
        try:
            row = _pick_image(_search("images", topic))
            if row is None or row.get("url") in seen:
                print(f"skip image {topic}")
                continue
            seen.add(row["url"])
            ext = ".png" if row["url"].lower().split("?")[0].endswith(".png") else ".jpg"
            rel = f"images/{index:02d}-{topic.replace(' ', '-')}{ext}"
            _download(row["url"], dest / rel)
            items.append(
                {
                    "file": rel,
                    "modality": "image",
                    "label": (row.get("title") or topic).strip()[:140],
                    "topic": topic,
                    "license": row.get("license"),
                    "license_version": row.get("license_version"),
                    "creator": row.get("creator"),
                    "credit": _credit(row),
                    "source": row.get("foreign_landing_url") or row.get("url"),
                    "provider": "Openverse",
                }
            )
            print(f"image {topic} · {row.get('license')} · {row.get('title')}")
        except Exception as exc:
            print(f"image {topic} failed: {type(exc).__name__}: {exc}")
        time.sleep(0.35)
    skip = audio_skip_reason()
    if skip and audio_topics:
        print(f"warning: {skip}")
        audio_topics = []
    for index, topic in enumerate(audio_topics, start=1):
        try:
            row = _pick_audio(_search("audio", topic))
            if row is None or row.get("url") in seen:
                print(f"skip audio {topic}")
                continue
            seen.add(row["url"])
            raw = dest / "audio" / f"{index:02d}-raw"
            wav_rel = f"audio/{index:02d}-{topic.replace(' ', '-')}.wav"
            _download(row["url"], raw)
            _trim_wav(raw, dest / wav_rel)
            raw.unlink(missing_ok=True)
            items.append(
                {
                    "file": wav_rel,
                    "modality": "audio",
                    "label": (row.get("title") or topic).strip()[:140],
                    "topic": topic,
                    "license": row.get("license"),
                    "license_version": row.get("license_version"),
                    "creator": row.get("creator"),
                    "credit": _credit(row),
                    "source": row.get("foreign_landing_url") or row.get("url"),
                    "provider": "Openverse",
                    "duration_ms": row.get("duration"),
                }
            )
            print(f"audio {topic} · {row.get('license')} · {row.get('title')}")
        except Exception as exc:
            print(f"audio {topic} failed: {type(exc).__name__}: {exc}")
        time.sleep(0.35)
    manifest = {
        "note": (
            "Small demo corpus for the local EmbeddingGemma showcase. "
            "CC0, public domain, and CC BY only. Not part of the git tree. "
            "Keep the credit line with each file."
        ),
        "source": "https://api.openverse.org/",
        "items": items,
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {dest / 'manifest.json'} items={len(items)}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=default_corpus_dir())
    parser.add_argument("--limit-images", type=int, default=None)
    parser.add_argument("--limit-audio", type=int, default=None)
    args = parser.parse_args()
    fetch(args.dest, limit_images=args.limit_images, limit_audio=args.limit_audio)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
