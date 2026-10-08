#!/usr/bin/env python3
"""Download a small CC0 / CC BY demo corpus outside git.

Images and short sounds come from Openverse (Wikimedia, Flickr, Freesound,
and similar). Each file's license, creator, and source page are written to
``manifest.json``. Audio is trimmed to at most 8 seconds and stored as
16 kHz mono WAV. Nothing is committed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from demo.settings import default_corpus_dir  # noqa: E402

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


def _trim_wav(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(src),
            "-t",
            "8",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(dest),
        ],
        check=False,
        capture_output=True,
    )
    if proc.returncode != 0 or not dest.is_file():
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(err or f"ffmpeg rc={proc.returncode}")


def _credit(row: dict) -> str:
    creator = row.get("creator") or "unknown"
    version = row.get("license_version") or ""
    license_name = row.get("license") or ""
    pretty = f"{license_name} {version}".strip().upper()
    return f"{creator} · {pretty}"


def fetch(dest: Path, *, limit_images: int | None, limit_audio: int | None) -> dict:
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
