#!/usr/bin/env python3
"""Download the camera-alert frames and sounds. Nothing is committed.

Pinned Wikimedia Commons files, checked at download time for an open
license (CC0, public domain, CC BY, or CC BY-SA). NC and ND are refused.
Attribution is written to ``manifest.json``. Audio is stored as 16 kHz
mono WAV, at most 8 seconds, via ffmpeg or macOS afconvert.

Sparky uses two photographs of the same black cat from Nikolai Bulykin's
Medeo series (bench shot as the reference, step shot as the camera frame).
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from demo.alert_catalog import catalog_items  # noqa: E402
from samples.fetch_corpus import _trim_wav, audio_skip_reason  # noqa: E402
from demo.settings import assert_outside_artifacts, default_alert_dir, env_path  # noqa: E402

UA = "anemll-embeddings-demo/1.0 (local educational showcase; https://github.com/Anemll/anemll-embeddings)"
API = "https://commons.wikimedia.org/w/api.php"


def _strip(value: str | None) -> str:
    if not value:
        return ""
    text = re.sub(r"<[^>]+>", "", value)
    return html.unescape(text).replace("\n", " ").strip()


def license_ok(name: str | None) -> bool:
    """CC0, public domain, CC BY, and CC BY-SA. Refuses NC and ND."""
    raw = _strip(name).lower().replace("_", " ")
    compact = re.sub(r"[^a-z0-9]+", "", raw)
    if not compact:
        return False
    if "nc" in compact or "nd" in compact:
        return False
    if compact in {"cc0", "publicdomain", "pdm"} or compact.startswith("cc0"):
        return True
    if "publicdomain" in compact:
        return True
    if compact.startswith("ccby"):
        return True
    return False


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=90) as response:
        dest.write_bytes(response.read())


def _commons_info(titles: list[str]) -> dict[str, dict]:
    params = {
        "action": "query",
        "titles": "|".join(titles),
        "prop": "imageinfo",
        "iiprop": "url|extmetadata|mime",
        "iiurlwidth": "1280",
        "format": "json",
    }
    url = API + "?" + urllib.parse.urlencode(params)
    data = _get(url)
    found: dict[str, dict] = {}
    for page in data.get("query", {}).get("pages", {}).values():
        title = page.get("title")
        info = (page.get("imageinfo") or [None])[0]
        if title and info:
            found[title] = info
    return found


def _credit(meta: dict) -> tuple[str, str, str]:
    def field(key: str) -> str:
        return _strip((meta.get(key) or {}).get("value"))

    license_name = field("LicenseShortName") or "unknown"
    creator = field("Artist") or "unknown"
    return creator, license_name, f"{creator} · {license_name}"


def fetch(dest: Path) -> dict:
    dest = assert_outside_artifacts(dest, env_path("ANEMLL_EMBEDDINGS_ARTIFACTS"))
    items = catalog_items()
    info = _commons_info([item["commons"] for item in items])
    missing = [item["commons"] for item in items if item["commons"] not in info]
    if missing:
        raise RuntimeError("Commons did not return " + ", ".join(missing))
    skip_audio = audio_skip_reason()
    if skip_audio:
        print(f"warning: {skip_audio}")
    written: list[dict] = []
    for item in items:
        meta_info = info[item["commons"]]
        meta = meta_info.get("extmetadata") or {}
        creator, license_name, credit = _credit(meta)
        if not license_ok(license_name):
            raise RuntimeError(f"{item['commons']} license {license_name!r} is not an allowed open license")
        description = meta_info.get("descriptionurl") or (
            "https://commons.wikimedia.org/wiki/" + urllib.parse.quote(item["commons"].replace(" ", "_"))
        )
        if item["modality"] == "audio":
            if skip_audio:
                print(f"skip {item['id']}: {skip_audio}")
                continue
            raw = dest / "audio-src" / f"{item['id']}.bin"
            source_url = meta_info.get("url")
            if not source_url:
                raise RuntimeError(f"{item['commons']} has no download url")
            _download(source_url, raw)
            _trim_wav(raw, dest / item["file"])
            raw.unlink(missing_ok=True)
        else:
            thumb = meta_info.get("thumburl") or meta_info.get("url")
            if not thumb:
                raise RuntimeError(f"{item['commons']} has no image url")
            _download(thumb, dest / item["file"])
        written.append(
            {
                "id": item["id"],
                "file": item["file"],
                "modality": item["modality"],
                "caption": item["caption"],
                "commons": item["commons"],
                "license": license_name,
                "creator": creator,
                "credit": credit,
                "source": description,
            }
        )
        print(f"{item['id']} · {license_name} · {item['commons']}")
    src = dest / "audio-src"
    if src.exists():
        for child in src.iterdir():
            child.unlink(missing_ok=True)
        src.rmdir()
    manifest = {
        "note": (
            "Camera-alert samples for the local showcase. Not part of the git tree. "
            "Sparky's reference and test frame are two photos of the same black cat "
            "(Nikolai Bulykin, Medeo, file numbers 1 and 5). Keep the credit with each file."
        ),
        "items": written,
    }
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {dest / 'manifest.json'} items={len(written)}")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=default_alert_dir())
    args = parser.parse_args(argv)
    fetch(args.dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
