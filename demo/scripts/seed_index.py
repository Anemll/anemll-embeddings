#!/usr/bin/env python3
"""Index a fetched corpus into a running showcase server."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from demo.settings import default_corpus_dir  # noqa: E402

MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".wav": "audio/wav",
    ".webm": "audio/webm",
    ".ogg": "audio/ogg",
}


def seed(base_url: str, corpus: Path, *, limit: int | None) -> int:
    manifest_path = corpus / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"missing {manifest_path} — run demo/scripts/fetch_corpus.py first")
    manifest = json.loads(manifest_path.read_text())
    items = list(manifest.get("items") or [])
    if limit is not None:
        items = items[: int(limit)]
    added = 0
    with httpx.Client(base_url=base_url.rstrip("/"), timeout=120.0) as client:
        health = client.get("/health")
        health.raise_for_status()
        print(f"server backend={health.json().get('backend')}")
        for item in items:
            path = corpus / item["file"]
            if not path.is_file():
                print(f"missing {path}")
                continue
            mime = MIME.get(path.suffix.lower(), "application/octet-stream")
            with path.open("rb") as handle:
                response = client.post(
                    "/index",
                    data={
                        "label": item.get("label") or path.stem,
                        "credit": item.get("credit") or "",
                        "source": item.get("source") or "",
                    },
                    files={"file": (path.name, handle, mime)},
                )
            if response.status_code >= 400:
                print(f"FAIL {path.name} {response.status_code} {response.text[:240]}")
                continue
            added += 1
            body = response.json()
            print(
                f"indexed {item.get('modality')} {item.get('label')} "
                f"{body.get('latency_ms')} ms"
            )
    print(f"added {added}")
    return 0 if added else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--corpus", type=Path, default=default_corpus_dir())
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    return seed(args.base_url, args.corpus, limit=args.limit)


if __name__ == "__main__":
    raise SystemExit(main())
