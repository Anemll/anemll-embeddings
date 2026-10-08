#!/usr/bin/env python3
"""Rank a few captions against a query image (or a generated color square).

    python samples/image_text_search.py --backend mock
    python samples/image_text_search.py --image photo.jpg --backend coreai
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from api import Embedder  # noqa: E402

DEFAULT_CAPTIONS = (
    "a red fox in snow",
    "northern lights over a lake",
    "a delivery truck on a street",
    "piano music",
)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, help="JPEG/PNG to search with")
    parser.add_argument("--backend", default="coreai", choices=("coreai", "mock", "reference"))
    parser.add_argument("--caption", action="append", dest="captions")
    args = parser.parse_args(argv)

    if args.image is None:
        image = Image.new("RGB", (64, 64), (20, 80, 160))
    else:
        image = Image.open(args.image).convert("RGB")
    captions = tuple(args.captions) if args.captions else DEFAULT_CAPTIONS

    embedder = Embedder(backend=args.backend)
    query = embedder.embed_image(image)
    ranked = sorted(
        ((_cosine(query, embedder.embed_text(text, role="document")), text) for text in captions),
        reverse=True,
    )
    embedder.close()
    print("query=image")
    for score, text in ranked:
        print(f"  {score:.4f}  {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
