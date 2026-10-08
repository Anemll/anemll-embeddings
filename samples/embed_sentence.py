#!/usr/bin/env python3
"""Embed one sentence and print the 768-d unit vector.

    python samples/embed_sentence.py "a red fox"
    python samples/embed_sentence.py --backend mock "a red fox"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from samples.cli import add_embedder_args, make_embedder  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="?", default="a red fox")
    parser.add_argument("--role", default="query", help="query/document or SearchQuery/Document")
    add_embedder_args(parser)
    args = parser.parse_args(argv)

    embedder = make_embedder(args)
    vec = embedder.embed_text(args.text, role=args.role)
    embedder.close()
    print(f"text={args.text!r} dim={vec.shape[0]} l2={float(np.linalg.norm(vec)):.6f}")
    print(" ".join(f"{x:.6f}" for x in vec[:8]), "...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
