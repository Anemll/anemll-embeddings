#!/usr/bin/env python3
"""Semantic grep: embed every line of a text file, then rank lines by a query.

    python samples/grep_embed.py notes.txt "how do I reset the cache"
    python samples/grep_embed.py --backend mock README.md "install" --top 5
    python samples/grep_embed.py src.py "parse arguments" --dim 256 --save idx.npz
    python samples/grep_embed.py --load idx.npz "parse arguments"

Lines are embedded as documents with ``Embedder.embed_texts`` (short lines
are packed several to a forward when the packed text tower is installed).
The query is embedded with the search-query prefix. ``--save`` keeps the
line vectors so later queries skip the indexing step.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from samples.cli import add_embedder_args, make_embedder  # noqa: E402


def read_lines(path: Path, *, min_chars: int = 2) -> list[tuple[int, str]]:
    """Non-blank lines as ``(line_number, text)``; numbers start at 1."""
    out = []
    for number, line in enumerate(path.read_text(errors="replace").splitlines(), start=1):
        text = line.strip()
        if len(text) >= min_chars:
            out.append((number, text))
    return out


def rank(query_vec: np.ndarray, line_vecs: np.ndarray, top: int) -> list[tuple[int, float]]:
    scores = line_vecs @ query_vec
    order = np.argsort(-scores, kind="stable")[: max(0, int(top))]
    return [(int(i), float(scores[i])) for i in order]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", nargs="?", type=Path, help="text file to index")
    parser.add_argument("query", nargs="?", help="search query")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--dim", type=int, default=None, help="Matryoshka size: 128/256/512/768")
    parser.add_argument("--no-pack", action="store_true", help="one line per forward")
    parser.add_argument("--chunk", type=int, default=256, help="lines per embed_texts call")
    parser.add_argument("--save", type=Path, default=None, help="write lines + vectors (.npz)")
    parser.add_argument("--load", type=Path, default=None, help="read an index from --save")
    add_embedder_args(parser)
    args = parser.parse_args(argv)
    if args.load is not None and args.query is None and args.file is not None:
        args.query, args.file = str(args.file), None
    if args.query is None or (args.file is None and args.load is None):
        parser.error("need FILE and QUERY, or --load INDEX and QUERY")

    embedder = make_embedder(args)
    try:
        if args.load is not None:
            data = np.load(args.load, allow_pickle=False)
            numbers = data["numbers"].tolist()
            texts = data["texts"].tolist()
            vecs = data["vectors"].astype(np.float32)
            dim = int(vecs.shape[1])
        else:
            rows = read_lines(args.file)
            numbers = [n for n, _ in rows]
            texts = [t for _, t in rows]
            t0 = time.perf_counter()
            parts = []
            calls = 0
            for start in range(0, len(texts), max(1, args.chunk)):
                chunk = texts[start : start + args.chunk]
                parts.append(
                    embedder.embed_texts(
                        chunk, role="document", dim=args.dim, pack=not args.no_pack
                    )
                )
                calls += embedder.last_batch.calls if embedder.last_batch else len(chunk)
            dim = int(args.dim or 768)
            vecs = np.concatenate(parts) if parts else np.zeros((0, dim), dtype=np.float32)
            secs = time.perf_counter() - t0
            rate = len(texts) / secs if secs > 0 else 0.0
            print(
                f"indexed {len(texts)} lines in {secs:.2f} s ({rate:.0f} lines/s, "
                f"{calls} forwards, backend={args.backend})",
                file=sys.stderr,
            )
            if args.save is not None:
                np.savez(
                    args.save,
                    numbers=np.asarray(numbers, dtype=np.int64),
                    texts=np.asarray(texts, dtype=str),
                    vectors=vecs,
                )
        query = embedder.embed_text(args.query, role="query", dim=dim if dim != 768 else None)
    finally:
        embedder.close()
    for idx, score in rank(query, vecs, args.top):
        print(f"{numbers[idx]:>6}  {score:.4f}  {texts[idx]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
