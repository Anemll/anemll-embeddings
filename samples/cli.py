"""Shared CLI flags for the Embedder examples."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Any

from api import Embedder


def add_embedder_args(parser: ArgumentParser) -> None:
    parser.add_argument("--backend", default="coreai", choices=("coreai", "mock", "reference"))
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=None,
        help="Core AI packages dir (ANEMLL_EMBEDDINGS_ARTIFACTS)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="EmbeddingGemma 2 checkpoint (ANEMLL_EMBEDDINGS_MODEL)",
    )
    parser.add_argument(
        "--coreai-python",
        type=Path,
        default=None,
        dest="coreai_python",
        help="interpreter with coreai.runtime (ANEMLL_COREAI_PYTHON)",
    )


def make_embedder(args: Any, **kwargs: Any) -> Embedder:
    return Embedder(
        artifacts=args.artifacts,
        model=args.model,
        coreai_python=args.coreai_python,
        backend=args.backend,
        **kwargs,
    )
