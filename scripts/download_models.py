#!/usr/bin/env python3
"""Download INFERENCE assets only: ANE towers + slim host files.

Does **not** download ``google/embeddinggemma-2`` ``model.safetensors``
(1.49 GB). Host lookup uses a 256 MiB embed table extracted from that file
(range request, or from a local full checkpoint if you already have one).

Writes:

    <dest>/ane/                              ANE towers @ 8ceba04
    <dest>/embeddinggemma-2/                 tokenizer / processor / embed_tokens.safetensors
    <dest>/artifacts/coreai/<name>.aimodel   symlinks Embedder expects
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.download_common import (  # noqa: E402
    ANE_ALLOW,
    ANE_REPO,
    ANE_REVISION,
    BASE_REPO,
    BASE_REVISION,
    INFERENCE_HOST_ALLOW,
    INFERENCE_HOST_IGNORE,
    SLIM_EMBED_NAME,
    TOWERS,
    all_bundles_complete,
    coreai_python_export,
    default_dest,
    ensure_slim_embed,
    env_exports,
    format_gb,
    inference_download_bytes,
    link_coreai,
    revision_matches,
    snapshot,
    write_revision,
)


def download_ane(ane_dir: Path, *, force: bool) -> str:
    if not force and all_bundles_complete(ane_dir) and revision_matches(ane_dir, ANE_REVISION):
        return "skipped"
    print(f"download {ANE_REPO} @{ANE_REVISION} → {ane_dir}")
    snapshot(ANE_REPO, ANE_REVISION, ane_dir, allow_patterns=ANE_ALLOW)
    if not all_bundles_complete(ane_dir):
        missing = [name for name in TOWERS if not (ane_dir / name / f"{name}.aimodel").exists()]
        raise SystemExit(f"download finished but packages incomplete: {', '.join(missing)}")
    write_revision(ane_dir, ANE_REVISION)
    return "downloaded"


def host_complete(model_dir: Path) -> bool:
    needed = ("config.json", "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json")
    return all((model_dir / name).is_file() for name in needed) and (model_dir / SLIM_EMBED_NAME).is_file()


def download_host(model_dir: Path, *, force: bool) -> str:
    if not force and host_complete(model_dir) and revision_matches(model_dir, BASE_REVISION):
        return "skipped"
    print(f"download slim host {BASE_REPO} @{BASE_REVISION} → {model_dir}")
    snapshot(
        BASE_REPO,
        BASE_REVISION,
        model_dir,
        allow_patterns=INFERENCE_HOST_ALLOW,
        ignore_patterns=INFERENCE_HOST_IGNORE,
    )
    write_revision(model_dir, BASE_REVISION)
    return "downloaded"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        default=default_dest(),
        help="parent directory (default: $ANEMLL_EMBEDDINGS_HOME or ~/.anemll-embeddings)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download even when the pinned revision is already present",
    )
    parser.add_argument(
        "--coreai-python",
        type=Path,
        default=None,
        help="value printed for ANEMLL_COREAI_PYTHON",
    )
    args = parser.parse_args(argv)

    dest = args.dest.expanduser().resolve()
    ane_dir = dest / "ane"
    model_dir = dest / "embeddinggemma-2"
    artifacts = dest / "artifacts"
    total = inference_download_bytes()

    print(
        "Inference download (no full model.safetensors).\n"
        f"ANE: https://huggingface.co/{ANE_REPO} @{ANE_REVISION}\n"
        f"Host files + embed table: https://huggingface.co/{BASE_REPO} @{BASE_REVISION}\n"
        f"About {format_gb(total)} on disk. No HF login. Weights Apache-2.0; code MIT."
    )
    ane_status = download_ane(ane_dir, force=bool(args.force))
    host_status = download_host(model_dir, force=bool(args.force))
    embed_status = ensure_slim_embed(model_dir, force=bool(args.force))
    if not (model_dir / SLIM_EMBED_NAME).is_file():
        raise SystemExit(f"missing {model_dir / SLIM_EMBED_NAME}")
    links = link_coreai(artifacts, ane_dir)
    py = (
        str(args.coreai_python.expanduser())
        if args.coreai_python is not None
        else coreai_python_export()
    )
    print(f"ane={ane_status}  host={host_status}  embed={embed_status}  links={links}")
    print()
    print("# add these to your shell (~/.zshrc or a sourced env file):")
    print(env_exports(artifacts=artifacts, model=model_dir, coreai_python=py))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
