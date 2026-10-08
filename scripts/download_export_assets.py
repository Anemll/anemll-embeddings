#!/usr/bin/env python3
"""Download the FULL Google checkpoint for export / re-conversion.

Use this only if you will run ``model/export_coreai_towers.py``. Everyday
inference uses ``scripts/download_models.py`` (slim host, no 1.49 GB
``model.safetensors``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.download_common import (  # noqa: E402
    BASE_REPO,
    BASE_REVISION,
    EXPORT_HOST_ALLOW,
    coreai_python_export,
    default_dest,
    env_exports,
    export_download_bytes,
    format_gb,
    revision_matches,
    snapshot,
    write_revision,
)


def host_complete(model_dir: Path) -> bool:
    return (model_dir / "model.safetensors").is_file() and (model_dir / "config.json").is_file()


def download_full_host(model_dir: Path, *, force: bool) -> str:
    if not force and host_complete(model_dir) and revision_matches(model_dir, BASE_REVISION):
        return "skipped"
    print(f"download FULL {BASE_REPO} @{BASE_REVISION} → {model_dir}")
    snapshot(BASE_REPO, BASE_REVISION, model_dir, allow_patterns=EXPORT_HOST_ALLOW)
    if not host_complete(model_dir):
        raise SystemExit(f"export download finished but {model_dir / 'model.safetensors'} is missing")
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
    model_dir = dest / "embeddinggemma-2-full"
    artifacts = dest / "artifacts"
    total = export_download_bytes()
    print(
        "Export download (full checkpoint).\n"
        f"https://huggingface.co/{BASE_REPO} @{BASE_REVISION}\n"
        f"About {format_gb(total)} on disk. No HF login."
    )
    status = download_full_host(model_dir, force=bool(args.force))
    py = (
        str(args.coreai_python.expanduser())
        if args.coreai_python is not None
        else coreai_python_export()
    )
    print(f"host={status}")
    print()
    print("# env for model/export_coreai_towers.py:")
    print(env_exports(artifacts=artifacts, model=model_dir, coreai_python=py))
    print("# conversion also needs torch, transformers, sentence-transformers")
    print("# and a Python that can import coreai.runtime (ANEMLL_COREAI_PYTHON)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
