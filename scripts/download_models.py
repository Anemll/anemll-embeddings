#!/usr/bin/env python3
"""Download the public ANE packages and the Google host checkpoint.

Idempotent. Writes:

    <dest>/ane/                         anemll/anemll-embeddinggemma-2-ane @ 8ceba04
    <dest>/embeddinggemma-2/            google/embeddinggemma-2 @ 914f7f8…
    <dest>/artifacts/coreai/<name>.aimodel   symlinks Embedder expects

Then prints the ``export`` lines for ``ANEMLL_EMBEDDINGS_ARTIFACTS``,
``ANEMLL_EMBEDDINGS_MODEL``, and ``ANEMLL_COREAI_PYTHON``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

try:
    from huggingface_hub import snapshot_download as _hf_snapshot_download
except ImportError:
    _hf_snapshot_download = None

ANE_REPO = "anemll/anemll-embeddinggemma-2-ane"
ANE_REVISION = "8ceba04"
BASE_REPO = "google/embeddinggemma-2"
BASE_REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
TOWERS = ("vision_s280", "audio_s280", "text_embeds_s320")
BUNDLE_FILES = ("metadata.json", "main.hash", "main.mlirb")
DEFAULT_COREAI_PY = Path("/Users/anemll/anemll-forge/coreai/.venv/bin/python")
REVISION_MARK = ".anemll-revision"


def default_dest() -> Path:
    raw = os.environ.get("ANEMLL_EMBEDDINGS_HOME")
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".anemll-embeddings"


def bundle_path(ane_dir: Path, name: str) -> Path:
    return ane_dir / name / f"{name}.aimodel"


def bundle_complete(ane_dir: Path, name: str) -> bool:
    root = bundle_path(ane_dir, name)
    if not root.is_dir():
        return False
    return all((root / filename).is_file() for filename in BUNDLE_FILES)


def all_bundles_complete(ane_dir: Path) -> bool:
    return all(bundle_complete(ane_dir, name) for name in TOWERS)


def revision_matches(folder: Path, revision: str) -> bool:
    mark = folder / REVISION_MARK
    return mark.is_file() and mark.read_text(encoding="utf-8").strip() == revision


def write_revision(folder: Path, revision: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / REVISION_MARK).write_text(revision + "\n", encoding="utf-8")


def ensure_symlink(link: Path, target: Path) -> str:
    """Point ``link`` at ``target``. Returns created / ok / replaced."""
    dest = target.expanduser().resolve()
    if not dest.exists():
        raise FileNotFoundError(f"symlink target missing: {dest}")
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        try:
            current = link.resolve()
        except OSError:
            current = None
        if current == dest:
            return "ok"
        link.unlink()
        link.symlink_to(dest, target_is_directory=dest.is_dir())
        return "replaced"
    if link.exists():
        raise FileExistsError(
            f"{link} exists and is not a symlink; move it aside or pick another --dest"
        )
    link.symlink_to(dest, target_is_directory=dest.is_dir())
    return "created"


def link_coreai(artifacts: Path, ane_dir: Path) -> dict[str, str]:
    coreai = artifacts / "coreai"
    coreai.mkdir(parents=True, exist_ok=True)
    status: dict[str, str] = {}
    for name in TOWERS:
        status[name] = ensure_symlink(coreai / f"{name}.aimodel", bundle_path(ane_dir, name))
    return status


def coreai_python_export() -> str:
    raw = os.environ.get("ANEMLL_COREAI_PYTHON")
    if raw:
        return raw
    if DEFAULT_COREAI_PY.is_file():
        return str(DEFAULT_COREAI_PY)
    return str(DEFAULT_COREAI_PY)


def env_exports(*, artifacts: Path, model: Path, coreai_python: str) -> str:
    return "\n".join(
        (
            f"export ANEMLL_EMBEDDINGS_ARTIFACTS={artifacts}",
            f"export ANEMLL_EMBEDDINGS_MODEL={model}",
            f"export ANEMLL_COREAI_PYTHON={coreai_python}",
        )
    )


def _snapshot(repo_id: str, revision: str, local_dir: Path) -> None:
    if _hf_snapshot_download is None:
        raise SystemExit(
            "huggingface_hub is required: python -m pip install huggingface_hub"
        )
    local_dir.mkdir(parents=True, exist_ok=True)
    _hf_snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(local_dir),
    )


def download_ane(ane_dir: Path, *, force: bool) -> str:
    if not force and all_bundles_complete(ane_dir) and revision_matches(ane_dir, ANE_REVISION):
        return "skipped"
    print(f"download {ANE_REPO} @{ANE_REVISION} → {ane_dir}")
    _snapshot(ANE_REPO, ANE_REVISION, ane_dir)
    if not all_bundles_complete(ane_dir):
        missing = [name for name in TOWERS if not bundle_complete(ane_dir, name)]
        raise SystemExit(f"download finished but packages incomplete: {', '.join(missing)}")
    write_revision(ane_dir, ANE_REVISION)
    return "downloaded"


def download_base(model_dir: Path, *, force: bool) -> str:
    config = model_dir / "config.json"
    if not force and config.is_file() and revision_matches(model_dir, BASE_REVISION):
        return "skipped"
    print(f"download {BASE_REPO} @{BASE_REVISION} → {model_dir}")
    _snapshot(BASE_REPO, BASE_REVISION, model_dir)
    if not config.is_file():
        raise SystemExit(f"download finished but {config} is missing")
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

    print(
        "Core AI packages: "
        f"https://huggingface.co/{ANE_REPO} (commit {ANE_REVISION})\n"
        "Host checkpoint: "
        f"https://huggingface.co/{BASE_REPO} (revision {BASE_REVISION})\n"
        "Weights are Apache-2.0 under Google's terms; this repo's code is MIT."
    )
    ane_status = download_ane(ane_dir, force=bool(args.force))
    base_status = download_base(model_dir, force=bool(args.force))
    links = link_coreai(artifacts, ane_dir)
    py = (
        str(args.coreai_python.expanduser())
        if args.coreai_python is not None
        else coreai_python_export()
    )
    print(f"ane={ane_status}  host={base_status}  links={links}")
    print("Embedder expects $ANEMLL_EMBEDDINGS_ARTIFACTS/coreai/<name>.aimodel")
    print()
    print("# add these to your shell:")
    print(env_exports(artifacts=artifacts, model=model_dir, coreai_python=py))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
