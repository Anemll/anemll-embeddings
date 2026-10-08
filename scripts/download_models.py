#!/usr/bin/env python3
"""Download INFERENCE assets from one HF repo (ANE towers + host/).

Prefers ``host/`` on ``anemll/anemll-embeddinggemma-2-ane`` (single download).
If that folder is not on the pinned revision yet, falls back to the slim
Google host files (tokenizer / processor / extracted embed table) — never
``model.safetensors`` (1.49 GB).

Writes:

    <dest>/ane/                              ANE towers (+ host/ when mirrored)
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
    ANE_INFERENCE_ALLOW,
    ANE_REPO,
    ANE_REVISION,
    BASE_REPO,
    BASE_REVISION,
    HOST_FOLDER,
    HOST_FOLDER_ALLOW,
    HOST_SHA256,
    INFERENCE_HOST_ALLOW,
    INFERENCE_HOST_IGNORE,
    SLIM_EMBED_NAME,
    TOWERS,
    all_bundles_complete,
    ane_revision,
    coreai_python_export,
    default_dest,
    ensure_slim_embed,
    env_exports,
    format_gb,
    host_checksums,
    host_payload_complete,
    inference_download_bytes,
    install_host,
    link_coreai,
    revision_matches,
    snapshot,
    tower_checksums,
    verify_sha256,
    write_revision,
)


def _check(problems: list[str], what: str) -> None:
    if problems:
        raise SystemExit(
            f"{what} failed checksum verification:\n  "
            + "\n  ".join(problems)
            + "\nDelete the folder or re-run with --force."
        )


def verify_ane(ane_dir: Path, rev: str) -> str:
    """Check tower ``main.mlirb`` digests against the git-tracked table."""
    if rev != ANE_REVISION:
        print(f"note: checksums are pinned for {ANE_REVISION}; not verifying towers at override {rev}")
        return "unpinned"
    _check(verify_sha256(tower_checksums(ane_dir)), "ANE towers")
    return "ok"


def verify_mirrored_host(folder: Path, rev: str) -> str:
    if rev != ANE_REVISION:
        return "unpinned"
    files, conflicts = host_checksums(folder)
    _check(conflicts + verify_sha256(files), f"host/ ({folder})")
    return "ok"


def verify_google_host(folder: Path) -> str:
    """Verbatim Google files must match the mirrored copies' digests."""
    files = {
        folder / name: digest
        for name, digest in HOST_SHA256.items()
        if name in INFERENCE_HOST_ALLOW and (folder / name).is_file()
    }
    _check(verify_sha256(files), f"Google host files ({folder})")
    return "ok"


def download_ane(ane_dir: Path, *, force: bool) -> str:
    rev = ane_revision()
    if not force and all_bundles_complete(ane_dir) and revision_matches(ane_dir, rev):
        return "skipped"
    print(f"download {ANE_REPO} @{rev} → {ane_dir}")
    snapshot(ANE_REPO, rev, ane_dir, allow_patterns=ANE_INFERENCE_ALLOW)
    if not all_bundles_complete(ane_dir):
        missing = [name for name in TOWERS if not (ane_dir / name / f"{name}.aimodel").exists()]
        raise SystemExit(f"download finished but packages incomplete: {', '.join(missing)}")
    # Verify before the revision marker, so a bad download is never trusted later.
    verify_ane(ane_dir, rev)
    write_revision(ane_dir, rev)
    return "downloaded"


def fetch_mirrored_host(ane_dir: Path) -> bool:
    mirrored = ane_dir / HOST_FOLDER
    if host_payload_complete(mirrored):
        return True
    rev = ane_revision()
    print(f"try mirrored host/ from {ANE_REPO} @{rev}")
    try:
        snapshot(ANE_REPO, rev, ane_dir, allow_patterns=HOST_FOLDER_ALLOW)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - fall back to Google's host files
        print(f"mirrored host/ fetch failed ({type(exc).__name__}: {exc})")
        return False
    return host_payload_complete(mirrored)


def download_host(model_dir: Path, *, ane_dir: Path, force: bool) -> str:
    if not force and host_payload_complete(model_dir):
        return "skipped"
    mirrored = ane_dir / HOST_FOLDER
    if fetch_mirrored_host(ane_dir):
        verify_mirrored_host(mirrored, ane_revision())
        status = install_host(mirrored, model_dir)
        write_revision(model_dir, ane_revision())
        print(f"using mirrored {ANE_REPO}/{HOST_FOLDER} ({status})")
        return "anemll-host"
    print(
        f"mirrored host/ not on {ANE_REPO} @{ane_revision()}; "
        f"falling back to {BASE_REPO} @{BASE_REVISION}"
    )
    snapshot(
        BASE_REPO,
        BASE_REVISION,
        model_dir,
        allow_patterns=INFERENCE_HOST_ALLOW,
        ignore_patterns=INFERENCE_HOST_IGNORE,
    )
    verify_google_host(model_dir)
    write_revision(model_dir, BASE_REVISION)
    return "google-fallback"


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
        "--verify",
        action="store_true",
        help="re-hash files already on disk against the pinned SHA-256 table "
        "(fresh downloads are always verified)",
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
    rev = ane_revision()

    print(
        "Inference download (single repo when host/ is on the pin).\n"
        f"https://huggingface.co/{ANE_REPO} @{rev}\n"
        f"Prefers {ANE_REPO}/{HOST_FOLDER}; falls back to {BASE_REPO} @{BASE_REVISION}.\n"
        f"About {format_gb(total)} on disk. No HF login. Weights Apache-2.0; code MIT."
    )
    ane_status = download_ane(ane_dir, force=bool(args.force))
    host_status = download_host(model_dir, ane_dir=ane_dir, force=bool(args.force))
    embed_status = "mirrored"
    if not host_payload_complete(model_dir):
        embed_status = ensure_slim_embed(model_dir, force=bool(args.force))
    if not (model_dir / SLIM_EMBED_NAME).is_file():
        raise SystemExit(f"missing {model_dir / SLIM_EMBED_NAME}")
    if args.verify:
        print(f"verify towers: {verify_ane(ane_dir, rev)}")
        if (ane_dir / HOST_FOLDER).is_dir():
            print(f"verify host/: {verify_mirrored_host(ane_dir / HOST_FOLDER, rev)}")
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
