#!/usr/bin/env python3
"""Stage ``host/`` for upload to anemll/anemll-embeddinggemma-2-ane.

Copies the Google host files ``api.Embedder`` needs (tokenizer, processor /
preprocessor configs, ``config.json``) verbatim, extracts the embedding table
from ``model.safetensors``, and writes checksums plus ``host/SOURCE.md``.

Writes **only** under ``--dest`` (default ``hf/``). Never writes
``embed_tokens.safetensors`` (or anything else) into ``--src``.

``--license`` and ``--notice`` are required files (default:
``hf/LICENSE`` and ``hf/NOTICE`` in this repo). They are copied into
``host/``.

Does **not** upload. Run on the Mac that will push the folder to the Hub:

    python scripts/prepare_hf_host_folder.py
    python scripts/prepare_hf_host_folder.py --src ~/.anemll-embeddings/embeddinggemma-2-full
    python scripts/prepare_hf_host_folder.py --license hf/LICENSE --notice hf/NOTICE
    # then upload hf/ (README, towers.yaml, LICENSE, NOTICE, host/) to the Hub
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.download_common import (  # noqa: E402
    BASE_REPO,
    BASE_REVISION,
    EMBED_SCALE,
    EMBED_TENSOR_KEY,
    HOST_FILE_BYTES,
    HOST_FOLDER,
    HOST_META_FILES,
    HOST_PAYLOAD,
    INFERENCE_HOST_ALLOW,
    INFERENCE_HOST_IGNORE,
    SLIM_EMBED_NAME,
    extract_embed_from_file,
    extract_embed_from_hf,
    format_gb,
    host_folder_bytes,
    host_payload_complete,
    sha256_file,
    snapshot,
)

ORIGIN_COPIED = "copied verbatim"
ORIGIN_EXTRACTED = "extracted"
ORIGIN_ADDED = "added for redistribution"


def _copy_verbatim(src: Path, dest: Path, name: str) -> dict[str, str | int]:
    source = src / name
    if not source.is_file():
        raise SystemExit(f"missing {source} (needed as a verbatim copy)")
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest / name)
    return {
        "file": name,
        "origin": ORIGIN_COPIED,
        "upstream": f"{BASE_REPO}@{BASE_REVISION}/{name}",
        "bytes": (dest / name).stat().st_size,
        "sha256": sha256_file(dest / name),
    }


def _write_sha256sums(host: Path, rows: list[dict[str, str | int]]) -> None:
    lines = [f"{row['sha256']}  {row['file']}\n" for row in rows]
    (host / "SHA256SUMS").write_text("".join(lines), encoding="utf-8")


def write_source_md(host: Path, rows: list[dict[str, str | int]]) -> None:
    table = [
        "| File | Origin | Upstream | SHA256 | Bytes |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        table.append(
            f"| `{row['file']}` | {row['origin']} | `{row['upstream']}` | `{row['sha256']}` | {row['bytes']} |"
        )
    body = f"""# Host-side files

Upstream: [{BASE_REPO}](https://huggingface.co/{BASE_REPO})
Revision: `{BASE_REVISION}`
Tree: https://huggingface.co/{BASE_REPO}/tree/{BASE_REVISION}

Apache-2.0 allows this redistribution. Files listed as **copied verbatim**
are byte-identical to the upstream objects. The embedding table is the
exception: it is **extracted** from `model.safetensors` tensor
`{EMBED_TENSOR_KEY}` (BF16 `[262144, 512]`) and stored as
`{SLIM_EMBED_NAME}` with Gemma's `embed_scale = sqrt(512)` ≈ {EMBED_SCALE:.6f}
recorded beside the raw matrix. The full 1.49 GB `model.safetensors` is
**not** mirrored.

See `LICENSE` and `NOTICE` in this folder (and at the repo root).

{chr(10).join(table)}
"""
    (host / "SOURCE.md").write_text(body, encoding="utf-8")


def stage_host(*, src: Path, host: Path, license_src: Path, notice_src: Path) -> list[dict[str, str | int]]:
    host.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | int]] = []
    for name in INFERENCE_HOST_ALLOW:
        rows.append(_copy_verbatim(src, host, name))

    dest_embed = host / SLIM_EMBED_NAME
    slim_src = src / SLIM_EMBED_NAME
    if slim_src.is_file():
        shutil.copy2(slim_src, dest_embed)
        origin = ORIGIN_EXTRACTED
        upstream = (
            f"{BASE_REPO}@{BASE_REVISION}/model.safetensors "
            f"({EMBED_TENSOR_KEY}; already extracted beside the source)"
        )
    else:
        local_full = src / "model.safetensors"
        if local_full.is_file():
            extract_embed_from_file(local_full, dest_embed)
        else:
            extract_embed_from_hf(dest_embed)
        if not dest_embed.is_file():
            raise SystemExit(f"failed to extract {dest_embed}")
        origin = ORIGIN_EXTRACTED
        upstream = f"{BASE_REPO}@{BASE_REVISION}/model.safetensors ({EMBED_TENSOR_KEY})"
    rows.append(
        {
            "file": SLIM_EMBED_NAME,
            "origin": origin,
            "upstream": upstream,
            "bytes": (host / SLIM_EMBED_NAME).stat().st_size,
            "sha256": sha256_file(host / SLIM_EMBED_NAME),
        }
    )

    for name, source in (("LICENSE", license_src), ("NOTICE", notice_src)):
        if not source.is_file():
            raise SystemExit(f"missing {source}")
        shutil.copy2(source, host / name)
        rows.append(
            {
                "file": name,
                "origin": ORIGIN_ADDED,
                "upstream": f"anemll-embeddings hf/{name} (Apache-2.0 / attribution)",
                "bytes": (host / name).stat().st_size,
                "sha256": sha256_file(host / name),
            }
        )

    _write_sha256sums(host, rows)
    sums = host / "SHA256SUMS"
    rows.append(
        {
            "file": "SHA256SUMS",
            "origin": ORIGIN_ADDED,
            "upstream": "generated by scripts/prepare_hf_host_folder.py",
            "bytes": sums.stat().st_size,
            "sha256": sha256_file(sums),
        }
    )
    write_source_md(host, rows)
    return rows


def resolve_src(src: Path | None) -> tuple[Path, str]:
    if src is not None:
        root = src.expanduser().resolve()
        if not root.is_dir():
            raise SystemExit(f"--src is not a directory: {root}")
        return root, "local"
    tmp = Path(tempfile.mkdtemp(prefix="anemll-host-src-"))
    print(f"download slim host {BASE_REPO} @{BASE_REVISION} → {tmp}")
    snapshot(
        BASE_REPO,
        BASE_REVISION,
        tmp,
        allow_patterns=INFERENCE_HOST_ALLOW,
        ignore_patterns=INFERENCE_HOST_IGNORE,
    )
    return tmp, "downloaded"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        type=Path,
        default=None,
        help="local google/embeddinggemma-2 dir (or slim host). Default: download the allow-list.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=REPO_ROOT / "hf",
        help="HF upload staging root (only this tree is written; host/ lives here)",
    )
    parser.add_argument(
        "--license",
        type=Path,
        default=REPO_ROOT / "hf" / "LICENSE",
        help="Apache-2.0 text copied into host/LICENSE (required file; default: hf/LICENSE)",
    )
    parser.add_argument(
        "--notice",
        type=Path,
        default=REPO_ROOT / "hf" / "NOTICE",
        help="NOTICE copied into host/NOTICE (required file; default: hf/NOTICE)",
    )
    args = parser.parse_args(argv)

    dest = args.dest.expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    license_src = args.license.expanduser().resolve()
    notice_src = args.notice.expanduser().resolve()
    if not license_src.is_file() or not notice_src.is_file():
        raise SystemExit(
            "LICENSE and NOTICE are required. Pass --license and --notice "
            f"(looked for {license_src} and {notice_src}; "
            "defaults are hf/LICENSE and hf/NOTICE in this repo)."
        )

    src, src_kind = resolve_src(args.src)
    host = dest / HOST_FOLDER
    rows = stage_host(src=src, host=host, license_src=license_src, notice_src=notice_src)
    if not host_payload_complete(host):
        raise SystemExit(f"host/ incomplete at {host}")

    total = sum(int(row["bytes"]) for row in rows)
    print(f"src={src_kind} {src}")
    print(f"staged {host}  ({total} bytes, about {format_gb(total)})")
    print(f"expected payload ~{host_folder_bytes()} bytes "
          f"(tokenizer/configs {sum(HOST_FILE_BYTES.values())} + embed table)")
    print("payload:", ", ".join(HOST_PAYLOAD))
    print("meta:", ", ".join(HOST_META_FILES))
    for row in rows:
        print(f"  {row['origin']:16} {row['bytes']:12}  {row['file']}")
    print()
    print("Wrote only under", dest, "(source snapshot was not modified).")
    print("Do not upload from this script. Upload hf/ (card + host/) from a Mac.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
