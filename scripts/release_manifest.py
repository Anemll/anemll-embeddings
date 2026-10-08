#!/usr/bin/env python3
"""Build, check, and verify the release manifest (SHA-256 of release artifacts).

The manifest lists every file of the Hugging Face package at the pinned
``ANE_REVISION`` (towers, ``host/``, card, license files) and the key files of
this repo (runtime, download/warmup scripts, packaging). It is committed as
``release/MANIFEST-<version>.json``; the release workflow
(``.github/workflows/release.yml``) regenerates it on the tag, fails if it
differs from the committed copy, and signs it keyless with Sigstore (cosign).

  python scripts/release_manifest.py build  --version v0.1.0   # write the manifest
  python scripts/release_manifest.py check  --version v0.1.0   # regenerate and compare
  python scripts/release_manifest.py verify --version v0.1.0 [--dest ~/.anemll-embeddings]

``verify`` hashes this checkout and, when present, the downloaded package under
``<dest>/ane`` against the manifest. Check the manifest's signature first; see
``release/README.md``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.download_common import (  # noqa: E402
    ANE_REPO,
    ANE_REVISION,
    BASE_REPO,
    BASE_REVISION,
    HOST_FOLDER,
    HOST_SHA256,
    TOWER_MLIRB_SHA256,
    default_dest,
    sha256_file,
)

SCHEMA = "anemll-embeddings/release-manifest/v1"
# Key repo files covered by the manifest: the runtime, the download / warmup
# path that fetches and checks the package, and the packaging pins.
REPO_GLOBS = (
    "api/*.py",
    "scripts/*.py",
    "scripts/*.sh",
    "hf/towers.yaml",
    "pyproject.toml",
    "constraints.txt",
    "LICENSE",
)
# Hub bookkeeping, not part of the package.
HF_SKIP = {".gitattributes"}


def manifest_path(version: str) -> Path:
    return REPO_ROOT / "release" / f"MANIFEST-{version}.json"


def repo_files() -> list[str]:
    """Git-tracked files matching REPO_GLOBS (sorted, POSIX paths)."""
    out = subprocess.run(
        ["git", "ls-files", "--", *REPO_GLOBS],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return sorted(line for line in out.splitlines() if line)


def hash_repo() -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    for rel in repo_files():
        path = REPO_ROOT / rel
        files[rel] = {"sha256": sha256_file(path), "size": path.stat().st_size}
    return files


def hash_hub(revision: str) -> dict[str, dict[str, Any]]:
    """SHA-256 of every file of ANE_REPO at ``revision``.

    LFS files use the Hub's LFS object id (the SHA-256 of the content); small
    files are downloaded and hashed. Tower and host/ digests are cross-checked
    against the tables tracked in scripts/download_common.py.
    """
    from huggingface_hub import HfApi, hf_hub_download

    info = HfApi().model_info(ANE_REPO, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise SystemExit(f"{ANE_REPO}: asked for {revision}, Hub resolved {info.sha}")
    files: dict[str, dict[str, Any]] = {}
    for sib in sorted(info.siblings, key=lambda s: s.rfilename):
        name = sib.rfilename
        if name in HF_SKIP:
            continue
        if sib.lfs is not None:
            digest = sib.lfs.sha256
        else:
            local = hf_hub_download(ANE_REPO, name, revision=revision)
            digest = sha256_file(Path(local))
        files[name] = {"sha256": digest, "size": sib.size}

    problems: list[str] = []
    for tower, want in TOWER_MLIRB_SHA256.items():
        key = f"{tower}/{tower}.aimodel/main.mlirb"
        if files.get(key, {}).get("sha256") != want:
            problems.append(key)
    for name, want in HOST_SHA256.items():
        key = f"{HOST_FOLDER}/{name}"
        if files.get(key, {}).get("sha256") != want:
            problems.append(key)
    if problems:
        raise SystemExit(
            "Hub digests disagree with scripts/download_common.py: " + ", ".join(problems)
        )
    return files


def build(version: str) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "version": version,
        "source": "https://github.com/Anemll/anemll-embeddings",
        "huggingface": {
            "repo": ANE_REPO,
            "revision": ANE_REVISION,
            "base_repo": BASE_REPO,
            "base_revision": BASE_REVISION,
            "files": hash_hub(ANE_REVISION),
        },
        "repo": {"files": hash_repo()},
    }


def dumps(manifest: dict[str, Any]) -> str:
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"


def sha256sums(manifest: dict[str, Any]) -> str:
    """``shasum -a 256 -c`` style listing (hf/ and repo/ prefixes)."""
    lines = [
        f"{meta['sha256']}  hf/{name}"
        for name, meta in sorted(manifest["huggingface"]["files"].items())
    ]
    lines += [
        f"{meta['sha256']}  repo/{name}" for name, meta in sorted(manifest["repo"]["files"].items())
    ]
    return "\n".join(lines) + "\n"


def cmd_build(args: argparse.Namespace) -> int:
    out = Path(args.out) if args.out else manifest_path(args.version)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = build(args.version)
    out.write_text(dumps(manifest), encoding="utf-8")
    print(f"wrote {out} ({len(manifest['huggingface']['files'])} hub files, "
          f"{len(manifest['repo']['files'])} repo files)")
    if args.sums:
        Path(args.sums).write_text(sha256sums(manifest), encoding="utf-8")
        print(f"wrote {args.sums}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    path = manifest_path(args.version)
    committed = path.read_text(encoding="utf-8")
    fresh = dumps(build(args.version))
    if committed != fresh:
        a = json.loads(committed)
        b = json.loads(fresh)
        for section in ("huggingface", "repo"):
            fa, fb = a[section]["files"], b[section]["files"]
            for name in sorted(set(fa) | set(fb)):
                if fa.get(name) != fb.get(name):
                    print(f"DIFF {section}/{name}: committed={fa.get(name)} now={fb.get(name)}")
        def header(m: dict[str, Any]) -> dict[str, Any]:
            hf = {k: v for k, v in m["huggingface"].items() if k != "files"}
            return {**{k: v for k, v in m.items() if k not in ("huggingface", "repo")}, **hf}

        if header(a) != header(b):
            print(f"DIFF header: committed={header(a)} now={header(b)}")
        print(f"FAIL: {path.relative_to(REPO_ROOT)} is stale; rerun build")
        return 1
    print(f"ok: {path.relative_to(REPO_ROOT)} matches this checkout and the Hub")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    manifest = json.loads(Path(args.manifest or manifest_path(args.version)).read_text())
    bad: list[str] = []
    checked = 0
    for rel, meta in manifest["repo"]["files"].items():
        path = REPO_ROOT / rel
        checked += 1
        if not path.is_file() or sha256_file(path) != meta["sha256"]:
            bad.append(f"repo/{rel}")
    ane_dir = Path(args.dest).expanduser() / "ane" if args.dest else default_dest() / "ane"
    present = 0
    if ane_dir.is_dir():
        for rel, meta in manifest["huggingface"]["files"].items():
            path = ane_dir / rel
            if not path.is_file():
                continue  # card / license files are not downloaded
            present += 1
            checked += 1
            if sha256_file(path) != meta["sha256"]:
                bad.append(f"hf/{rel}")
    print(f"manifest {manifest['version']} @ {manifest['huggingface']['repo']}"
          f"@{manifest['huggingface']['revision']}")
    print(f"checked {checked} files ({present} downloaded package files under {ane_dir})")
    if bad:
        for item in bad:
            print(f"MISMATCH {item}")
        return 1
    print("ok")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build", help="write release/MANIFEST-<version>.json")
    p.add_argument("--version", required=True)
    p.add_argument("--out", help="output path (default: release/MANIFEST-<version>.json)")
    p.add_argument("--sums", help="also write a SHA256SUMS-style listing here")
    p.set_defaults(func=cmd_build)
    p = sub.add_parser("check", help="regenerate and compare with the committed manifest")
    p.add_argument("--version", required=True)
    p.set_defaults(func=cmd_check)
    p = sub.add_parser("verify", help="hash this checkout (+ downloaded package) against it")
    p.add_argument("--version", required=True)
    p.add_argument("--manifest", help="manifest path (default: the committed one)")
    p.add_argument("--dest", help="download parent dir (default: ~/.anemll-embeddings)")
    p.set_defaults(func=cmd_verify)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
