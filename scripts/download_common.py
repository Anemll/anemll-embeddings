"""Shared pins, allow-patterns, and helpers for the download scripts."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import struct
import urllib.request
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

try:
    from huggingface_hub import snapshot_download as _hf_snapshot_download
except ImportError:
    _hf_snapshot_download = None

from api.runtime_paths import (
    COREAI_PYTHON_ENV,
    CoreAIPythonNotFound,
    coreai_python_candidates,
    resolve_coreai_python,
)

ANE_REPO = "anemll/anemll-embeddinggemma-2-ane"
# Towers + mirrored host/ on the Hub. Override with ANEMLL_ANE_REVISION.
ANE_REVISION = "47d05aa218a227e887858fe571f8deb2f2a1d532"
BASE_REPO = "google/embeddinggemma-2"
BASE_REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
HOST_FOLDER = "host"
TOWERS = ("vision_s280", "audio_s280", "text_embeds_s320")
BUNDLE_FILES = ("metadata.json", "main.hash", "main.mlirb")
# SHA-256 of each tower's main.mlirb at ANE_REVISION (same values as
# hf/towers.yaml). Git-tracked, so a download is checked against this repo,
# not against a manifest fetched from the same place as the payload.
TOWER_MLIRB_SHA256 = {
    "vision_s280": "d11f9d91a41a978ef419cd15b7b3633d47fc388b5d7327b302f4b4355f68b097",
    "audio_s280": "bbbd714866d5b37a2ccd9c8ec0899f966dbf5dac66031f886522a8fc6824c35f",
    "text_embeds_s320": "c52be1ab6401d69bb28b880498a90ca3f5891d419726e1531809584907a2d536",
}
# SHA-256 of the mirrored host/ payload at ANE_REVISION (host/SHA256SUMS).
HOST_SHA256 = {
    "config.json": "b8f1e9931b57fbc054acdb445c41765d55b0074c58d145fa82839941ad1b5bb3",
    "config_sentence_transformers.json": "031e56a498d33c349ab489a21885bcfe25b4fcba841149dc99e1e90d4a7c28f5",
    "tokenizer.json": "4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4",
    "tokenizer.model": "e594c8a90eb08d8bda498ff4747977dc827ae0c3c56b5c0d41a605a22d02ef03",
    "tokenizer_config.json": "17bd5d6e9364ca49a534e1502076593317c298d4a663623091ed45388f004874",
    "preprocessor_config.json": "ea2ae257e901064abdd98dceb19f2b0da06af600bed15e0f99f5c85c37ee9d78",
    "processor_config.json": "168f6a08522f3ce5dea596d94d003af2fd691742d4f41fe1f9d8cce76bfbf69c",
    "chat_template.jinja": "4b852efc0b9960283e735363331e6f325b33bc74bdbaa076f595bc4e9b94d85e",
    "embed_tokens.safetensors": "a37078ff5786d9bf75b059bdc8f4009f4311bf8e7ffbc287b66bab3d38457996",
}
REVISION_MARK = ".anemll-revision"
SLIM_EMBED_NAME = "embed_tokens.safetensors"
EMBED_TENSOR_KEY = "language_model.embed_tokens.weight"
# Gemma scales token rows by sqrt(hidden). Text hidden is 512.
EMBED_SCALE = 512**0.5

# Inference host: tokenizer + processor + configs + prompts. Never model.safetensors.
INFERENCE_HOST_ALLOW = (
    "config.json",
    "config_sentence_transformers.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "chat_template.jinja",
)
INFERENCE_HOST_IGNORE = ("model.safetensors", "*.safetensors")

ANE_ALLOW = (
    "vision_s280/**",
    "audio_s280/**",
    "text_embeds_s320/**",
)
HOST_FOLDER_ALLOW = (f"{HOST_FOLDER}/**",)
# Single-repo inference snapshot: towers + mirrored host/.
ANE_INFERENCE_ALLOW = ANE_ALLOW + HOST_FOLDER_ALLOW
HOST_PAYLOAD = INFERENCE_HOST_ALLOW + (SLIM_EMBED_NAME,)
HOST_META_FILES = ("LICENSE", "NOTICE", "SOURCE.md", "SHA256SUMS")
HOST_REQUIRED = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "preprocessor_config.json",
    SLIM_EMBED_NAME,
)

# Export: the full checkpoint (allow None = every file).
EXPORT_HOST_ALLOW: tuple[str, ...] | None = None

# HF tree sizes at the pinned revisions (mlirb + hash + metadata per tower).
ANE_BYTES = {
    "vision_s280": 307_104_360,
    "audio_s280": 588_591_810,
    "text_embeds_s320": 290_890_780,
}
# google/embeddinggemma-2 @ 914f7f8… files the inference allow-list actually fetches.
HOST_FILE_BYTES = {
    "tokenizer.json": 32_170_510,
    "tokenizer.model": 4_689_013,
    "config.json": 4_455,
    "config_sentence_transformers.json": 1_565,
    "tokenizer_config.json": 1_599,
    "preprocessor_config.json": 511,
    "processor_config.json": 1_788,
    "chat_template.jinja": 1_016,
}
EMBED_TABLE_BYTES = 268_435_456  # BF16 [262144, 512]
FULL_SAFE_TENSORS_BYTES = 1_488_915_288
# Full google/embeddinggemma-2 tree at the pinned revision (export).
EXPORT_REPO_BYTES = 1_525_810_123


def default_dest() -> Path:
    raw = os.environ.get("ANEMLL_EMBEDDINGS_HOME")
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".anemll-embeddings"


def ane_revision() -> str:
    return os.environ.get("ANEMLL_ANE_REVISION") or ANE_REVISION


def inference_download_bytes() -> int:
    return sum(ANE_BYTES.values()) + sum(HOST_FILE_BYTES.values()) + EMBED_TABLE_BYTES


def host_folder_bytes() -> int:
    """Mirrored host/ payload (copied files + extracted embed table)."""
    return sum(HOST_FILE_BYTES.values()) + EMBED_TABLE_BYTES


def export_download_bytes() -> int:
    return EXPORT_REPO_BYTES


def format_gb(num: int) -> str:
    return f"{num / 1_000_000_000:.2f} GB"


def matches_hf_patterns(
    name: str,
    allow: tuple[str, ...] | None,
    ignore: tuple[str, ...] | None = None,
) -> bool:
    """Same idea as huggingface_hub allow/ignore globs (basename or full path)."""
    names = (name, Path(name).name)
    if allow is not None and not any(fnmatch(candidate, pat) for candidate in names for pat in allow):
        return False
    return not (ignore is not None and any(fnmatch(candidate, pat) for candidate in names for pat in ignore))


def bundle_path(ane_dir: Path, name: str) -> Path:
    return ane_dir / name / f"{name}.aimodel"


def bundle_complete(ane_dir: Path, name: str) -> bool:
    root = bundle_path(ane_dir, name)
    if not root.is_dir():
        return False
    return all((root / filename).is_file() for filename in BUNDLE_FILES)


def all_bundles_complete(ane_dir: Path) -> bool:
    return all(bundle_complete(ane_dir, name) for name in TOWERS)


def host_payload_complete(folder: Path) -> bool:
    return all((folder / name).is_file() for name in HOST_REQUIRED)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_host_payload(src: Path, dest: Path) -> list[str]:
    dest.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in HOST_PAYLOAD + HOST_META_FILES:
        source = src / name
        if source.is_file():
            shutil.copy2(source, dest / name)
            copied.append(name)
    return copied


def install_host(src: Path, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink() or not dest.exists():
        return ensure_symlink(dest, src)
    copy_host_payload(src, dest)
    return "copied"


def revision_matches(folder: Path, revision: str) -> bool:
    mark = folder / REVISION_MARK
    return mark.is_file() and mark.read_text(encoding="utf-8").strip() == revision


def write_revision(folder: Path, revision: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / REVISION_MARK).write_text(revision + "\n", encoding="utf-8")


def ensure_symlink(link: Path, target: Path) -> str:
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


def discover_coreai_python(
    explicit: Path | str | None = None,
    *,
    required: bool = True,
) -> Path | None:
    """Resolve a Python that can ``import coreai.runtime``.

    Order: ``explicit`` / ``--coreai-python``, then ``ANEMLL_COREAI_PYTHON``,
    then the documented locations in ``api.runtime_paths``. Exits with setup
    instructions when ``required`` and nothing is found.
    """
    try:
        return resolve_coreai_python(explicit, required=required)
    except CoreAIPythonNotFound as exc:
        raise SystemExit(str(exc)) from exc


def coreai_python_export() -> str | None:
    """Path to print for ``ANEMLL_COREAI_PYTHON``, or ``None`` if none exists yet."""
    found = discover_coreai_python(required=False)
    return str(found) if found is not None else None


def verify_sha256(files: dict[Path, str]) -> list[str]:
    """Return one message per missing or mismatched file (empty = all good)."""
    problems: list[str] = []
    for path, want in files.items():
        if not path.is_file():
            problems.append(f"missing {path}")
            continue
        got = sha256_file(path)
        if got != want:
            problems.append(f"sha256 mismatch {path}: got {got}, want {want}")
    return problems


def tower_checksums(ane_dir: Path) -> dict[Path, str]:
    return {
        bundle_path(ane_dir, name) / "main.mlirb": digest
        for name, digest in TOWER_MLIRB_SHA256.items()
    }


def read_sha256sums(path: Path) -> dict[str, str]:
    """Parse a ``sha256sum``-style file into ``{name: digest}``."""
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2:
            out[parts[1].lstrip("*").strip()] = parts[0].lower()
    return out


def host_checksums(folder: Path) -> tuple[dict[Path, str], list[str]]:
    """Digests to check for a mirrored ``host/`` folder, plus manifest conflicts.

    The git-tracked :data:`HOST_SHA256` table is authoritative for the payload.
    ``folder/SHA256SUMS`` (shipped with the download) adds the files the table
    does not pin (``LICENSE``, ``NOTICE``). A shipped entry that disagrees
    with the pinned table is reported, never trusted.
    """
    files = {folder / name: digest for name, digest in HOST_SHA256.items()}
    conflicts: list[str] = []
    sums = folder / "SHA256SUMS"
    if sums.is_file():
        for name, digest in read_sha256sums(sums).items():
            pinned = HOST_SHA256.get(name)
            if pinned is None:
                files[folder / name] = digest
            elif pinned != digest:
                conflicts.append(f"{sums} lists {name} as {digest}, pinned {pinned}")
    return files, conflicts


def coreai_cache_dir(cache_dir: Path | str | None = None) -> Path:
    """Core AI specialization cache (``…/Library/Caches/coreai-cache``).

    ``--cache-dir`` wins, then ``$CFFIXED_USER_HOME``, then ``Path.home()``.
    """
    if cache_dir is not None:
        return Path(cache_dir).expanduser()
    raw = os.environ.get("CFFIXED_USER_HOME")
    home = Path(raw).expanduser() if raw else Path.home()
    return home / "Library" / "Caches" / "coreai-cache"


def fixed_user_home_for_cache(cache: Path) -> Path | None:
    """Home that makes Core AI write ``cache`` via ``CFFIXED_USER_HOME``."""
    if (
        cache.name == "coreai-cache"
        and cache.parent.name == "Caches"
        and cache.parent.parent.name == "Library"
    ):
        return cache.parent.parent.parent
    return None


def coreai_home_for_cache_dir(cache: Path) -> Path:
    """``CFFIXED_USER_HOME`` for a ``--cache-dir``, or exit with a clear error.

    Core AI has no cache-path option. It writes under
    ``$CFFIXED_USER_HOME/Library/Caches/coreai-cache``, so only a cache
    directory of exactly that shape can be honoured. Anything else would
    be silently ignored, so it is rejected instead.
    """
    home = fixed_user_home_for_cache(Path(cache).expanduser())
    if home is None:
        raise SystemExit(
            f"--cache-dir {cache} is not supported: Core AI only caches under "
            "<home>/Library/Caches/coreai-cache. Pass a path ending in "
            "Library/Caches/coreai-cache, or use --coreai-home <dir> "
            "(cache becomes <dir>/Library/Caches/coreai-cache)."
        )
    return home


def cache_unwritable_hint(cache: Path) -> str:
    notes: list[str] = []
    if cache.is_symlink():
        try:
            cache.resolve(strict=True)
        except OSError:
            notes.append(f"{cache} is a broken symlink.")
    elif cache.exists() and not os.access(cache, os.W_OK):
        notes.append(f"{cache} exists but is not writable.")
    detail = (" ".join(notes) + " ") if notes else ""
    return (
        f"The Core AI worker died during load. {detail}"
        f"The specialization cache ({cache}) may be unwritable or a broken symlink.\n"
        "Fix that path, or redirect Core AI's home:\n"
        "  python scripts/warmup.py --coreai-home /path/to/writable/home\n"
        "  # same as: export CFFIXED_USER_HOME=/path/to/writable/home\n"
        "  # cache becomes /path/to/writable/home/Library/Caches/coreai-cache"
    )


def env_exports(*, artifacts: Path, model: Path, coreai_python: str | None) -> str:
    """Shell lines to paste. Values are ``shlex.quote``d (spaces, ``$``, quotes)."""
    lines = [
        f"export ANEMLL_EMBEDDINGS_ARTIFACTS={shlex.quote(str(artifacts))}",
        f"export ANEMLL_EMBEDDINGS_MODEL={shlex.quote(str(model))}",
    ]
    if coreai_python:
        lines.append(f"export {COREAI_PYTHON_ENV}={shlex.quote(str(coreai_python))}")
    else:
        looked = ", ".join(str(path) for path in coreai_python_candidates())
        lines.append(
            f"# {COREAI_PYTHON_ENV}: no Core AI Python found (looked in: {looked}).\n"
            f"# Set it up first (README.md, \"Core AI runtime\"), then:\n"
            f"# export {COREAI_PYTHON_ENV}=/path/to/coreai-venv/bin/python"
        )
    return "\n".join(lines)


def snapshot(
    repo_id: str,
    revision: str,
    local_dir: Path,
    *,
    allow_patterns: tuple[str, ...] | None = None,
    ignore_patterns: tuple[str, ...] | None = None,
) -> None:
    if _hf_snapshot_download is None:
        raise SystemExit(
            "huggingface_hub is required: python -m pip install huggingface_hub"
        )
    local_dir.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "revision": revision,
        "local_dir": str(local_dir),
    }
    if allow_patterns is not None:
        kwargs["allow_patterns"] = list(allow_patterns)
    if ignore_patterns is not None:
        kwargs["ignore_patterns"] = list(ignore_patterns)
    _hf_snapshot_download(**kwargs)


def _http_range(url: str, start: int, end: int) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes={start}-{end}",
            "User-Agent": "anemll-embeddings",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def parse_safetensors_header(blob: bytes) -> tuple[int, dict[str, Any]]:
    if len(blob) < 8:
        raise ValueError("safetensors header prefix missing")
    header_len = struct.unpack_from("<Q", blob, 0)[0]
    need = 8 + header_len
    if len(blob) < need:
        raise ValueError(f"safetensors header truncated ({len(blob)} < {need})")
    header = json.loads(blob[8:need].decode("utf-8"))
    return header_len, header


def safetensors_data_offset(header_len: int, start: int) -> int:
    return 8 + header_len + int(start)


def write_safetensors(path: Path, tensors: dict[str, dict[str, Any]]) -> None:
    """Write a tiny safetensors file. Each value has dtype, shape, data (bytes)."""
    ordered = list(tensors.items())
    header: dict[str, Any] = {}
    payload = bytearray()
    cursor = 0
    for name, spec in ordered:
        raw: bytes = spec["data"]
        header[name] = {
            "dtype": spec["dtype"],
            "shape": list(spec["shape"]),
            "data_offsets": [cursor, cursor + len(raw)],
        }
        payload.extend(raw)
        cursor += len(raw)
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(payload))


def write_slim_embed(dest: Path, *, dtype: str, shape: list[int], data: bytes) -> None:
    scale = struct.pack("<f", float(EMBED_SCALE))
    write_safetensors(
        dest,
        {
            "embed_tokens.weight": {
                "dtype": dtype,
                "shape": list(shape),
                "data": data,
            },
            "embed_scale": {"dtype": "F32", "shape": [1], "data": scale},
        },
    )


def extract_embed_from_bytes(raw: bytes, dest: Path) -> None:
    header_len, header = parse_safetensors_header(raw)
    info = header.get(EMBED_TENSOR_KEY)
    if not isinstance(info, dict):
        raise KeyError(f"missing {EMBED_TENSOR_KEY} in safetensors header")
    start, end = info["data_offsets"]
    abs_start = safetensors_data_offset(header_len, start)
    data = raw[abs_start : safetensors_data_offset(header_len, end)]
    if len(data) != end - start:
        raise ValueError(f"embed tensor short: {len(data)} != {end - start}")
    write_slim_embed(dest, dtype=str(info["dtype"]), shape=list(info["shape"]), data=data)


def extract_embed_from_file(src: Path, dest: Path) -> None:
    with src.open("rb") as handle:
        prefix = handle.read(8)
        header_len = struct.unpack_from("<Q", prefix, 0)[0]
        header_blob = prefix + handle.read(header_len)
        _, header = parse_safetensors_header(header_blob)
        info = header.get(EMBED_TENSOR_KEY)
        if not isinstance(info, dict):
            raise KeyError(f"missing {EMBED_TENSOR_KEY}")
        start, end = info["data_offsets"]
        handle.seek(safetensors_data_offset(header_len, start))
        data = handle.read(end - start)
    if len(data) != end - start:
        raise ValueError(f"embed tensor short: {len(data)} != {end - start}")
    write_slim_embed(dest, dtype=str(info["dtype"]), shape=list(info["shape"]), data=data)


def extract_embed_from_hf(dest: Path) -> None:
    url = (
        f"https://huggingface.co/{BASE_REPO}/resolve/{BASE_REVISION}/model.safetensors"
    )
    prefix = _http_range(url, 0, 7)
    header_len = struct.unpack_from("<Q", prefix, 0)[0]
    head = _http_range(url, 0, 8 + header_len - 1)
    _, header = parse_safetensors_header(head)
    info = header.get(EMBED_TENSOR_KEY)
    if not isinstance(info, dict):
        raise KeyError(f"missing {EMBED_TENSOR_KEY}")
    start, end = info["data_offsets"]
    abs_start = safetensors_data_offset(header_len, start)
    abs_end = safetensors_data_offset(header_len, end) - 1
    print(f"extract {EMBED_TENSOR_KEY} ({end - start} bytes) from HF range request")
    data = _http_range(url, abs_start, abs_end)
    if len(data) != end - start:
        raise ValueError(f"range short: {len(data)} != {end - start}")
    write_slim_embed(dest, dtype=str(info["dtype"]), shape=list(info["shape"]), data=data)


def ensure_slim_embed(model_dir: Path, *, force: bool) -> str:
    dest = model_dir / SLIM_EMBED_NAME
    if dest.is_file() and not force:
        return "skipped"
    local_full = model_dir / "model.safetensors"
    if local_full.is_file():
        print(f"extract embed table from local {local_full} (not downloaded)")
        extract_embed_from_file(local_full, dest)
        return "extracted-local"
    extract_embed_from_hf(dest)
    return "extracted-hf"
