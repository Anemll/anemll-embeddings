"""Shared pins, allow-patterns, and helpers for the download scripts."""

from __future__ import annotations

import json
import os
import struct
import urllib.request
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

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


def inference_download_bytes() -> int:
    return sum(ANE_BYTES.values()) + sum(HOST_FILE_BYTES.values()) + EMBED_TABLE_BYTES


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
    if allow is not None:
        if not any(fnmatch(candidate, pat) for candidate in names for pat in allow):
            return False
    if ignore is not None:
        if any(fnmatch(candidate, pat) for candidate in names for pat in ignore):
            return False
    return True


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
    return str(DEFAULT_COREAI_PY)


def env_exports(*, artifacts: Path, model: Path, coreai_python: str) -> str:
    return "\n".join(
        (
            f"export ANEMLL_EMBEDDINGS_ARTIFACTS={artifacts}",
            f"export ANEMLL_EMBEDDINGS_MODEL={model}",
            f"export ANEMLL_COREAI_PYTHON={coreai_python}",
        )
    )


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
