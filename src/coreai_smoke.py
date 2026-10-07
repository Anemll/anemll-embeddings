"""Core AI tower smoke helpers (no ``coreai`` / checkpoint import).

Dummy I/O matches ``tower_io_spec`` / PR #6. Placement helpers classify a
device run list: begin/end CPU↔device switches are OK; mid-graph CPU islands
are flagged. Cache-manifest strings from forge are coarse (ANE / GPU / mixed).
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Keep these in sync with ``src/coreai_towers.py`` (no ``src`` package import —
# Core AI venv has no sentence-transformers, and ``src/__init__.py`` pulls it).
VISION_PATCHES = 2520
VISION_PATCH_DIM = 768
VISION_SOFT_TOKENS = 280
AUDIO_FRAMES = 280
AUDIO_FEAT = 128
AUDIO_SOFT_TOKENS = 70
TEXT_HIDDEN = 512
TEXT_EMBED = 768

TOWER_PACKAGES = {
    "vision": "vision_s280.aimodel",
    "audio": "audio_s280.aimodel",
    "text": "text_s128.aimodel",
    "text_embeds": "text_embeds_s320.aimodel",
}

TOWER_ENTRIES = {
    "vision": "vision_s280",
    "audio": "audio_s280",
    "text": "text_s128",
    "text_embeds": "text_embeds_s320",
}

TEXT_SEQ_LEN = 128
TEXT_EMBEDS_S = 320

_DTYPE_ALIASES = {
    "float16": np.float16,
    "fp16": np.float16,
    "float32": np.float32,
    "fp32": np.float32,
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "uint8": np.uint8,
    "bool": np.bool_,
    "boolean": np.bool_,
}

_ACCEL = frozenset({"ANE", "GPU"})


def tower_smoke_io(name: str) -> dict[str, Any]:
    """Fixed shapes used for smoke (text S=128)."""
    if name == "vision":
        return {
            "inputs": {
                "pixel_values": [1, VISION_PATCHES, VISION_PATCH_DIM],
                "pixel_position_ids": [1, VISION_PATCHES, 2],
            },
            "outputs": {"soft_tokens": [1, VISION_SOFT_TOKENS, TEXT_HIDDEN]},
        }
    if name == "audio":
        return {
            "inputs": {
                "input_features": [1, AUDIO_FRAMES, AUDIO_FEAT],
                "input_features_mask": [1, AUDIO_FRAMES],
            },
            "outputs": {"soft_tokens": [1, AUDIO_SOFT_TOKENS, TEXT_HIDDEN]},
        }
    if name == "text":
        return {
            "inputs": {"input_ids": [1, TEXT_SEQ_LEN], "attention_mask": [1, TEXT_SEQ_LEN]},
            "outputs": {"embedding": [1, TEXT_EMBED]},
        }
    if name == "text_embeds":
        return {
            "inputs": {
                "inputs_embeds": [1, TEXT_EMBEDS_S, TEXT_HIDDEN],
                "attention_mask": [1, TEXT_EMBEDS_S],
            },
            "outputs": {"embedding": [1, TEXT_EMBED]},
        }
    raise ValueError(f"unknown tower {name!r}")


def resolve_dtype(name: str, default: np.dtype) -> np.dtype:
    key = str(name).strip().lower().replace(" ", "")
    if key in _DTYPE_ALIASES:
        return np.dtype(_DTYPE_ALIASES[key])
    try:
        return np.dtype(key)
    except TypeError:
        return np.dtype(default)


def dummy_numpy_inputs(
    name: str, dtypes: dict[str, str] | None = None
) -> dict[str, np.ndarray]:
    """One legal feed per tower. ``dtypes`` override from function descriptors."""
    dtypes = dtypes or {}
    if name == "vision":
        pos_dt = resolve_dtype(dtypes.get("pixel_position_ids", "int16"), np.int16)
        pix_dt = resolve_dtype(dtypes.get("pixel_values", "float32"), np.float32)
        xs = np.arange(VISION_PATCHES, dtype=pos_dt) % np.array(70, dtype=pos_dt)
        ys = np.arange(VISION_PATCHES, dtype=pos_dt) // np.array(70, dtype=pos_dt)
        pos = np.stack((xs, ys), axis=-1)[None, ...].copy()
        pixels = np.full(
            (1, VISION_PATCHES, VISION_PATCH_DIM), 0.5, dtype=pix_dt
        )
        return {"pixel_values": pixels, "pixel_position_ids": pos}
    if name == "audio":
        feat_dt = resolve_dtype(dtypes.get("input_features", "float16"), np.float16)
        mask_dt = resolve_dtype(dtypes.get("input_features_mask", "int16"), np.int16)
        feat = np.zeros((1, AUDIO_FRAMES, AUDIO_FEAT), dtype=feat_dt)
        mask = np.ones((1, AUDIO_FRAMES), dtype=mask_dt)
        return {"input_features": feat, "input_features_mask": mask}
    if name == "text":
        ids_dt = resolve_dtype(dtypes.get("input_ids", "int32"), np.int32)
        mask_dt = resolve_dtype(dtypes.get("attention_mask", "int32"), np.int32)
        ids = np.arange(1, TEXT_SEQ_LEN + 1, dtype=ids_dt)[None, :]
        mask = np.ones((1, TEXT_SEQ_LEN), dtype=mask_dt)
        return {"input_ids": ids, "attention_mask": mask}
    if name == "text_embeds":
        emb_dt = resolve_dtype(dtypes.get("inputs_embeds", "float16"), np.float16)
        mask_dt = resolve_dtype(dtypes.get("attention_mask", "float16"), np.float16)
        embeds = np.full((1, TEXT_EMBEDS_S, TEXT_HIDDEN), 0.02, dtype=emb_dt)
        mask = np.ones((1, TEXT_EMBEDS_S), dtype=mask_dt)
        mask[:, -8:] = 0
        return {"inputs_embeds": embeds, "attention_mask": mask}
    raise ValueError(f"unknown tower {name!r}")


def output_is_finite(arr: np.ndarray) -> bool:
    if arr.size == 0:
        return False
    if np.issubdtype(arr.dtype, np.floating) or np.issubdtype(arr.dtype, np.complexfloating):
        return bool(np.isfinite(arr).all())
    return True


def classify_device_runs(devices: list[str]) -> dict[str, Any]:
    """Flag mid-graph CPU islands. Begin/end CPU↔device switches are OK."""
    norm = [str(d).upper() for d in devices if d]
    runs: list[dict[str, Any]] = []
    for i, dev in enumerate(norm):
        if runs and runs[-1]["device"] == dev:
            runs[-1]["end"] = i
            runs[-1]["count"] += 1
        else:
            runs.append({"device": dev, "start": i, "end": i, "count": 1})

    first_accel = next((i for i, d in enumerate(norm) if d in _ACCEL), None)
    last_accel = next(
        (i for i, d in reversed(list(enumerate(norm))) if d in _ACCEL), None
    )

    def _kind(start: int) -> str:
        if first_accel is None:
            return "cpu_only"
        if start < first_accel:
            return "begin"
        if last_accel is not None and start > last_accel:
            return "end"
        return "mid"

    mid = []
    for run in runs:
        if run["device"] != "CPU":
            continue
        kind = _kind(run["start"])
        if kind == "mid":
            mid.append({**run, "kind": kind})

    switches = [
        {"from": a["device"], "to": b["device"], "at": b["start"]}
        for a, b in zip(runs, runs[1:])
    ]
    return {
        "device_runs": runs,
        "device_switches": switches,
        "mid_graph_cpu_islands": mid,
        "begin_end_switches_only": len(mid) == 0,
        "cpu_only": first_accel is None and bool(norm),
    }


def placement_from_cache_manifest(blob: bytes) -> str:
    """Coarse forge-style label from a compiled ``manifest.plist``."""
    if b"mps.fullyPlacedOnANE" in blob:
        return "ANE"
    if b"_ANE_region_" in blob:
        return "ANE+GPU"
    return "GPU"


def extract_devices_from_debug(obj: Any, *, limit: int = 4000) -> list[str]:
    """Walk Core AI ``_debug_infos`` JSON for device-like strings."""
    found: list[str] = []
    keys = {
        "device",
        "computeunit",
        "compute_unit",
        "preferreddevice",
        "preferred_compute_device",
        "computeunitkind",
    }

    def _norm(val: str) -> str | None:
        s = val.strip().upper()
        if "NEURAL" in s or s in {"ANE", "NE"} or "NEURALENGINE" in s.replace(" ", ""):
            return "ANE"
        if "GPU" in s or "METAL" in s:
            return "GPU"
        if s == "CPU" or "CPU" in s:
            return "CPU"
        return None

    def walk(node: Any) -> None:
        if len(found) >= limit:
            return
        if isinstance(node, dict):
            for k, v in node.items():
                if str(k).lower().replace("_", "") in {x.replace("_", "") for x in keys}:
                    if isinstance(v, str):
                        n = _norm(v)
                        if n:
                            found.append(n)
                walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(obj)
    return found
