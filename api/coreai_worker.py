#!/usr/bin/env python3
"""Resident Core AI process: load three towers once, run numpy feeds.

The showcase server (torch / transformers) talks to this process over
stdin/stdout JSON. Forwards themselves are timed here so the latency badge
is the tower time, not the pipe.

Re-execs into ``ANEMLL_COREAI_PYTHON`` when ``coreai.runtime`` is missing,
same as ``model/_coreai_run_npy.py``. Logs go to stderr; stdout is JSON
lines only.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COREAI_PY = Path("/Users/anemll/anemll-forge/coreai/.venv/bin/python")


def _coreai_python() -> Path:
    raw = os.environ.get("ANEMLL_COREAI_PYTHON")
    return Path(raw) if raw else DEFAULT_COREAI_PY


def _reexec_if_needed() -> None:
    try:
        import coreai.runtime  # noqa: F401
    except ImportError:
        py = _coreai_python()
        if not py.is_file():
            _reply({"ok": False, "error": f"coreai python missing at {py}"})
            raise SystemExit(1)
        os.execv(str(py), [str(py), *sys.argv])


def _reply(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def _log(msg: str) -> None:
    sys.stderr.write(msg.rstrip() + "\n")
    sys.stderr.flush()


# Import placement helpers without ``model/__init__`` (that package pulls
# sentence-transformers, which the Core AI venv does not have).
sys.path.insert(0, str(REPO_ROOT / "model"))
from coreai_smoke import (  # noqa: E402
    TOWER_ENTRIES,
    dummy_numpy_inputs,
    extract_devices_from_debug,
    placement_from_cache_manifest,
)

CACHE = Path.home() / "Library/Caches/coreai-cache"
TOWER_ORDER = ("vision", "audio", "text")


def _host_specialization(compute: str):
    """Use the parity runner's loader switch.

    ``specialization_options`` imports ``coreai.runtime``, so call this only
    after the worker has re-execed into that interpreter. Unset
    ``ANEMLL_COREAI_COMPUTE`` still means CPU inside the helper; the server
    sets it to ``ane`` before load.
    """
    key = compute.strip().lower()
    if key not in {"ane", "cpu"}:
        raise ValueError(f"compute must be ane or cpu, got {compute!r}")
    os.environ["ANEMLL_COREAI_COMPUTE"] = key
    sys.path.insert(0, str(REPO_ROOT / "model"))
    from _coreai_run_npy import specialization_options

    return specialization_options(), key


def package_main_hash_hex(pkg: Path) -> str | None:
    """Hex folder name Core AI uses for this package under ``coreai-cache``.

    A ``.aimodel`` bundle stores the content hash in ``main.hash``. The cache
    directory is that digest in hex, so a warm start can find this package's
    own manifest instead of whatever compile finished most recently.
    """
    path = Path(pkg) / "main.hash"
    if not path.is_file():
        return None
    raw = path.read_bytes().strip()
    if not raw:
        return None
    try:
        text = raw.decode("ascii").strip().lower()
    except UnicodeDecodeError:
        text = ""
    if text and all(ch in "0123456789abcdef" for ch in text) and len(text) >= 8:
        return text
    return raw.hex()


def _label_from_blob(blob: bytes) -> str:
    label = placement_from_cache_manifest(blob)
    if label == "ANE":
        return "fullyOnANE"
    return label


def _manifests_named(cache: Path, digest: str) -> list[Path]:
    """Manifests whose cache folder is this package hash."""
    patterns = (
        f"*/*/*/{digest}/model.aimodelx/**/manifest.plist",
        f"**/{digest}/model.aimodelx/**/manifest.plist",
        f"**/{digest}/**/manifest.plist",
    )
    for pattern in patterns:
        found = list(cache.glob(pattern))
        if found:
            return found
    return []


def manifest_label_for_package(pkg: Path, cache: Path | None = None) -> str | None:
    """Placement from this package's own cached compile, if one exists."""
    root = CACHE if cache is None else cache
    digest = package_main_hash_hex(pkg)
    if digest is None or not root.is_dir():
        return None
    found = _manifests_named(root, digest)
    if not found:
        return None
    newest = max(found, key=lambda path: path.stat().st_mtime)
    return _label_from_blob(newest.read_bytes())


def _manifest_label(since: float, cache: Path | None = None) -> str | None:
    """Newest manifest written around this load. Fallback for a missing hash."""
    root = CACHE if cache is None else cache
    if not root.is_dir():
        return None
    mans = [
        path
        for path in root.glob("*/*/*/*/model.aimodelx/**/manifest.plist")
        if path.stat().st_mtime >= since - 1.0
    ]
    if not mans:
        return None
    newest = max(mans, key=lambda path: path.stat().st_mtime)
    return _label_from_blob(newest.read_bytes())


def _placement(model, started: float, package: Path) -> str:
    for attr in ("placement", "compute_placement"):
        val = getattr(model, attr, None)
        if val is None:
            continue
        text = str(val)
        if "fullyOnANE" in text or "fullyPlacedOnANE" in text:
            return "fullyOnANE"
    devices: list[str] = []
    try:
        raw = model._debug_infos  # noqa: SLF001
        blob = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
        devices = extract_devices_from_debug(json.loads(blob))
    except Exception as exc:
        _log(f"debug_infos: {type(exc).__name__}: {exc}")
    # Warm starts do not rewrite the cache, so a time window around ``started``
    # is empty and the globally newest manifest can belong to another compile.
    label = manifest_label_for_package(package)
    if label is None:
        label = _manifest_label(started)
    if label == "fullyOnANE":
        return "fullyOnANE"
    if devices and all(device == "ANE" for device in devices):
        return "fullyOnANE"
    if label:
        return label
    if devices:
        return "+".join(sorted(set(devices)))
    return "unknown"


def _desc_dtypes(fn) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in list(fn.desc.input_names):
        try:
            out[name] = str(fn.desc.input_descriptor(name).dtype)
        except Exception:
            continue
    return out


async def _load_all(packages: dict[str, str], compute: str) -> dict:
    from coreai.runtime import AIModel, NDArray

    opts, spec = _host_specialization(compute)
    towers: dict[str, dict] = {}
    loaded = {}
    for key in TOWER_ORDER:
        pkg = Path(packages[key])
        entry = TOWER_ENTRIES[key if key != "text" else "text_embeds"]
        if key == "text":
            entry = TOWER_ENTRIES["text_embeds"]
        started = time.time()
        if not pkg.exists():
            raise FileNotFoundError(f"missing {key} package: {pkg}")
        _log(f"load {key} {pkg} spec={spec}")
        t_load = time.perf_counter()
        if opts is not None:
            model = await AIModel.load(pkg, specialization_options=opts)
        else:
            model = await AIModel.load(pkg)
        names = list(model.function_names)
        fn_name = entry if entry in names else names[0]
        fn = model.load_function(fn_name)
        load_ms = (time.perf_counter() - t_load) * 1000.0
        dtypes = _desc_dtypes(fn)
        dummy_key = "text_embeds" if key == "text" else key
        feed_np = dummy_numpy_inputs(dummy_key, dtypes)
        feed = {name: NDArray(feed_np[name]) for name in fn.desc.input_names}
        t0 = time.perf_counter()
        await fn(inputs=feed)
        warmup_ms = (time.perf_counter() - t0) * 1000.0
        place = _placement(model, started, pkg)
        loaded[key] = fn
        towers[fn_name] = {
            "loaded": True,
            "load_ms": load_ms,
            "warmup_ms": warmup_ms,
            "placement": place,
            "simulated": False,
            "entry": fn_name,
            "specialization": spec,
        }
        _log(
            f"ready {fn_name} load_ms={load_ms:.1f} warmup_ms={warmup_ms:.1f} "
            f"placement={place}"
        )
    places = [row["placement"] for row in towers.values()]
    overall = places[0] if places and all(item == places[0] for item in places) else "mixed"
    return {"functions": loaded, "towers": towers, "placement": overall, "specialization": spec}


async def _forward(fn, feed: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
    from coreai.runtime import NDArray

    arrays = {name: NDArray(np.ascontiguousarray(feed[name])) for name in fn.desc.input_names}
    missing = [name for name in fn.desc.input_names if name not in feed]
    if missing:
        raise KeyError(f"feed missing {missing}")
    t0 = time.perf_counter()
    outs = await fn(inputs=arrays)
    elapsed = (time.perf_counter() - t0) * 1000.0
    first = next(iter(outs))
    return outs[first].numpy(), elapsed


async def _serve() -> None:
    state: dict = {}
    loop = asyncio.get_running_loop()
    while True:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            cmd = req.get("cmd")
            if cmd == "load":
                report = await _load_all(req["packages"], req.get("compute") or "ane")
                state["functions"] = report.pop("functions")
                _reply({"ok": True, **{k: v for k, v in report.items()}})
            elif cmd == "forward":
                fns = state.get("functions") or {}
                tower = req["tower"]
                if tower not in fns:
                    raise KeyError(f"tower {tower!r} is not loaded")
                data = np.load(req["npz"])
                feed = {key: data[key] for key in data.files}
                arr, elapsed = await _forward(fns[tower], feed)
                out = Path(req["out"])
                out.parent.mkdir(parents=True, exist_ok=True)
                np.save(out, arr)
                _reply(
                    {
                        "ok": True,
                        "latency_ms": elapsed,
                        "shape": list(arr.shape),
                        "dtype": str(arr.dtype),
                    }
                )
            elif cmd == "shutdown":
                _reply({"ok": True})
                break
            else:
                _reply({"ok": False, "error": f"unknown cmd {cmd!r}"})
        except Exception as exc:
            _log(f"worker error: {type(exc).__name__}: {exc}")
            _reply({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def main() -> int:
    _reexec_if_needed()
    try:
        asyncio.run(_serve())
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
