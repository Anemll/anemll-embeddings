#!/usr/bin/env python3
"""Resident Core AI process: load three towers once, run numpy feeds.

The showcase server (torch / transformers) talks to this process over
stdin/stdout JSON. Forwards themselves are timed here so the latency badge
is the tower time, not the pipe.

Re-execs into the Core AI interpreter (``ANEMLL_COREAI_PYTHON``, then the
documented locations in ``api/runtime_paths.py``) when ``coreai.runtime`` is
missing. If none is found it replies with an actionable error and exits.
Logs go to stderr; stdout is JSON lines only.
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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api.runtime_paths import CoreAIPythonNotFound, resolve_coreai_python  # noqa: E402


def _reexec_if_needed() -> None:
    try:
        import coreai.runtime  # noqa: F401
    except ImportError:
        try:
            py = resolve_coreai_python(required=True)
        except CoreAIPythonNotFound as exc:
            _reply({"ok": False, "error": str(exc)})
            raise SystemExit(1) from exc
        if Path(py).resolve() == Path(sys.executable).resolve():
            _reply(
                {
                    "ok": False,
                    "error": f"{py} cannot import coreai.runtime. "
                    "Install coreai-core into that venv or point ANEMLL_COREAI_PYTHON elsewhere.",
                }
            )
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


def _coreai_cache_dir() -> Path:
    raw = os.environ.get("CFFIXED_USER_HOME")
    home = Path(raw).expanduser() if raw else Path.home()
    return home / "Library" / "Caches" / "coreai-cache"


CACHE = _coreai_cache_dir()
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


def function_label_from_manifest(blob: bytes, fn_name: str) -> str | None:
    """Placement of ONE function in a multi-function package's manifest.

    The compiled manifest has an ``Entry Function Attributes`` table keyed
    by ``<function>_<hash>_<n>``. A function placed whole on the ANE carries
    ``mps.fullyPlacedOnANE`` there. ``None`` when the function is not listed.
    """
    import plistlib

    try:
        data = plistlib.loads(blob)
    except Exception:  # noqa: BLE001 - not a plist; caller falls back
        return None
    prefix = f"{fn_name}_"
    found: list[str] = []

    def walk(node) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "Entry Function Attributes" and isinstance(value, dict):
                    for name, attrs in value.items():
                        if str(name).startswith(prefix):
                            found.append(repr(attrs))
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(data)
    if not found:
        return None
    text = " ".join(found)
    if "fullyPlacedOnANE" in text:
        return "fullyOnANE"
    if "_ANE_region_" in text:
        return "ANE+GPU"
    return "GPU"


def function_label_for_package(pkg: Path, fn_name: str, cache: Path | None = None) -> str | None:
    root = CACHE if cache is None else cache
    digest = package_main_hash_hex(pkg)
    if digest is None or not root.is_dir():
        return None
    found = _manifests_named(root, digest)
    if not found:
        return None
    newest = max(found, key=lambda path: path.stat().st_mtime)
    return function_label_from_manifest(newest.read_bytes(), fn_name)


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
        raw = model._debug_infos
        blob = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
        devices = extract_devices_from_debug(json.loads(blob))
    except Exception as exc:  # noqa: BLE001 - debug info is optional; log and continue
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
        except Exception:  # noqa: BLE001, S112 - best-effort dtype probe
            continue
    return out


def _zeros_feed(fn) -> dict[str, np.ndarray]:
    """Warmup feed from the function's own descriptors (extra text towers).

    Zeros everywhere except ``attention_mask`` (ones), so a bucket sees a
    full-length text and a packed tower sees an empty pack.
    """
    feed = {}
    for name in list(fn.desc.input_names):
        desc = fn.desc.input_descriptor(name)
        shape = tuple(int(dim or 1) for dim in desc.shape)
        dtype = np.float16 if "16" in str(desc.dtype) else np.float32
        fill = 1.0 if name == "attention_mask" else 0.0
        feed[name] = np.full(shape, fill, dtype=dtype)
    return feed


async def _load_one(
    model, fn_name: str, feed_np, pkg: Path, started: float, spec: str, multi: bool = False
):
    from coreai.runtime import NDArray

    fn = model.load_function(fn_name)
    if feed_np is None:
        feed_np = _zeros_feed(fn)
    else:
        feed_np = feed_np(_desc_dtypes(fn))
    feed = {name: NDArray(feed_np[name]) for name in fn.desc.input_names}
    t0 = time.perf_counter()
    await fn(inputs=feed)
    warmup_ms = (time.perf_counter() - t0) * 1000.0
    # A second, warm call: the host uses it to choose between text towers.
    t0 = time.perf_counter()
    await fn(inputs=feed)
    warm_ms = (time.perf_counter() - t0) * 1000.0
    if multi:
        # One manifest covers every function; read this function's entry.
        # Not listed means unknown, never a borrowed ANE label.
        placement = function_label_for_package(pkg, fn_name) or "unknown"
    else:
        placement = _placement(model, started, pkg)
    row = {
        "loaded": True,
        "warmup_ms": warmup_ms,
        "warm_ms": warm_ms,
        "placement": placement,
        "simulated": False,
        "entry": fn_name,
        "specialization": spec,
        "inputs": {
            name: list(fn.desc.input_descriptor(name).shape) for name in fn.desc.input_names
        },
    }
    return fn, row


async def _load_model(pkg: Path, opts):
    from coreai.runtime import AIModel

    if opts is not None:
        return await AIModel.load(pkg, specialization_options=opts)
    return await AIModel.load(pkg)


async def _load_all(
    packages: dict[str, str], compute: str, text_packages: list[str] | None = None
) -> dict:
    opts, spec = _host_specialization(compute)
    towers: dict[str, dict] = {}
    loaded = {}
    for key in TOWER_ORDER:
        if key not in packages:
            continue
        pkg = Path(packages[key])
        entry = TOWER_ENTRIES["text_embeds" if key == "text" else key]
        started = time.time()
        if not pkg.exists():
            raise FileNotFoundError(f"missing {key} package: {pkg}")
        _log(f"load {key} {pkg} spec={spec}")
        t_load = time.perf_counter()
        model = await _load_model(pkg, opts)
        names = list(model.function_names)
        fn_name = entry if entry in names else names[0]
        load_ms = (time.perf_counter() - t_load) * 1000.0
        dummy_key = "text_embeds" if key == "text" else key
        fn, row = await _load_one(
            model,
            fn_name,
            lambda dtypes, k=dummy_key: dummy_numpy_inputs(k, dtypes),
            pkg,
            started,
            spec,
        )
        row["load_ms"] = load_ms
        loaded[key] = fn
        towers[fn_name] = row
        _log(
            f"ready {fn_name} load_ms={load_ms:.1f} warmup_ms={row['warmup_ms']:.1f} "
            f"placement={row['placement']}"
        )
    # Optional text buckets / packed towers: every function in each package,
    # keyed by function name. A failure here leaves the main towers working.
    for raw in text_packages or []:
        pkg = Path(raw)
        started = time.time()
        try:
            _log(f"load text extras {pkg} spec={spec}")
            t_load = time.perf_counter()
            model = await _load_model(pkg, opts)
            load_ms = (time.perf_counter() - t_load) * 1000.0
            names = list(model.function_names)
            for fn_name in names:
                if fn_name in towers:
                    # Same function already loaded from its own package
                    # (a combined package may also carry text_embeds_s320).
                    continue
                fn, row = await _load_one(
                    model, fn_name, None, pkg, started, spec, multi=len(names) > 1
                )
                row["load_ms"] = load_ms
                row["extra"] = True
                loaded[fn_name] = fn
                towers[fn_name] = row
                _log(
                    f"ready {fn_name} warmup_ms={row['warmup_ms']:.1f} "
                    f"placement={row['placement']}"
                )
        except Exception as exc:  # noqa: BLE001 - extras are optional
            _log(f"text extras {pkg} failed: {type(exc).__name__}: {exc}")
            towers[pkg.name] = {"loaded": False, "extra": True, "error": str(exc)}
    places = [row["placement"] for row in towers.values() if row.get("loaded")]
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
                report = await _load_all(
                    req["packages"],
                    req.get("compute") or "ane",
                    req.get("text_packages") or [],
                )
                state["functions"] = report.pop("functions")
                _reply({"ok": True, **{k: v for k, v in report.items()}})
            elif cmd == "forward":
                fns = state.get("functions") or {}
                tower = req["tower"]
                if tower not in fns:
                    raise KeyError(f"tower {tower!r} is not loaded")
                with np.load(req["npz"]) as data:
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
        except Exception as exc:  # noqa: BLE001 - the worker must reply, not die
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
