#!/usr/bin/env python3
"""Smoke-load Core AI tower ``.aimodel`` packages (no re-export).

Requires ``coreai`` runtime. Re-execs under ``ANEMLL_COREAI_PYTHON`` when the
embeddings venv does not have it. Does **not** call ``forge.py convert`` and
does not touch the FLOAT32 Core ML tree.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE = Path.home() / "Library/Caches/coreai-cache"


def _coreai_python(required: bool = True) -> Path | None:
    """Core AI interpreter: ``ANEMLL_COREAI_PYTHON``, then documented locations.

    Exits with setup instructions when none exists (see ``api/runtime_paths.py``).
    """
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from api.runtime_paths import CoreAIPythonNotFound, resolve_coreai_python

    try:
        return resolve_coreai_python(required=required)
    except CoreAIPythonNotFound as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


def _have_coreai() -> bool:
    try:
        import coreai.runtime  # noqa: F401
    except ImportError:
        return False
    return True


def _reexec_if_needed() -> None:
    if _have_coreai():
        return
    py = _coreai_python()
    if not py.is_file():
        raise SystemExit(
            "ERROR: coreai runtime missing and ANEMLL_COREAI_PYTHON not found at "
            f"{py}. Point it at a venv with coreai-core (see README, Core AI runtime)."
        )
    print(f"re-exec {py} (coreai runtime)")
    os.execv(str(py), [str(py), *sys.argv])


# Re-exec before importing ``model.*`` — Core AI venv has no sentence-transformers,
# so load helpers from this directory instead of the ``model`` package.
_reexec_if_needed()
sys.path.insert(0, str(REPO_ROOT / "model"))
from coreai_smoke import (  # noqa: E402
    TOWER_ENTRIES,
    TOWER_PACKAGES,
    classify_device_runs,
    dummy_numpy_inputs,
    extract_devices_from_debug,
    output_is_finite,
    placement_from_cache_manifest,
    tower_smoke_io,
)
from export_utils import artifacts_root, git_sha, utc_now, write_json  # noqa: E402


def _io_from_export_json(path: Path, tower: str) -> dict | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    for row in data.get("towers") or []:
        if row.get("tower") == tower and isinstance(row.get("io"), dict):
            return row["io"]
    return None


def _expected_out_shape(tower: str, export_io: dict | None) -> list[int]:
    spec = export_io or tower_smoke_io(tower)
    outs = spec.get("outputs") or {}
    if not outs:
        return []
    return list(next(iter(outs.values())))


def _desc_dtypes(fn) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in list(fn.desc.input_names):
        try:
            out[name] = str(fn.desc.input_descriptor(name).dtype)
        except Exception:  # noqa: BLE001, S112 - best-effort dtype probe
            continue
    return out


def _cache_placement_since(since: float) -> dict:
    mans = [
        p
        for p in CACHE.glob("*/*/*/*/model.aimodelx/**/manifest.plist")
        if p.stat().st_mtime >= since
    ]
    if not mans:
        return {"source": "cache_manifest", "label": "cached", "path": None}
    newest = max(mans, key=lambda p: p.stat().st_mtime)
    label = placement_from_cache_manifest(newest.read_bytes())
    return {"source": "cache_manifest", "label": label, "path": str(newest)}


async def _smoke_one(
    tower: str, pkg: Path, export_io: dict | None, *, compute: str = "cpu"
) -> dict:
    from coreai.runtime import (
        AIModel,
        ComputeUnitKind,
        NDArray,
        SpecializationOptions,
        _AIModelAsset,
    )

    started = time.time()
    report: dict = {
        "tower": tower,
        "package": str(pkg),
        "exists": pkg.is_dir(),
        "pass": False,
        "_compute": compute,
    }
    if not pkg.is_dir():
        report["error"] = "package missing"
        return report

    try:
        report["asset_valid"] = bool(_AIModelAsset.is_valid(pkg))
        asset = _AIModelAsset(pkg)
        summary = asset.summary(include_statistics=False)
        if summary is not None:
            report["function_names"] = list(summary.function_names)
            report["compute_types"] = list(summary.compute_types)
    except Exception as exc:  # noqa: BLE001 - asset summary is optional
        report["asset_error"] = f"{type(exc).__name__}: {exc}"

    opts = None
    spec_note = "default_load"
    compute = str(report.get("_compute") or "cpu")
    if SpecializationOptions.is_supported():
        try:
            if compute == "ane":
                opts = SpecializationOptions.from_preferred_compute_unit_kind(
                    ComputeUnitKind.neural_engine()
                )
                spec_note = "preferred_ane"
            elif compute == "cpu":
                opts = SpecializationOptions.cpu_only()
                spec_note = "cpu_only"
        except Exception as exc:  # noqa: BLE001 - fall back to default specialization
            spec_note = f"options_failed:{type(exc).__name__}"
            opts = None
    report["specialization"] = spec_note

    try:
        if opts is not None:
            model = await AIModel.load(pkg, specialization_options=opts)
        else:
            model = await AIModel.load(pkg)
    except Exception as exc:  # noqa: BLE001 - record the load failure in the report
        report["error"] = f"{type(exc).__name__}: {exc}"
        return report

    names = list(model.function_names)
    report["loaded_functions"] = names
    want = TOWER_ENTRIES[tower]
    fn_name = want if want in names else (names[0] if names else want)
    report["entry"] = fn_name
    fn = model.load_function(fn_name)
    dtypes = _desc_dtypes(fn)
    report["input_dtypes"] = dtypes
    report["input_names"] = list(fn.desc.input_names)
    report["output_names"] = list(fn.desc.output_names)

    feed_np = dummy_numpy_inputs(tower, dtypes)
    missing = [n for n in fn.desc.input_names if n not in feed_np]
    if missing:
        report["error"] = f"dummy feed missing {missing}"
        return report
    feed = {k: NDArray(feed_np[k]) for k in fn.desc.input_names}
    t_fwd = time.perf_counter()
    outs = await fn(inputs=feed)
    report["forward_ms"] = round((time.perf_counter() - t_fwd) * 1e3, 3)

    if not outs:
        report["error"] = "empty outputs"
        return report
    first_name = next(iter(outs))
    arr = outs[first_name].numpy()
    expect = _expected_out_shape(tower, export_io)
    finite = output_is_finite(arr)
    shape_ok = not expect or list(arr.shape) == expect
    report["output_name"] = first_name
    report["output_shape"] = list(arr.shape)
    report["expected_shape"] = expect
    report["dtype"] = str(arr.dtype)
    report["finite"] = finite
    report["shape_ok"] = shape_ok
    if np.issubdtype(arr.dtype, np.floating) and arr.size:
        flat = arr.astype(np.float32, copy=False).reshape(-1)
        report["abs_max"] = float(np.nanmax(np.abs(flat)))
        report["l2"] = float(np.linalg.norm(flat))

    placement = _cache_placement_since(started)
    devices: list[str] = []
    try:
        raw = model._debug_infos
        dbg = json.loads(raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw)
        devices = extract_devices_from_debug(dbg)
        report["debug_device_count"] = len(devices)
    except Exception as exc:  # noqa: BLE001 - debug info is optional
        report["debug_infos"] = f"{type(exc).__name__}: {exc}"

    classified = classify_device_runs(devices)
    mid = list(classified["mid_graph_cpu_islands"])
    report["placement"] = {
        **placement,
        "mid_graph_cpu_islands": mid,
        "begin_end_switches_only": bool(classified["begin_end_switches_only"]),
        "cpu_only": bool(classified["cpu_only"]),
        "device_run_count": len(classified["device_runs"]),
    }
    if mid:
        report["placement"]["flag"] = "mid_graph_cpu_island"
    elif compute == "cpu":
        report["placement"]["flag"] = None
        report["placement"]["note"] = (
            "CPU-only smoke (finite/shape). Integer I/O exported as si16."
        )
    elif placement["label"] == "ANE+GPU" and not devices:
        report["placement"]["flag"] = "mixed_ane_gpu_unknown_midgraph"
    elif classified["cpu_only"]:
        report["placement"]["flag"] = "cpu_only"
    else:
        report["placement"]["flag"] = None

    report["pass"] = bool(finite and shape_ok)
    if not report["pass"]:
        report["error"] = (
            f"finite={finite} shape_ok={shape_ok} "
            f"got={list(arr.shape)} expect={expect}"
        )
    return report


async def _run(towers: list[str], out_dir: Path, *, compute: str) -> dict:
    export_json = out_dir / "towers.export.json"
    rows = []
    fail = False
    for name in towers:
        pkg = out_dir / TOWER_PACKAGES[name]
        export_io = _io_from_export_json(export_json, name)
        print(f"smoke {name} compute={compute} {pkg}")
        try:
            row = await _smoke_one(name, pkg, export_io, compute=compute)
        except Exception as exc:  # noqa: BLE001 - record per-tower failure and continue
            row = {
                "tower": name,
                "package": str(pkg),
                "pass": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        rows.append(row)
        status = "PASS" if row.get("pass") else "FAIL"
        place = (row.get("placement") or {}).get("label")
        flag = (row.get("placement") or {}).get("flag")
        print(
            f"  {status} shape={row.get('output_shape')} finite={row.get('finite')} "
            f"place={place} flag={flag}"
        )
        if not row.get("pass"):
            fail = True
            if row.get("error"):
                print(f"  error: {row['error']}")
    return {
        "created_at_utc": utc_now(),
        "git_sha": git_sha(REPO_ROOT),
        "target": "coreai_smoke",
        "out_dir": str(out_dir),
        "towers": rows,
        "all_pass": not fail,
        "compute": compute,
        "notes": [
            "Load + one forward only.",
            "Default compute=cpu. Integer I/O is si16 (ANE-legal).",
            "Begin/end CPU↔device switches are OK. Mid-graph CPU islands are flagged.",
            "Not forge.py convert. FLOAT32 Core ML text tree untouched.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tower",
        action="append",
        choices=tuple(TOWER_PACKAGES),
        help="Repeatable. Default: vision, text, audio. text_embeds is the caption / mixed-media package.",
    )
    parser.add_argument("--artifacts", type=Path, default=artifacts_root())
    parser.add_argument(
        "--compute",
        choices=("cpu", "ane", "default"),
        default="cpu",
        help="Specialization. Default cpu. Use ane for the Neural Engine (pair with --isolated if one tower abort should not kill the rest).",
    )
    parser.add_argument(
        "--isolated",
        action="store_true",
        help="Run each --tower in a child process so ANE abort cannot kill the rest.",
    )
    args = parser.parse_args()
    towers = args.tower or ["vision", "text", "audio"]
    out_dir = Path(args.artifacts) / "coreai"
    print(f"out_dir={out_dir} towers={towers} compute={args.compute} isolated={args.isolated}")
    print(f"ANEMLL_COREAI_PYTHON={_coreai_python(required=False)} coreai_here={_have_coreai()}")
    if args.isolated:
        import subprocess

        rows = []
        fail = False
        for name in towers:
            log = Path(f"/tmp/coreai-ane-{name}.log")
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--compute",
                args.compute,
                "--tower",
                name,
                "--artifacts",
                str(args.artifacts),
            ]
            print(f"isolated {name}: {' '.join(cmd)}")
            with log.open("w") as fh:
                rc = subprocess.call(cmd, stdout=fh, stderr=fh)
            print(f"  child_rc={rc} log={log}")
            child_row = None
            smoke_dest = out_dir / "towers.smoke.json"
            if smoke_dest.is_file() and rc == 0:
                try:
                    child_meta = json.loads(smoke_dest.read_text())
                    child_row = next(
                        (r for r in (child_meta.get("towers") or []) if r.get("tower") == name),
                        None,
                    )
                except json.JSONDecodeError:
                    child_row = None
            if child_row is not None:
                child_row["child_rc"] = rc
                rows.append(child_row)
                if not child_row.get("pass"):
                    fail = True
            else:
                tail = ""
                if log.is_file():
                    tail = log.read_text(errors="replace")[-4000:]
                fail = True
                rows.append(
                    {
                        "tower": name,
                        "pass": False,
                        "child_rc": rc,
                        "error": f"isolated child exit {rc}",
                        "log_tail": tail,
                    }
                )
        dest = out_dir / "towers.ane.json"
        write_json(
            dest,
            {
                "created_at_utc": utc_now(),
                "git_sha": git_sha(REPO_ROOT),
                "target": "coreai_ane_isolated",
                "compute": args.compute,
                "towers": rows,
                "all_pass": not fail,
            },
        )
        print(f"wrote {dest}")
        return 0 if not fail else 1
    meta = asyncio.run(_run(towers, out_dir, compute=args.compute))
    dest = out_dir / "towers.smoke.json"
    write_json(dest, meta)
    print(f"wrote {dest}")
    return 0 if meta.get("all_pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
