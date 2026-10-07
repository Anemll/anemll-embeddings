#!/usr/bin/env python3
"""Warm + p50/p90 timing for Core AI ``text_embeds_s320`` (no re-export).

Same style as ``ane_smoke.py`` (warmup=2, iters=5). Times CPU, preferred-ANE,
and preferred-GPU (Core AI analog of Core ML ``CPU_AND_GPU``). Fixed dummy
S=320 f16 embeds + f16 mask. Does not touch the FLOAT32 Core ML tree.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COREAI_PY = Path("/Users/anemll/anemll-forge/coreai/.venv/bin/python")
CACHE = Path.home() / "Library/Caches/coreai-cache"


def _coreai_python() -> Path:
    raw = os.environ.get("ANEMLL_COREAI_PYTHON")
    return Path(raw) if raw else DEFAULT_COREAI_PY


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
        raise SystemExit(f"ERROR: coreai python missing at {py}")
    print(f"re-exec {py} (coreai runtime)")
    os.execv(str(py), [str(py), *sys.argv])


_reexec_if_needed()
sys.path.insert(0, str(REPO_ROOT / "src"))
from coreai_smoke import (  # noqa: E402
    TOWER_ENTRIES,
    dummy_numpy_inputs,
    extract_devices_from_debug,
    output_is_finite,
    parse_mpsgraph_regions,
    percentile_ms,
    placement_from_cache_manifest,
)
from export_utils import artifacts_root, git_sha, utc_now, write_json  # noqa: E402


def _opts(compute: str):
    from coreai.runtime import ComputeUnitKind, SpecializationOptions

    if not SpecializationOptions.is_supported():
        return None, "unsupported"
    if compute == "cpu":
        return SpecializationOptions.cpu_only(), "cpu_only"
    if compute == "ane":
        return (
            SpecializationOptions.from_preferred_compute_unit_kind(
                ComputeUnitKind.neural_engine()
            ),
            "preferred_ane",
        )
    if compute == "gpu":
        return (
            SpecializationOptions.from_preferred_compute_unit_kind(
                ComputeUnitKind.gpu()
            ),
            "preferred_gpu",
        )
    raise ValueError(compute)


def _cache_since(since: float) -> dict:
    mans = [
        p
        for p in CACHE.glob("*/*/*/*/model.aimodelx/**/manifest.plist")
        if p.stat().st_mtime >= since
    ]
    if not mans:
        return {"source": "cache_manifest", "label": "cached", "path": None}
    newest = max(mans, key=lambda p: p.stat().st_mtime)
    spec = newest.parent / "specialized_model_0.mpsgraph"
    row = {
        "source": "cache_manifest",
        "label": placement_from_cache_manifest(newest.read_bytes()),
        "path": str(newest),
        "mpsgraph": str(spec) if spec.is_file() else None,
    }
    if spec.is_file():
        row["regions"] = parse_mpsgraph_regions(spec.read_bytes())
    return row


def _desc_dtypes(fn) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in list(fn.desc.input_names):
        try:
            out[name] = str(fn.desc.input_descriptor(name).dtype)
        except Exception:
            continue
    return out


async def _time_one(
    pkg: Path, *, compute: str, warmup: int, iters: int
) -> dict:
    from coreai.runtime import AIModel, NDArray

    started = time.time()
    report: dict = {"compute": compute, "pass": False}
    opts, spec_note = _opts(compute)
    report["specialization"] = spec_note
    if opts is not None:
        report["allowed"] = [str(k) for k in opts.allowed_compute_unit_kinds]
        report["preferred"] = (
            str(opts.preferred_compute_unit_kind)
            if opts.preferred_compute_unit_kind is not None
            else None
        )
    try:
        if opts is not None:
            model = await AIModel.load(pkg, specialization_options=opts)
        else:
            model = await AIModel.load(pkg)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        return report

    names = list(model.function_names)
    want = TOWER_ENTRIES["text_embeds"]
    fn_name = want if want in names else names[0]
    fn = model.load_function(fn_name)
    dtypes = _desc_dtypes(fn)
    report["input_dtypes"] = dtypes
    feed_np = dummy_numpy_inputs("text_embeds", dtypes)
    feed = {k: NDArray(feed_np[k]) for k in fn.desc.input_names}

    for _ in range(warmup):
        outs = await fn(inputs=feed)
    times: list[float] = []
    last = None
    for _ in range(iters):
        t0 = time.perf_counter()
        outs = await fn(inputs=feed)
        times.append(time.perf_counter() - t0)
        last = outs
    arr = next(iter(last.values())).numpy() if last else None
    finite = bool(arr is not None and output_is_finite(arr))
    report.update(
        {
            "entry": fn_name,
            "warmup": warmup,
            "iters": iters,
            "p50_ms": round(percentile_ms(times, 50.0), 3),
            "p90_ms": round(percentile_ms(times, 90.0), 3),
            "min_ms": round(min(times) * 1000.0, 3) if times else None,
            "max_ms": round(max(times) * 1000.0, 3) if times else None,
            "output_shape": list(arr.shape) if arr is not None else None,
            "finite": finite,
            "pass": finite,
        }
    )
    try:
        raw = model._debug_infos  # noqa: SLF001
        dbg = json.loads(raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw)
        report["debug_devices"] = extract_devices_from_debug(dbg)
    except Exception as exc:
        report["debug_infos"] = f"{type(exc).__name__}: {exc}"
    report["placement"] = _cache_since(started)
    return report


async def _run(pkg: Path, *, warmup: int, iters: int) -> dict:
    rows = {}
    for compute in ("cpu", "ane", "gpu"):
        print(f"time {compute} warmup={warmup} iters={iters}")
        row = await _time_one(pkg, compute=compute, warmup=warmup, iters=iters)
        rows[compute] = row
        print(
            f"  {compute} pass={row.get('pass')} "
            f"p50={row.get('p50_ms')} p90={row.get('p90_ms')} "
            f"place={(row.get('placement') or {}).get('label')}"
        )
        if row.get("error"):
            print(f"  error: {row['error']}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=artifacts_root())
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    args = parser.parse_args()
    out_dir = Path(args.artifacts) / "coreai"
    pkg = out_dir / "text_embeds_s320.aimodel"
    print(f"package={pkg} warmup={args.warmup} iters={args.iters}")
    if not pkg.is_dir():
        print(f"ERROR: missing {pkg}")
        return 1
    rows = asyncio.run(_run(pkg, warmup=args.warmup, iters=args.iters))
    ane_place = (rows.get("ane") or {}).get("placement") or {}
    regions = ane_place.get("regions") or {}
    gpu_top = regions.get("gpu_top_regions") or []
    report = {
        "created_at_utc": utc_now(),
        "git_sha": git_sha(REPO_ROOT),
        "target": "coreai_text_embeds_time",
        "package": str(pkg),
        "seq_len": 320,
        "warmup": args.warmup,
        "iters": args.iters,
        "units": rows,
        "gpu_island": {
            "named": gpu_top[0] if gpu_top else None,
            "regions": regions,
            "note": (
                "Dump names GPU_region_0 (mpsx.parallelGPURegion) plus nested "
                "subs, not a single math op. Nearby affine maps include "
                "[1,320,512] embeds and [1,768] output — I/O/layout/cast, "
                "not mid-graph SDPA. _debug_infos has no device-run list, so "
                "begin/end vs mid stays unknown."
            ),
        },
        "notes": [
            "Warm + p50/p90 match ane_smoke.py (default warmup=2, iters=5).",
            "preferred_gpu is Core AI analog of Core ML CPU_AND_GPU.",
            "Fixed dummy S=320 f16 embeds + f16 mask. No re-export.",
            "FLOAT32 Core ML text tree untouched.",
        ],
        "all_pass": all(bool((rows.get(k) or {}).get("pass")) for k in ("cpu", "ane", "gpu")),
    }
    dest = out_dir / "text_embeds_s320.time.json"
    write_json(dest, report)
    print(f"wrote {dest} all_pass={report['all_pass']}")
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
