#!/usr/bin/env python3
"""T7: ANE smoke + placement on the same T5 ``.mlpackage``.

Loads the package with CPU_ONLY and CPU_AND_NE, scores T1 fixtures on the
ANE-capable path, dumps an MLComputePlan, and times both units after warmup.

Fails if the compute plan has no ANE-preferred ops (silent CPU fallback) or
if ANE embeddings are non-finite / miss PLAN cosine/rel-L2 gates.
Does **not** treat T6 CPU parity as ANE success.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

try:
    import coremltools as ct
    from coremltools.models.compute_plan import MLComputePlan
except ImportError:
    ct = None
    MLComputePlan = None

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.ane_placement import placement_verdict, summarize_compute_plan  # noqa: E402
from src.embed_wrapper import tokenize_with_st_prompt  # noqa: E402
from src.export_utils import (  # noqa: E402
    DEFAULT_SEQ_LEN,
    artifact_subdir,
    artifacts_root,
    git_sha,
    host_versions,
    package_stem,
    pad_to_seq_len,
    utc_now,
    write_json,
)
from src.load_text_model import load_sentence_transformer  # noqa: E402
from src.parity_metrics import (  # noqa: E402
    COSINE_GATE,
    REL_L2_GATE,
    cosine,
    finite_counts,
    l2_norm,
    rel_l2,
)

DEFAULT_PROMPTS = REPO_ROOT / "tests" / "fixtures" / "prompts.json"
DEFAULT_NPY = REPO_ROOT / "tests" / "fixtures" / "embeddings.npy"


def _soc_note() -> dict[str, str]:
    brand = ""
    model = ""
    try:
        brand = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        brand = platform.processor()
    try:
        model = subprocess.check_output(["sysctl", "-n", "hw.model"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        model = ""
    return {
        "brand": brand,
        "hw_model": model,
        "note": (
            "Forge bonded-compile policy is M5+/H17+ (Core AI). "
            "This script leaves MPSGRAPH_ANE_BONDED_COMPILE_MODE unset "
            "and uses public Core ML compute plans."
        ),
    }


def _predict_one(mlmodel, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    pred = mlmodel.predict(
        {"input_ids": input_ids, "attention_mask": attention_mask}
    )
    if "embedding" in pred:
        vec = np.asarray(pred["embedding"])
    else:
        vec = np.asarray(next(iter(pred.values())))
    return vec.astype(np.float32).reshape(-1)


def _load_model(pkg_path: Path, units):
    return ct.models.MLModel(str(pkg_path), compute_units=units)


def _encode_batches(st_model, prompts, seq_len: int, pad_id: int):
    batches = []
    truncated = 0
    for p in prompts:
        raw = tokenize_with_st_prompt(
            st_model, p["text"], p.get("prompt_name"), device="cpu"
        )
        raw_len = int(raw["attention_mask"].sum().item())
        ids, mask = pad_to_seq_len(
            raw["input_ids"], raw["attention_mask"], seq_len, pad_token_id=pad_id
        )
        if raw_len > seq_len:
            truncated += 1
        batches.append(
            {
                "id": p["id"],
                "raw_tokens": raw_len,
                "truncated": raw_len > seq_len,
                "input_ids": ids.numpy().astype(np.int32),
                "attention_mask": mask.numpy().astype(np.int32),
            }
        )
    return batches, truncated


def _run_fixtures(mlmodel, batches, ref: np.ndarray) -> list[dict]:
    rows = []
    for i, b in enumerate(batches):
        vec = _predict_one(mlmodel, b["input_ids"], b["attention_mask"])
        rows.append(
            {
                "id": b["id"],
                "raw_tokens": b["raw_tokens"],
                "truncated": b["truncated"],
                "vector": vec,
                "cosine": cosine(vec, ref[i]),
                "rel_l2": rel_l2(vec, ref[i]),
                "l2_pred": l2_norm(vec),
                "l2_ref": l2_norm(ref[i]),
                "finite": finite_counts(vec),
            }
        )
    return rows


def _median_ms(times: list[float]) -> float:
    if not times:
        return float("nan")
    s = sorted(times)
    mid = len(s) // 2
    if len(s) % 2:
        return float(s[mid] * 1000.0)
    return float((s[mid - 1] + s[mid]) * 500.0)


def _time_predict(mlmodel, batch, *, warmup: int, iters: int) -> dict[str, float]:
    for _ in range(warmup):
        _predict_one(mlmodel, batch["input_ids"], batch["attention_mask"])
    times: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _predict_one(mlmodel, batch["input_ids"], batch["attention_mask"])
        times.append(time.perf_counter() - t0)
    return {
        "warmup": warmup,
        "iters": iters,
        "p50_ms": _median_ms(times),
        "min_ms": float(min(times) * 1000.0) if times else float("nan"),
        "max_ms": float(max(times) * 1000.0) if times else float("nan"),
    }


def _compute_plan_summary(mlmodel, units) -> dict:
    compiled = mlmodel.get_compiled_model_path()
    plan = MLComputePlan.load_from_path(compiled, compute_units=units)
    summary = summarize_compute_plan(plan)
    summary["compiled_path"] = str(compiled)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--artifacts", type=Path, default=artifacts_root())
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--reference", type=Path, default=DEFAULT_NPY)
    parser.add_argument("--mlpackage", type=Path, default=None)
    parser.add_argument("--cosine-gate", type=float, default=COSINE_GATE)
    parser.add_argument("--rel-l2-gate", type=float, default=REL_L2_GATE)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    args = parser.parse_args()

    if ct is None or MLComputePlan is None:
        print("ERROR: coremltools missing (need compute_plan). pip install -r requirements-conversion.txt")
        return 1

    seq_len = int(args.seq_len)
    out_dir = artifact_subdir(seq_len, args.artifacts)
    stem = package_stem(seq_len)
    pkg_path = args.mlpackage or (out_dir / f"{stem}.mlpackage")
    report_path = out_dir / f"{stem}.ane.json"

    os.environ.setdefault("ANEMLL_EMBEDDINGS_ARTIFACTS", str(args.artifacts))
    print(f"ANEMLL_EMBEDDINGS_ARTIFACTS={args.artifacts}")
    print(f"mlpackage={pkg_path}")
    soc = _soc_note()
    print(f"soc={soc['brand']} hw={soc['hw_model']}")

    if not pkg_path.exists():
        print(f"ERROR: missing {pkg_path}")
        return 1

    with args.prompts.open() as f:
        prompts = json.load(f)["prompts"]
    ref = np.load(args.reference)
    if ref.shape != (len(prompts), 768):
        print(f"ERROR: reference shape {ref.shape} != ({len(prompts)}, 768)")
        return 1

    print("Loading tokenizer via text-only ST model …")
    st_model, load_meta = load_sentence_transformer(device="cpu")
    pad_id = int(getattr(st_model.tokenizer, "pad_token_id", 0) or 0)
    batches, truncated = _encode_batches(st_model, prompts, seq_len, pad_id)

    print("Loading Core ML CPU_ONLY …")
    cpu_model = _load_model(pkg_path, ct.ComputeUnit.CPU_ONLY)
    print("Loading Core ML CPU_AND_NE …")
    ane_model = _load_model(pkg_path, ct.ComputeUnit.CPU_AND_NE)

    print("Scoring fixtures on CPU_AND_NE …")
    ane_rows_raw = _run_fixtures(ane_model, batches, ref)
    print("Scoring fixtures on CPU_ONLY (same package) …")
    cpu_rows_raw = _run_fixtures(cpu_model, batches, ref)

    ane_rows = []
    cpu_vs_ane = []
    for a, c in zip(ane_rows_raw, cpu_rows_raw, strict=True):
        print(
            f"{a['id']:<18} ane_cos={a['cosine']:.8f} ane_rel_l2={a['rel_l2']:.6e} "
            f"l2={a['l2_pred']:.6f}"
        )
        ane_rows.append({k: v for k, v in a.items() if k != "vector"})
        cpu_vs_ane.append(
            {
                "id": a["id"],
                "cosine_cpu_vs_ane": cosine(c["vector"], a["vector"]),
                "rel_l2_ane_vs_cpu": rel_l2(a["vector"], c["vector"]),
            }
        )

    print("Compute plan (CPU_AND_NE) …")
    plan_summary = _compute_plan_summary(ane_model, ct.ComputeUnit.CPU_AND_NE)
    placed_ok, place_reason = placement_verdict(plan_summary)
    print(
        f"  by_device={plan_summary.get('by_device')} "
        f"supported_ane={plan_summary.get('supported_ane_ops')} {place_reason}"
    )
    print(f"  top_ane_ops={plan_summary.get('top_ane_ops')}")
    print(f"  top_cpu_ops={plan_summary.get('top_cpu_ops')}")

    print("Timing (same first prompt) …")
    timing = {
        "cpu_only": _time_predict(
            cpu_model, batches[0], warmup=args.warmup, iters=args.iters
        ),
        "cpu_and_ne": _time_predict(
            ane_model, batches[0], warmup=args.warmup, iters=args.iters
        ),
    }
    print(
        f"  CPU p50={timing['cpu_only']['p50_ms']:.2f} ms  "
        f"CPU_AND_NE p50={timing['cpu_and_ne']['p50_ms']:.2f} ms"
    )

    ane_cos = [float(r["cosine"]) for r in ane_rows]
    ane_rel = [float(r["rel_l2"]) for r in ane_rows]
    min_c = float(np.min(ane_cos))
    max_rel = float(np.max(ane_rel))
    all_finite = all(r["finite"]["nan"] == 0 and r["finite"]["inf"] == 0 for r in ane_rows)
    min_cpu_ane = float(np.min([x["cosine_cpu_vs_ane"] for x in cpu_vs_ane]))

    fail_reasons: list[str] = []
    if not all_finite:
        fail_reasons.append("NaN/Inf on CPU_AND_NE path")
    if truncated:
        fail_reasons.append(f"{truncated} prompts truncated to S={seq_len}")
    if min_c < args.cosine_gate:
        fail_reasons.append(f"ANE vs T1 min cosine {min_c:.8f} < {args.cosine_gate}")
    if max_rel > args.rel_l2_gate:
        fail_reasons.append(f"ANE vs T1 max rel-L2 {max_rel:.6e} > {args.rel_l2_gate}")
    if not placed_ok:
        fail_reasons.append(place_reason)

    report = {
        "ticket": "T7",
        "created_at_utc": utc_now(),
        "git_sha": git_sha(REPO_ROOT),
        "seq_len": seq_len,
        "mlpackage": str(pkg_path),
        "reference": str(args.reference),
        "soc": soc,
        "compute_units_scored": "CPU_AND_NE",
        "gates": {"cosine": args.cosine_gate, "rel_l2": args.rel_l2_gate},
        "summary": {
            "min_cosine_ane_vs_t1": min_c,
            "max_rel_l2_ane_vs_t1": max_rel,
            "min_cosine_cpu_vs_ane": min_cpu_ane,
            "all_finite": all_finite,
            "truncated": truncated,
            "placement_ok": placed_ok,
            "placement_reason": place_reason,
            "pass": not fail_reasons,
        },
        "placement": plan_summary,
        "timing_ms": timing,
        "rows_ane_vs_t1": ane_rows,
        "rows_cpu_vs_ane": cpu_vs_ane,
        "load": load_meta,
        "host": host_versions(),
        "notes": [
            "Same .mlpackage as T6. CPU_AND_NE is not itself proof of ANE.",
            "Placement is from MLComputePlan preferred devices (const ops skipped).",
            "Forge bonded-compile mode not applied (Core ML, host may be pre-M5).",
            "Do not promote T6 CPU parity to ANE correctness.",
        ],
        "fail_reasons": fail_reasons,
    }
    write_json(report_path, report)
    print(f"wrote {report_path}")
    print(
        f"min_ane_cos={min_c:.8f} max_ane_rel_l2={max_rel:.6e} "
        f"min_cpu_vs_ane={min_cpu_ane:.8f} placement_ok={placed_ok}"
    )

    if fail_reasons:
        print("FAIL: T7 ANE smoke")
        for reason in fail_reasons:
            print(f" - {reason}")
        return 1
    print("OK: T7 ANE smoke (plan has ANE ops; fixtures finite and on-gate)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
