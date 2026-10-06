#!/usr/bin/env python3
"""T6: Core ML (CPU) vs T1 Sentence-Transformers fixture parity.

Compares the fixed-S ``.mlpackage`` embeddings to committed
``tests/fixtures/embeddings.npy`` (cosine, rel-L2, L2 norms, NaN/Inf,
pairwise ``cos(q,d)`` agreement). Does **not** claim ANE placement.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

try:
    import coremltools as ct
except ImportError:
    ct = None

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

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
    FIXTURE_PAIRS,
    PAIR_DELTA_GATE,
    REL_L2_GATE,
    cosine,
    finite_counts,
    l2_norm,
    pair_deltas,
    pairwise_cosines,
    rel_l2,
)

DEFAULT_PROMPTS = REPO_ROOT / "tests" / "fixtures" / "prompts.json"
DEFAULT_NPY = REPO_ROOT / "tests" / "fixtures" / "embeddings.npy"


def _predict_one(mlmodel, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    pred = mlmodel.predict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
    )
    if "embedding" in pred:
        vec = np.asarray(pred["embedding"])
    else:
        vec = np.asarray(next(iter(pred.values())))
    return vec.astype(np.float32).reshape(-1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--artifacts", type=Path, default=artifacts_root())
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--reference", type=Path, default=DEFAULT_NPY)
    parser.add_argument("--mlpackage", type=Path, default=None)
    parser.add_argument("--cosine-gate", type=float, default=COSINE_GATE)
    parser.add_argument("--rel-l2-gate", type=float, default=REL_L2_GATE)
    parser.add_argument("--pair-delta-gate", type=float, default=PAIR_DELTA_GATE)
    args = parser.parse_args()

    if ct is None:
        print(
            "ERROR: coremltools is not installed in this venv.\n"
            "  pip install -r requirements-conversion.txt"
        )
        return 1

    seq_len = int(args.seq_len)
    out_dir = artifact_subdir(seq_len, args.artifacts)
    stem = package_stem(seq_len)
    pkg_path = args.mlpackage or (out_dir / f"{stem}.mlpackage")
    report_path = out_dir / f"{stem}.parity.json"

    os.environ.setdefault("ANEMLL_EMBEDDINGS_ARTIFACTS", str(args.artifacts))
    print(f"ANEMLL_EMBEDDINGS_ARTIFACTS={args.artifacts}")
    print(f"mlpackage={pkg_path}")

    if not pkg_path.is_dir() and not pkg_path.is_file():
        print(f"ERROR: missing Core ML package {pkg_path}")
        return 1

    with args.prompts.open() as f:
        prompt_pack = json.load(f)
    prompts = prompt_pack["prompts"]
    ref = np.load(args.reference)
    if ref.shape != (len(prompts), 768):
        print(f"ERROR: reference shape {ref.shape} != ({len(prompts)}, 768)")
        return 1

    print("Loading tokenizer via text-only ST model …")
    st_model, load_meta = load_sentence_transformer(device="cpu")
    pad_id = int(getattr(st_model.tokenizer, "pad_token_id", 0) or 0)

    print(f"Loading Core ML (CPU_ONLY) {pkg_path}")
    mlmodel = ct.models.MLModel(str(pkg_path), compute_units=ct.ComputeUnit.CPU_ONLY)

    rows: list[dict] = []
    vectors: list[np.ndarray] = []
    truncated = 0
    for p in prompts:
        batch = tokenize_with_st_prompt(
            st_model,
            p["text"],
            p.get("prompt_name"),
            device="cpu",
        )
        raw_len = int(batch["attention_mask"].sum().item())
        ids, mask = pad_to_seq_len(
            batch["input_ids"],
            batch["attention_mask"],
            seq_len,
            pad_token_id=pad_id,
        )
        was_trunc = raw_len > seq_len
        if was_trunc:
            truncated += 1
        vec = _predict_one(
            mlmodel,
            ids.numpy().astype(np.int32),
            mask.numpy().astype(np.int32),
        )
        vectors.append(vec)
        idx = len(vectors) - 1
        row = {
            "id": p["id"],
            "prompt_name": p.get("prompt_name"),
            "raw_tokens": raw_len,
            "truncated": was_trunc,
            "cosine": cosine(vec, ref[idx]),
            "rel_l2": rel_l2(vec, ref[idx]),
            "l2_pred": l2_norm(vec),
            "l2_ref": l2_norm(ref[idx]),
            "finite": finite_counts(vec),
        }
        rows.append(row)
        print(
            f"{p['id']:<18} cosine={row['cosine']:.8f} "
            f"rel_l2={row['rel_l2']:.6e} l2={row['l2_pred']:.6f} "
            f"tok={raw_len}{' TRUNC' if was_trunc else ''}"
        )

    got = np.stack(vectors, axis=0).astype(np.float32)
    ids = [p["id"] for p in prompts]
    st_pairs = pairwise_cosines(ref, ids)
    cml_pairs = pairwise_cosines(got, ids)
    deltas = pair_deltas(st_pairs, cml_pairs)

    print("pair                 st_cos    cml_cos   abs_delta")
    for d in deltas:
        if d["skipped"]:
            print(f"{d['left']}+{d['right']:<12} skipped")
            continue
        print(
            f"{d['left']}+{d['right']:<12} "
            f"{d['st_cosine']:.6f}  {d['coreml_cosine']:.6f}  "
            f"{d['abs_delta']:.6e}"
        )

    cosines = [float(r["cosine"]) for r in rows]
    rels = [float(r["rel_l2"]) for r in rows]
    pair_abs = [float(d["abs_delta"]) for d in deltas if d["abs_delta"] is not None]
    min_c = float(np.min(cosines))
    max_rel = float(np.max(rels))
    max_pair = float(np.max(pair_abs)) if pair_abs else float("nan")
    all_finite = all(r["finite"]["nan"] == 0 and r["finite"]["inf"] == 0 for r in rows)

    print(
        f"min_cosine={min_c:.8f} max_rel_l2={max_rel:.6e} "
        f"max_pair_delta={max_pair:.6e} finite={all_finite} "
        f"truncated={truncated}"
    )

    fail_reasons: list[str] = []
    if not all_finite:
        fail_reasons.append("NaN/Inf in Core ML embeddings")
    if truncated:
        fail_reasons.append(f"{truncated} prompts truncated to S={seq_len}")
    if min_c < args.cosine_gate:
        fail_reasons.append(f"min cosine {min_c:.8f} < {args.cosine_gate}")
    if max_rel > args.rel_l2_gate:
        fail_reasons.append(f"max rel-L2 {max_rel:.6e} > {args.rel_l2_gate}")
    if pair_abs and max_pair > args.pair_delta_gate:
        fail_reasons.append(f"max pair |Δcos| {max_pair:.6e} > {args.pair_delta_gate}")

    report = {
        "ticket": "T6",
        "created_at_utc": utc_now(),
        "git_sha": git_sha(REPO_ROOT),
        "seq_len": seq_len,
        "mlpackage": str(pkg_path),
        "reference": str(args.reference),
        "compute_units": "CPU_ONLY",
        "gates": {
            "cosine": args.cosine_gate,
            "rel_l2": args.rel_l2_gate,
            "pair_delta": args.pair_delta_gate,
        },
        "summary": {
            "min_cosine": min_c,
            "mean_cosine": float(np.mean(cosines)),
            "max_rel_l2": max_rel,
            "max_pair_delta": max_pair,
            "all_finite": all_finite,
            "truncated": truncated,
            "pass": not fail_reasons,
        },
        "rows": rows,
        "pairs": deltas,
        "load": load_meta,
        "host": host_versions(),
        "notes": [
            "Core ML CPU vs committed T1 ST fixtures (not a live ST re-encode).",
            "CPU_ONLY does not prove ANE placement (T7).",
            f"Fixture pairs: {list(FIXTURE_PAIRS)}",
        ],
        "fail_reasons": fail_reasons,
    }
    write_json(report_path, report)
    print(f"wrote {report_path}")

    if fail_reasons:
        print("FAIL: T6 parity")
        for reason in fail_reasons:
            print(f" - {reason}")
        return 1
    print("OK: T6 Core ML CPU matches T1 fixtures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
