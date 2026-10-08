#!/usr/bin/env python3
"""T5: coremltools convert of the T4 TorchScript package.

Reads ``embeddinggemma2-text-s{S}.pt`` from ``ANEMLL_EMBEDDINGS_ARTIFACTS``,
converts a fixed-S Core ML ``.mlpackage`` (CPU_AND_NE, FP32 compute), and
writes I/O metadata. Does **not** call anemll-forge ``forge.py convert``.

I/O (also documented in README)::

    input_ids        int32  [1, S]
    attention_mask   int32  [1, S]   # 1=token, 0=pad
    embedding        fp32   [1, 768] # L2-normalized when T4 normalize=True
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
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

from src.export_utils import (  # noqa: E402
    DEFAULT_SEQ_LEN,
    artifact_subdir,
    artifacts_root,
    example_trace_inputs,
    git_sha,
    host_versions,
    package_stem,
    sha256_file,
    utc_now,
    write_json,
)


def _resolve_compute_units(name: str, ct):
    key = name.upper().replace("-", "_")
    mapping = {
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        "ALL": ct.ComputeUnit.ALL,
    }
    if hasattr(ct.ComputeUnit, "CPU_AND_GPU"):
        mapping["CPU_AND_GPU"] = ct.ComputeUnit.CPU_AND_GPU
    if key not in mapping:
        raise ValueError(f"unknown compute units {name!r}; choose {sorted(mapping)}")
    return mapping[key]


def _resolve_target(name: str, ct):
    key = name.replace("-", "").replace("_", "").lower()
    aliases = {
        "macos15": "macOS15",
        "macos14": "macOS14",
        "macos13": "macOS13",
        "ios18": "iOS18",
        "ios17": "iOS17",
        "macos26": "macOS26",
        "ios26": "iOS26",
    }
    attr = aliases.get(key, name)
    if not hasattr(ct.target, attr):
        available = [a for a in dir(ct.target) if a[:1].isupper()]
        raise ValueError(
            f"deployment target {name!r} not in coremltools "
            f"({available}); pick one of {sorted(aliases)}"
        )
    return getattr(ct.target, attr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--artifacts", type=Path, default=artifacts_root())
    parser.add_argument(
        "--compute-units",
        default="CPU_AND_NE",
        help="CPU_AND_NE (default), CPU_ONLY, ALL",
    )
    parser.add_argument(
        "--deployment-target",
        default="macos15",
        help="coremltools ct.target name (default macos15)",
    )
    parser.add_argument(
        "--precision",
        default="FLOAT32",
        choices=("FLOAT32",),
        help="FP32 baseline only (FP16 is T10; never the first convert).",
    )
    parser.add_argument(
        "--skip-predict",
        action="store_true",
        help="Skip post-convert CPU predict smoke",
    )
    args = parser.parse_args()

    if ct is None:
        print(
            "ERROR: coremltools is not installed in this venv.\n"
            "  /Volumes/Models/anemll-embeddings/.venv/bin/python -m pip install "
            "'coremltools==9.0'"
        )
        return 1

    seq_len = int(args.seq_len)
    out_dir = artifact_subdir(seq_len, args.artifacts)
    stem = package_stem(seq_len)
    pt_path = out_dir / f"{stem}.pt"
    pkg_path = out_dir / f"{stem}.mlpackage"
    meta_path = out_dir / f"{stem}.convert.json"

    os.environ.setdefault("ANEMLL_EMBEDDINGS_ARTIFACTS", str(args.artifacts))
    print(f"ANEMLL_EMBEDDINGS_ARTIFACTS={args.artifacts}")

    if not pt_path.is_file():
        print(f"ERROR: missing TorchScript {pt_path} (run scripts/export_torchscript.py)")
        return 1

    compute_units = _resolve_compute_units(args.compute_units, ct)
    min_target = _resolve_target(args.deployment_target, ct)
    precision = ct.precision.FLOAT32

    print(f"Loading {pt_path}")
    ts = torch.jit.load(str(pt_path), map_location="cpu")
    ts.eval()

    print(
        f"Converting seq_len={seq_len} units={args.compute_units} "
        f"target={args.deployment_target} precision={args.precision}"
    )
    mlmodel = ct.convert(
        ts,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, seq_len), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, seq_len), dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="embedding", dtype=np.float32)],
        minimum_deployment_target=min_target,
        compute_units=compute_units,
        compute_precision=precision,
    )
    mlmodel.short_description = (
        "EmbeddingGemma 2 text-only (270M): ids/mask → 768-d L2 embedding. "
        f"Fixed S={seq_len}. Not from forge.py convert."
    )
    mlmodel.author = "anemll-embeddings"
    pkg_path.parent.mkdir(parents=True, exist_ok=True)
    if pkg_path.exists():
        if pkg_path.is_dir():
            shutil.rmtree(pkg_path)
        else:
            pkg_path.unlink()
    mlmodel.save(str(pkg_path))
    print(f"wrote {pkg_path}")

    spec = mlmodel.get_spec()
    in_names = [inp.name for inp in spec.description.input]
    out_names = [out.name for out in spec.description.output]
    export_meta = out_dir / f"{stem}.pt.meta.json"
    ts_sha = None
    if export_meta.is_file():
        ts_sha = json.loads(export_meta.read_text()).get("torchscript_sha256")
    spec_path = pkg_path / "Data" / "com.apple.CoreML" / "model.mlmodel"

    meta = {
        "ticket": "T5",
        "created_at_utc": utc_now(),
        "git_sha": git_sha(REPO_ROOT),
        "seq_len": seq_len,
        "torchscript": str(pt_path),
        "torchscript_sha256": ts_sha,
        "mlpackage": str(pkg_path),
        "mlmodel_sha256": sha256_file(spec_path) if spec_path.is_file() else None,
        "compute_units": args.compute_units,
        "compute_precision": args.precision,
        "minimum_deployment_target": args.deployment_target,
        "inputs": {
            "input_ids": {"shape": [1, seq_len], "dtype": "int32"},
            "attention_mask": {"shape": [1, seq_len], "dtype": "int32"},
        },
        "outputs": {
            "embedding": {
                "shape": [1, 768],
                "dtype": "float32",
                "l2_normalized": True,
            }
        },
        "spec_input_names": in_names,
        "spec_output_names": out_names,
        "cpu_predict_smoke": {
            "ran": False,
            "finite": None,
            "shape": None,
            "l2_norm": None,
        },
        "host": host_versions(),
        "notes": [
            "CPU_AND_NE does not prove ANE placement (T7).",
            "FP32 baseline; FP16 hazard is T10.",
            "Do not call forge.py convert.",
        ],
    }
    write_json(meta_path, meta)
    print(f"wrote {meta_path}")
    print(f"I/O names: inputs={in_names} outputs={out_names}")

    if not args.skip_predict:
        ids, mask = example_trace_inputs(seq_len, pad_last=8, device="cpu")
        loaded = ct.models.MLModel(str(pkg_path), compute_units=ct.ComputeUnit.CPU_ONLY)
        pred = loaded.predict(
            {
                "input_ids": ids.numpy(),
                "attention_mask": mask.numpy(),
            }
        )
        if "embedding" in pred:
            vec = np.asarray(pred["embedding"])
        else:
            vec = np.asarray(next(iter(pred.values())))
        predict_shape = list(vec.shape)
        predict_ok = bool(np.isfinite(vec).all())
        predict_norm = float(np.linalg.norm(vec.reshape(-1)))
        print(
            f"  CPU predict shape={predict_shape} finite={predict_ok} "
            f"l2={predict_norm:.6f} keys={list(pred)}"
        )
        meta["cpu_predict_smoke"] = {
            "ran": True,
            "finite": predict_ok,
            "shape": predict_shape,
            "l2_norm": predict_norm,
            "output_keys": list(pred),
        }
        write_json(meta_path, meta)
        if not predict_ok:
            print("ERROR: Core ML predict produced NaN/Inf")
            return 1
        if vec.reshape(-1).shape[-1] != 768:
            print(f"ERROR: embedding dim {vec.shape} != 768")
            return 1

    print("OK: T5 Core ML convert")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
