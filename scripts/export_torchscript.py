#!/usr/bin/env python3
"""T4: torch.jit.trace EmbeddingGemma 2 at a fixed sequence length.

Loads the text-only ST checkpoint, wraps it with the T3 pool/project graph
plus in-graph 4D pad+window masks, traces at S (default 512), and writes
``.pt`` + metadata under ``ANEMLL_EMBEDDINGS_ARTIFACTS``.

Does **not** call anemll-forge ``forge.py convert``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.embed_wrapper import EmbeddingGemma2Wrapper  # noqa: E402
from src.export_utils import (  # noqa: E402
    DEFAULT_SEQ_LEN,
    artifact_subdir,
    artifacts_root,
    example_trace_inputs,
    force_eager_attention,
    git_sha,
    host_versions,
    package_stem,
    sha256_file,
    utc_now,
    write_json,
)
from src.load_text_model import load_sentence_transformer  # noqa: E402
from src.trace_patches import apply_fixed_shape_patches  # noqa: E402
from src.traceable_wrapper import TraceableEmbeddingGemma2  # noqa: E402


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a64 = a.detach().float().reshape(-1)
    b64 = b.detach().float().reshape(-1)
    denom = float(a64.norm() * b64.norm())
    if denom == 0:
        return float("nan")
    return float(torch.dot(a64, b64) / denom)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=artifacts_root(),
        help="ANEMLL_EMBEDDINGS_ARTIFACTS root",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Trace device (cpu recommended for coremltools)",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32",),
        default="float32",
        help="Export dtype. FP32 only — never FP16.",
    )
    parser.add_argument("--pad-last", type=int, default=8)
    parser.add_argument("--strict", action="store_true", help="jit.trace strict=True")
    parser.add_argument(
        "--no-check-trace",
        action="store_true",
        help="Skip second-example check_trace",
    )
    args = parser.parse_args()

    seq_len = int(args.seq_len)
    if seq_len < 8:
        print("ERROR: --seq-len must be >= 8")
        return 1

    out_dir = artifact_subdir(seq_len, args.artifacts)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = package_stem(seq_len)
    pt_path = out_dir / f"{stem}.pt"
    meta_path = out_dir / f"{stem}.pt.meta.json"

    os.environ.setdefault("ANEMLL_EMBEDDINGS_ARTIFACTS", str(args.artifacts))
    print(f"ANEMLL_EMBEDDINGS_ARTIFACTS={args.artifacts}")
    print(f"seq_len={seq_len} device={args.device} dtype={args.dtype}")

    patches = apply_fixed_shape_patches(seq_len, batch=1)
    print(f"  export patches {patches}")

    print("Loading text-only ST model …")
    st_model, load_meta = load_sentence_transformer()
    wrapper = EmbeddingGemma2Wrapper.from_sentence_transformer(
        st_model, normalize=True
    )
    force_eager_attention(wrapper)
    # Convert on CPU FP32 for a coremltools-friendly graph (never FP16).
    wrapper = wrapper.float().to(args.device).eval()
    traced_mod = TraceableEmbeddingGemma2(wrapper, seq_len=seq_len, batch=1)
    traced_mod = traced_mod.to(args.device).eval()

    sliding = int(getattr(wrapper.text_model.config, "sliding_window", -1))
    print(f"  sliding_window={sliding} (checkpoint; PLAN mentioned 1024)")

    ids, mask = example_trace_inputs(
        seq_len, pad_last=args.pad_last, device=args.device
    )
    ids2, mask2 = example_trace_inputs(
        seq_len, pad_last=max(1, args.pad_last // 2), device=args.device
    )

    with torch.inference_mode():
        eager = traced_mod(ids, mask)
        if not torch.isfinite(eager).all():
            print("ERROR: eager trace wrapper produced NaN/Inf")
            return 1
        print(f"  eager embedding shape={tuple(eager.shape)} norm={eager.norm():.6f}")

        print("Tracing …")
        ts = torch.jit.trace(
            traced_mod,
            (ids, mask),
            strict=bool(args.strict),
            check_trace=not args.no_check_trace,
            check_inputs=[(ids2, mask2)] if not args.no_check_trace else None,
        )
        ts_out = ts(ids, mask)
        if not torch.isfinite(ts_out).all():
            print("ERROR: TorchScript produced NaN/Inf")
            return 1
        cos = _cosine(eager, ts_out)
        print(f"  jit vs eager cosine={cos:.8f}")
        if cos < 0.999:
            print("ERROR: jit vs eager cosine below 0.999")
            return 1

    ts.save(str(pt_path))
    print(f"wrote {pt_path}")

    meta = {
        "ticket": "T4",
        "created_at_utc": utc_now(),
        "git_sha": git_sha(REPO_ROOT),
        "seq_len": seq_len,
        "batch": 1,
        "dtype": args.dtype,
        "device": args.device,
        "normalize": True,
        "mask_neg": traced_mod.mask_neg,
        "sliding_window": sliding,
        "inputs": {
            "input_ids": {"shape": [1, seq_len], "dtype": "int32"},
            "attention_mask": {"shape": [1, seq_len], "dtype": "int32"},
        },
        "outputs": {
            "embedding": {"shape": [1, 768], "dtype": "float32", "l2_normalized": True}
        },
        "torchscript": str(pt_path),
        "torchscript_sha256": sha256_file(pt_path),
        "load": load_meta,
        "export_patches": patches,
        "host": host_versions(),
        "notes": [
            "Fixed-S jit.trace; no dynamic shapes.",
            "In-graph 4D pad + sliding-window additive bias (finite -1e4).",
            "Eager attention only. Fixed (B,S) view/reshape patches for coremltools.",
            "Do not call forge.py convert.",
            "MRL truncate+renorm stays on the host (T9).",
        ],
    }
    write_json(meta_path, meta)
    print(f"wrote {meta_path}")
    print("OK: T4 TorchScript export")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
