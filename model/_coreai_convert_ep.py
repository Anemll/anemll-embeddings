#!/usr/bin/env python3
"""Convert a saved torch.export program to .aimodel (Core AI venv only).

Standalone: no repo ``model`` / ``api`` imports, no transformers, no forge.py convert.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ep", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--entry", required=True)
    parser.add_argument("--input-names", required=True, help="comma-separated")
    parser.add_argument("--output-names", required=True, help="comma-separated")
    parser.add_argument(
        "--no-cast16",
        action="store_true",
        help="Skip cast_to_16_bit_precision (debug int-div dtype clashes).",
    )
    args = parser.parse_args()

    try:
        import torch
        import coreai_torch
        from coreai_opt.casting import cast_to_16_bit_precision
    except ImportError as exc:
        print(f"ERROR: need coreai_torch in this interpreter: {exc}")
        return 2

    if not args.ep.is_file():
        print(f"ERROR: missing exported program {args.ep}")
        return 1

    print(f"load {args.ep} no_cast16={args.no_cast16}")
    ep = torch.export.load(str(args.ep))
    ep = ep.run_decompositions(coreai_torch.get_decomp_table())
    if not args.no_cast16:
        cast_to_16_bit_precision(ep)
    conv = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    conv.add_exported_program(
        ep,
        input_names=[n for n in args.input_names.split(",") if n],
        output_names=[n for n in args.output_names.split(",") if n],
        entrypoint_name=args.entry,
    )
    prog = conv.to_coreai()
    prog.optimize()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(args.out, ignore_errors=True)
    prog.save_asset(args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
