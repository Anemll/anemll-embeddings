#!/usr/bin/env python3
"""Convert several saved torch.export programs into ONE multi-function .aimodel.

Core AI venv only (no repo imports). Each ``--program`` is
``entry=path.pt2:in1,in2:out1``. Every program becomes a named function in
the same package. The weights of identical constants are stored once when
the asset writer dedups equal blobs; check the package size to confirm.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _parse(spec: str) -> tuple[str, Path, list[str], list[str]]:
    entry, rest = spec.split("=", 1)
    path, ins, outs = rest.split(":")
    return entry, Path(path), [n for n in ins.split(",") if n], [n for n in outs.split(",") if n]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import coreai_torch
    import torch
    from coreai_opt.casting import cast_to_16_bit_precision

    conv = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    for spec in args.program:
        entry, path, ins, outs = _parse(spec)
        print(f"load {entry} <- {path}")
        ep = torch.export.load(str(path))
        ep = ep.run_decompositions(coreai_torch.get_decomp_table())
        cast_to_16_bit_precision(ep)
        conv.add_exported_program(ep, input_names=ins, output_names=outs, entrypoint_name=entry)
    prog = conv.to_coreai()
    prog.optimize()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(args.out, ignore_errors=True)
    prog.save_asset(args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
