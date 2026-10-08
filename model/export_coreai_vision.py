#!/usr/bin/env python3
"""Core AI export entry (vision first). Does **not** call forge.py convert.

Primary path: torch.export → coreai_torch.TorchConverter → .aimodel.

The embeddings venv has no ``coreai_torch``. This script re-execs under
``ANEMLL_COREAI_PYTHON`` (default: anemll-forge ``coreai/.venv``) for the
toolchain probe / convert. It never imports forge's Qwen convert.

``--probe`` writes a tiny conv .aimodel to prove the Core AI toolchain.
Real towers: ``model/export_coreai_towers.py``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COREAI_PY = Path("/Users/anemll/anemll-forge/coreai/.venv/bin/python")


def _coreai_python() -> Path:
    raw = os.environ.get("ANEMLL_COREAI_PYTHON")
    return Path(raw) if raw else DEFAULT_COREAI_PY


def _have_coreai() -> bool:
    try:
        import coreai_torch  # noqa: F401

        return True
    except ImportError:
        return False


def _reexec_if_needed(argv: list[str]) -> None:
    if _have_coreai():
        return
    py = _coreai_python()
    if not py.is_file():
        print(
            "ERROR: coreai_torch missing and ANEMLL_COREAI_PYTHON not found at "
            f"{py}. Point it at forge coreai/.venv/bin/python. "
            "Do not run forge.py convert."
        )
        raise SystemExit(2)
    cmd = [str(py), str(Path(__file__).resolve()), *argv[1:]]
    print(f"re-exec {py} (coreai_torch toolchain)")
    raise SystemExit(subprocess.call(cmd))


def _probe_aimodel(out: Path) -> None:
    import torch
    import torch.nn as nn
    import coreai_torch
    from coreai_opt.casting import cast_to_16_bit_precision

    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            torch.manual_seed(0)
            self.conv = nn.Conv2d(3, 8, 1, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv(x)

    m = Tiny().eval().to(torch.float16)
    x = torch.randn(1, 3, 16, 16, dtype=torch.float16)
    ep = torch.export.export(m, (x,), strict=False).run_decompositions(
        coreai_torch.get_decomp_table()
    )
    cast_to_16_bit_precision(ep)
    conv = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    conv.add_exported_program(
        ep, input_names=["pixel_values"], output_names=["features"], entrypoint_name="vision_probe"
    )
    prog = conv.to_coreai()
    prog.optimize()
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(out, ignore_errors=True)
    prog.save_asset(out)
    print(f"wrote {out}")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true", help="Tiny conv .aimodel toolchain check")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output .aimodel path (default: $ANEMLL_EMBEDDINGS_ARTIFACTS/coreai/…)",
    )
    args, _unknown = parser.parse_known_args(argv[1:])

    artifacts = Path(os.environ.get("ANEMLL_EMBEDDINGS_ARTIFACTS", "/Volumes/Models/anemll-embeddings/artifacts"))
    out = args.out or (artifacts / "coreai" / "vision_probe.aimodel")

    if not args.probe:
        print(
            "Core AI is the primary target (torch.export → .aimodel). "
            "Pass --probe to compile a tiny conv package with the forge Core AI "
            "venv. Real vision_tower / audio_tower export follows ST fixtures."
        )
        print(f"ANEMLL_COREAI_PYTHON={_coreai_python()}")
        print(f"coreai_torch_here={_have_coreai()}")
        return 0

    _reexec_if_needed(argv)
    print(f"probe out={out}")
    _probe_aimodel(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
