#!/usr/bin/env python3
"""Load one Core AI ``.aimodel`` and run a numpy feed. No ST imports.

``ANEMLL_COREAI_COMPUTE`` selects the device. Unset stays CPU, which is the
parity default. ``ane`` prefers the Neural Engine. The showcase server sets
the variable to ``ane`` in its worker; it does not change this default.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COREAI_PY = Path("/Users/anemll/anemll-forge/coreai/.venv/bin/python")


def _coreai_python() -> Path:
    raw = os.environ.get("ANEMLL_COREAI_PYTHON")
    return Path(raw) if raw else DEFAULT_COREAI_PY


def _reexec_if_needed() -> None:
    try:
        import coreai.runtime  # noqa: F401
    except ImportError:
        py = _coreai_python()
        if not py.is_file():
            raise SystemExit(f"ERROR: coreai python missing at {py}")
        os.execv(str(py), [str(py), *sys.argv])


def parity_compute() -> str:
    """Device the parity runner uses. Unset means CPU."""
    return os.environ.get("ANEMLL_COREAI_COMPUTE", "cpu")


def specialization_options():
    """Options for ``AIModel.load``. ``None`` when the runtime has no switch.

    Same branch ``parity_coreai_host.py`` gets by shelling out to this script.
    """
    from coreai.runtime import ComputeUnitKind, SpecializationOptions

    if not SpecializationOptions.is_supported():
        return None
    if parity_compute() == "ane":
        return SpecializationOptions.from_preferred_compute_unit_kind(
            ComputeUnitKind.neural_engine()
        )
    return SpecializationOptions.cpu_only()


async def _run(pkg: Path, entry: str, feed: dict[str, np.ndarray], out: Path) -> None:
    from coreai.runtime import AIModel, NDArray

    opts = specialization_options()
    model = await AIModel.load(pkg, specialization_options=opts) if opts else await AIModel.load(pkg)
    names = list(model.function_names)
    fn_name = entry if entry in names else names[0]
    fn = model.load_function(fn_name)
    inputs = {k: NDArray(v) for k, v in feed.items()}
    outs = await fn(inputs=inputs)
    first = next(iter(outs))
    arr = outs[first].numpy()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, arr)
    print(f"wrote {out} shape={list(arr.shape)} dtype={arr.dtype} entry={fn_name}")


def main() -> int:
    _reexec_if_needed()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--entry", required=True)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = np.load(args.npz)
    feed = {k: data[k] for k in data.files}
    asyncio.run(_run(args.package, args.entry, feed, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
