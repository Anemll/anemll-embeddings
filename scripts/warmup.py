#!/usr/bin/env python3
"""Load each Core AI tower once so this Mac specializes and caches it.

There is no per-hardware compile step to ship. Core AI compiles the
``.aimodel`` graph on first load for the local chip and stores the result
under ``$CFFIXED_USER_HOME/Library/Caches/coreai-cache`` (or
``~/Library/Caches/coreai-cache``). Later loads reuse that cache.

Reports per tower: ANE placement, package load time, and first-run time.
With ``--require-ane`` the exit status is non-zero (3) unless every tower
is fully on the Neural Engine, so CI and scripts can tell ANE success
from a GPU fallback.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api.embedder import (  # noqa: E402
    CoreAIWorkerClient,
    default_artifacts,
    package_paths,
)
from scripts.download_common import (  # noqa: E402
    cache_unwritable_hint,
    coreai_cache_dir,
    coreai_home_for_cache_dir,
    discover_coreai_python,
)

EXIT_NOT_ON_ANE = 3


def _ane(placement: str | None) -> str:
    text = str(placement or "")
    if "fullyOnANE" in text or text == "ANE":
        return "yes"
    if not text:
        return "unknown"
    return f"no ({text})"


def towers_not_on_ane(report: dict[str, Any]) -> list[str]:
    """Names of towers whose placement is not fully on the Neural Engine."""
    towers = report.get("towers") or {}
    if not towers:
        return ["<no towers reported>"]
    return [name for name, row in towers.items() if _ane(row.get("placement")) != "yes"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=None,
        help="dir that contains coreai/*.aimodel (default: $ANEMLL_EMBEDDINGS_ARTIFACTS, "
        "then ~/.anemll-embeddings/artifacts)",
    )
    parser.add_argument(
        "--coreai-python",
        type=Path,
        default=None,
        dest="coreai_python",
        help="interpreter that can import coreai.runtime "
        "(default: $ANEMLL_COREAI_PYTHON, then the documented locations)",
    )
    where = parser.add_mutually_exclusive_group()
    where.add_argument(
        "--coreai-home",
        type=Path,
        default=None,
        help="redirect Core AI's home (sets CFFIXED_USER_HOME); the cache becomes "
        "<dir>/Library/Caches/coreai-cache",
    )
    where.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Core AI cache directory. Must end in Library/Caches/coreai-cache "
        "(Core AI has no free-form cache path); other paths are rejected",
    )
    parser.add_argument("--compute", default="ane", choices=("ane", "cpu"))
    parser.add_argument(
        "--require-ane",
        action="store_true",
        help=f"exit {EXIT_NOT_ON_ANE} unless every tower is fully on the Neural Engine",
    )
    return parser


def resolve_coreai_home(args: argparse.Namespace) -> Path | None:
    """``CFFIXED_USER_HOME`` implied by ``--coreai-home`` / ``--cache-dir`` (or None)."""
    if args.coreai_home is not None:
        return Path(args.coreai_home).expanduser().resolve()
    if args.cache_dir is not None:
        return coreai_home_for_cache_dir(Path(args.cache_dir).expanduser().resolve())
    return None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.require_ane and args.compute != "ane":
        raise SystemExit("--require-ane needs --compute ane")

    artifacts = args.artifacts if args.artifacts is not None else default_artifacts()
    if artifacts is None:
        raise SystemExit(
            "no Core AI packages found: run scripts/download_models.py "
            "(writes ~/.anemll-embeddings/artifacts), or set "
            "ANEMLL_EMBEDDINGS_ARTIFACTS / pass --artifacts"
        )
    packages = package_paths(artifacts)
    missing = [str(path) for path in packages.values() if not path.exists()]
    if missing:
        raise SystemExit(
            "missing Core AI packages: "
            + ", ".join(missing)
            + "\nrun scripts/download_models.py"
        )

    py = discover_coreai_python(args.coreai_python, required=True)
    home = resolve_coreai_home(args)
    if home is not None:
        (home / "Library" / "Caches").mkdir(parents=True, exist_ok=True)
        os.environ["CFFIXED_USER_HOME"] = str(home)
    cache = coreai_cache_dir()
    print(
        "No per-hardware compile is published. First load specializes for this "
        "chip and Core AI caches it; later loads are fast."
    )
    print(f"artifacts={Path(artifacts).resolve()} compute={args.compute}")
    print(f"coreai_python={py}")
    print(f"cache={cache}")
    client = CoreAIWorkerClient(packages, compute=args.compute, python=py)
    started = time.perf_counter()
    try:
        report = client.startup()
    except Exception as exc:
        print(cache_unwritable_hint(cache), file=sys.stderr)
        raise SystemExit(f"warmup failed ({type(exc).__name__}: {exc})") from exc
    finally:
        client.close()
    elapsed = (time.perf_counter() - started) * 1000.0

    towers = report.get("towers") or {}
    print(f"{'tower':<22} {'on ANE':<18} {'load_ms':>10} {'first_run_ms':>14}")
    for name, row in towers.items():
        place = row.get("placement")
        load_ms = row.get("load_ms")
        first_ms = row.get("warmup_ms")
        load_s = f"{float(load_ms):.1f}" if load_ms is not None else "-"
        first_s = f"{float(first_ms):.1f}" if first_ms is not None else "-"
        print(f"{name:<22} {_ane(place):<18} {load_s:>10} {first_s:>14}")
    print(f"overall_placement={report.get('placement')}  wall_ms={elapsed:.1f}")
    if args.require_ane:
        off = towers_not_on_ane(report)
        if off:
            print(
                f"--require-ane: not fully on the Neural Engine: {', '.join(off)}",
                file=sys.stderr,
            )
            return EXIT_NOT_ON_ANE
        print("--require-ane: all towers fully on the Neural Engine")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
