#!/usr/bin/env python3
"""Load each Core AI tower once so this Mac specializes and caches it.

There is no per-hardware compile step to ship. Core AI compiles the
``.aimodel`` graph on first load for the local chip and stores the result
under ``$CFFIXED_USER_HOME/Library/Caches/coreai-cache`` (or
``~/Library/Caches/coreai-cache``). Later loads reuse that cache.

Reports per tower: ANE placement, package load time, and first-run time.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

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
    discover_coreai_python,
    fixed_user_home_for_cache,
)


def _ane(placement: str | None) -> str:
    text = str(placement or "")
    if "fullyOnANE" in text or text == "ANE":
        return "yes"
    if not text:
        return "unknown"
    return f"no ({text})"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=None,
        help="ANEMLL_EMBEDDINGS_ARTIFACTS (dir that contains coreai/*.aimodel)",
    )
    parser.add_argument(
        "--coreai-python",
        type=Path,
        default=None,
        dest="coreai_python",
        help="interpreter that can import coreai.runtime "
        "(default: $ANEMLL_COREAI_PYTHON, then common anemll-forge venvs)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Core AI cache directory (default: $CFFIXED_USER_HOME or "
        "~/Library/Caches/coreai-cache)",
    )
    parser.add_argument("--compute", default="ane", choices=("ane", "cpu"))
    args = parser.parse_args(argv)

    artifacts = args.artifacts if args.artifacts is not None else default_artifacts()
    if artifacts is None:
        raise SystemExit(
            "set ANEMLL_EMBEDDINGS_ARTIFACTS or pass --artifacts "
            "(run scripts/download_models.py first)"
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
    cache = coreai_cache_dir(args.cache_dir)
    home = fixed_user_home_for_cache(cache)
    if home is not None:
        os.environ["CFFIXED_USER_HOME"] = str(home)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
