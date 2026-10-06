#!/usr/bin/env python3
"""Unit checks for T7 placement helpers (no checkpoint, no .mlpackage)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.ane_placement import (  # noqa: E402
    device_kind,
    placement_verdict,
    summarize_compute_plan,
)


def _fail(msg: str) -> None:
    raise AssertionError(msg)


def test_device_kind_none() -> None:
    if device_kind(None) != "unknown":
        _fail(device_kind(None))


def test_verdict_fail_closed() -> None:
    ok, reason = placement_verdict({"available": False, "reason": "missing", "ane_ops": 0})
    if ok:
        _fail(reason)
    ok2, reason2 = placement_verdict(
        {"available": True, "ane_ops": 0, "cpu_ops": 10, "reason": None}
    )
    if ok2 or "0 non-const" not in reason2:
        _fail(reason2)


def test_verdict_pass_with_ane_ops() -> None:
    ok, reason = placement_verdict({"available": True, "ane_ops": 4, "reason": None})
    if not ok or "4" not in reason:
        _fail(reason)


def test_summarize_skips_const() -> None:
    # Fake devices: device_kind falls through to the type name.
    cpu_dev = type("MLCPUComputeDeviceX", (), {})()
    ane_dev = type("MLNeuralEngineComputeDeviceX", (), {})()

    ops = [
        SimpleNamespace(operator_name="const"),
        SimpleNamespace(operator_name="matmul"),
        SimpleNamespace(operator_name="gather"),
    ]
    usages = {
        id(ops[1]): SimpleNamespace(preferred_compute_device=ane_dev),
        id(ops[2]): SimpleNamespace(preferred_compute_device=cpu_dev),
    }

    class _Plan:
        model_structure = SimpleNamespace(
            program=SimpleNamespace(
                functions={
                    "main": SimpleNamespace(block=SimpleNamespace(operations=ops))
                }
            )
        )

        def get_compute_device_usage_for_mlprogram_operation(self, op):
            return usages.get(id(op))

    # device_kind uses isinstance first; fake types fall through to name.
    summary = summarize_compute_plan(_Plan())
    if not summary["available"]:
        _fail(str(summary))
    if summary["total_ops"] != 2:
        _fail(f"const not skipped: {summary}")
    if summary["ane_ops"] != 1 or summary["cpu_ops"] != 1:
        _fail(str(summary["by_device"]))


def main() -> int:
    tests = [
        test_device_kind_none,
        test_verdict_fail_closed,
        test_verdict_pass_with_ane_ops,
        test_summarize_skips_const,
    ]
    for fn in tests:
        fn()
        print(f"OK {fn.__name__}")
    print(f"OK: {len(tests)} ane-placement checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
