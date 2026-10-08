#!/usr/bin/env python3
"""Unit checks for T7 placement helpers (no checkpoint, no .mlpackage)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.ane_placement import (  # noqa: E402
    classify_device_sequence,
    cpu_op_role,
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


def test_cpu_op_role() -> None:
    if cpu_op_role("ios18.gather", ["embed_weight"]) != "embedding_gather":
        _fail(cpu_op_role("ios18.gather", ["embed_weight"]))
    if cpu_op_role("ios18.greater_equal", ["input_ids"]) != "ids_bounds":
        _fail(cpu_op_role("ios18.greater_equal", ["input_ids"]))
    if cpu_op_role("ios18.logical_and", ["keep_1"]) != "mask_bias_sliding":
        _fail(cpu_op_role("ios18.logical_and", ["keep_1"]))
    if cpu_op_role("ios18.cast", ["attention_mask"]) != "mask_bias":
        _fail(cpu_op_role("ios18.cast", ["attention_mask"]))


def test_begin_end_only_no_mid_island() -> None:
    rows = (
        [{"op": "cast", "device": "CPU", "supported": ["CPU"], "bindings": ["attention_mask"]}]
        * 3
        + [{"op": "linear", "device": "ANE", "supported": ["ANE"], "bindings": []}] * 5
    )
    got = classify_device_sequence(rows)
    if not got["begin_end_switches_only"]:
        _fail(str(got))
    if got["mid_graph_switch_count"] != 0 or got["mid_graph_cpu_islands"]:
        _fail(str(got))
    if [r["device"] for r in got["device_runs"]] != ["CPU", "ANE"]:
        _fail(str(got["device_runs"]))
    if any(op["position"] != "begin" for op in got["cpu_ops_classified"]):
        _fail(str(got["cpu_ops_classified"]))


def test_mid_graph_island_is_flagged() -> None:
    rows = [
        {"op": "cast", "device": "CPU", "supported": ["CPU"], "bindings": []},
        {"op": "linear", "device": "ANE", "supported": ["ANE"], "bindings": []},
        {"op": "gather", "device": "CPU", "supported": ["CPU"], "bindings": []},
        {"op": "linear", "device": "ANE", "supported": ["ANE"], "bindings": []},
        {"op": "cast", "device": "CPU", "supported": ["CPU"], "bindings": []},
    ]
    got = classify_device_sequence(rows)
    if got["begin_end_switches_only"]:
        _fail("mid island should fail begin_end_only")
    if len(got["mid_graph_cpu_islands"]) != 1:
        _fail(str(got["mid_graph_cpu_islands"]))
    if got["mid_graph_switch_count"] < 2:
        _fail(str(got["device_switches"]))
    positions = [op["position"] for op in got["cpu_ops_classified"]]
    if positions != ["begin", "mid", "end"]:
        _fail(positions)


def main() -> int:
    tests = [
        test_device_kind_none,
        test_verdict_fail_closed,
        test_verdict_pass_with_ane_ops,
        test_summarize_skips_const,
        test_cpu_op_role,
        test_begin_end_only_no_mid_island,
        test_mid_graph_island_is_flagged,
    ]
    for fn in tests:
        fn()
        print(f"OK {fn.__name__}")
    print(f"OK: {len(tests)} ane-placement checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
