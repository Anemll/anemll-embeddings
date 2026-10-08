"""T7 helpers: classify Core ML compute-plan devices (no model load)."""

from __future__ import annotations

from collections import Counter
from itertools import pairwise
from typing import Any

from coremltools.models.compute_device import (
    MLCPUComputeDevice,
    MLGPUComputeDevice,
    MLNeuralEngineComputeDevice,
)

_ACCEL = frozenset({"ANE", "GPU"})
_EMPTY_CLASSIFICATION = {
    "cpu_ops_classified": [],
    "device_runs": [],
    "mid_graph_cpu_islands": [],
    "device_switches": [],
    "mid_graph_switch_count": 0,
    "begin_end_switches_only": True,
}


def device_kind(device: object | None) -> str:
    if device is None:
        return "unknown"
    if isinstance(device, MLNeuralEngineComputeDevice):
        return "ANE"
    if isinstance(device, MLGPUComputeDevice):
        return "GPU"
    if isinstance(device, MLCPUComputeDevice):
        return "CPU"
    name = type(device).__name__
    if "Neural" in name or name.endswith("NE"):
        return "ANE"
    if "GPU" in name:
        return "GPU"
    if "CPU" in name:
        return "CPU"
    return name or "unknown"


def input_binding_names(op: object) -> list[str]:
    """Collect MIL argument binding names (for role hints, not shapes)."""
    inputs = getattr(op, "inputs", None) or {}
    names: list[str] = []
    if not isinstance(inputs, dict):
        return names
    for arg in inputs.values():
        for binding in getattr(arg, "bindings", None) or []:
            name = getattr(binding, "name", None)
            if name:
                names.append(str(name))
    return names


def cpu_op_role(op_name: str, bindings: list[str] | None = None) -> str:
    """Heuristic role for a CPU leftover (mask / gather / I/O / other)."""
    short = str(op_name).split(".")[-1]
    blob = " ".join([str(op_name), *(bindings or [])]).lower()
    if short == "gather" or "gather" in short:
        return "embedding_gather"
    if "input_ids" in blob and short in {
        "greater_equal",
        "add",
        "select",
        "less",
        "less_equal",
        "clip",
    }:
        return "ids_bounds"
    if short in {"logical_and", "logical_not"} or "keep" in blob:
        return "mask_bias_sliding"
    if "attention_mask" in blob or short in {"tile", "expand_dims", "sub", "mul"}:
        return "mask_bias"
    if short == "cast":
        return "io_cast"
    return "other"


def classify_device_sequence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Collapse preferred-device runs and flag mid-graph CPU islands.

    A CPU island is ``begin`` (before first ANE/GPU), ``end`` (after last
    ANE/GPU), or ``mid`` (ANE/GPU on both sides — a ping-pong). Edge
    CPU↔accel switches at graph begin/end are allowed; mid-graph ones are not.
    """
    if not rows:
        return dict(_EMPTY_CLASSIFICATION)

    runs: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        device = str(row.get("device") or "unknown")
        op_name = str(row.get("op") or "")
        if not runs or runs[-1]["device"] != device:
            runs.append(
                {
                    "device": device,
                    "count": 1,
                    "start": i,
                    "end": i,
                    "ops_sample": [op_name] if op_name else [],
                }
            )
        else:
            runs[-1]["count"] += 1
            runs[-1]["end"] = i
            sample = runs[-1]["ops_sample"]
            if op_name and len(sample) < 6:
                sample.append(op_name)

    first_accel = next(
        (i for i, row in enumerate(rows) if row.get("device") in _ACCEL), None
    )
    last_accel = None
    for i in range(len(rows) - 1, -1, -1):
        if rows[i].get("device") in _ACCEL:
            last_accel = i
            break

    def _position(idx: int) -> str:
        if first_accel is None:
            return "cpu_only"
        if idx < first_accel:
            return "begin"
        if last_accel is not None and idx > last_accel:
            return "end"
        return "mid"

    cpu_ops: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        if row.get("device") != "CPU":
            continue
        bindings = [str(b) for b in (row.get("bindings") or [])][:8]
        supported = [str(s) for s in (row.get("supported") or [])]
        cpu_ops.append(
            {
                "index": i,
                "op": str(row.get("op") or ""),
                "role": cpu_op_role(str(row.get("op") or ""), bindings),
                "position": _position(i),
                "supported": supported,
                "bindings": bindings,
                "ane_supported": "ANE" in supported,
            }
        )

    mid_islands: list[dict[str, Any]] = []
    for run in runs:
        if run["device"] != "CPU":
            continue
        kind = _position(run["start"])
        island = {
            "kind": kind,
            "start": run["start"],
            "end": run["end"],
            "count": run["count"],
        }
        if kind == "mid":
            mid_islands.append(island)

    switches: list[dict[str, Any]] = []
    for prev, nxt in pairwise(runs):
        switches.append(
            {
                "from": prev["device"],
                "to": nxt["device"],
                "at": nxt["start"],
            }
        )

    edge_idx: set[int] = set()
    if (
        runs
        and runs[0]["device"] == "CPU"
        and switches
        and switches[0]["from"] == "CPU"
        and switches[0]["to"] in _ACCEL
    ):
        edge_idx.add(0)
    if (
        runs
        and runs[-1]["device"] == "CPU"
        and switches
        and switches[-1]["to"] == "CPU"
        and switches[-1]["from"] in _ACCEL
    ):
        edge_idx.add(len(switches) - 1)

    mid_switch_count = sum(
        1
        for i, sw in enumerate(switches)
        if i not in edge_idx and ({sw["from"], sw["to"]} & ({"CPU"} | _ACCEL))
        and (sw["from"] == "CPU" or sw["to"] == "CPU")
    )

    return {
        "cpu_ops_classified": cpu_ops,
        "device_runs": runs,
        "mid_graph_cpu_islands": mid_islands,
        "device_switches": switches,
        "mid_graph_switch_count": int(mid_switch_count),
        "begin_end_switches_only": len(mid_islands) == 0,
    }


def _unavailable(reason: str) -> dict[str, Any]:
    return {
        "available": False,
        "reason": reason,
        "by_device": {},
        "ane_ops": 0,
        "cpu_ops": 0,
        "gpu_ops": 0,
        "other_ops": 0,
        "total_ops": 0,
        "supported_ane_ops": 0,
        "top_cpu_ops": [],
        "top_ane_ops": [],
        **_EMPTY_CLASSIFICATION,
    }


def summarize_compute_plan(plan: Any, *, skip_const: bool = True) -> dict[str, Any]:
    """Count non-const ML program ops by preferred compute device."""
    structure = getattr(plan, "model_structure", None)
    program = getattr(structure, "program", None) if structure is not None else None
    if program is None:
        return _unavailable("no ML program structure on compute plan")

    functions = getattr(program, "functions", None) or {}
    main = functions.get("main") if isinstance(functions, dict) else None
    if main is None:
        return _unavailable("no main function in program")

    block = getattr(main, "block", None)
    operations = list(getattr(block, "operations", []) or [])
    by_device: Counter[str] = Counter()
    by_op_device: dict[str, Counter[str]] = {"ANE": Counter(), "CPU": Counter(), "GPU": Counter()}
    supported_ane = 0
    records: list[dict[str, Any]] = []
    getter = plan.get_compute_device_usage_for_mlprogram_operation
    for op in operations:
        op_name = getattr(op, "operator_name", None) or getattr(op, "operatorName", "")
        if skip_const and op_name == "const":
            continue
        usage = getter(op)
        preferred = getattr(usage, "preferred_compute_device", None) if usage is not None else None
        kind = device_kind(preferred)
        by_device[kind] += 1
        if kind in by_op_device:
            by_op_device[kind][str(op_name)] += 1
        supported = []
        if usage is not None:
            supported = [
                device_kind(d)
                for d in (getattr(usage, "supported_compute_devices", None) or [])
            ]
        if "ANE" in supported:
            supported_ane += 1
        records.append(
            {
                "op": str(op_name),
                "device": kind,
                "supported": supported,
                "bindings": input_binding_names(op),
            }
        )

    def _top(counter: Counter[str], n: int = 8) -> list[dict[str, Any]]:
        return [{"op": name, "count": int(c)} for name, c in counter.most_common(n)]

    ane = int(by_device.get("ANE", 0))
    cpu = int(by_device.get("CPU", 0))
    gpu = int(by_device.get("GPU", 0))
    total = int(sum(by_device.values()))
    classified = classify_device_sequence(records)
    return {
        "available": True,
        "reason": None,
        "by_device": {k: int(v) for k, v in sorted(by_device.items())},
        "ane_ops": ane,
        "cpu_ops": cpu,
        "gpu_ops": gpu,
        "other_ops": total - ane - cpu - gpu,
        "total_ops": total,
        "supported_ane_ops": int(supported_ane),
        "top_cpu_ops": _top(by_op_device["CPU"]),
        "top_ane_ops": _top(by_op_device["ANE"]),
        **classified,
    }


def placement_verdict(summary: dict[str, Any]) -> tuple[bool, str]:
    """Return (ok, reason). Fail closed if the plan is missing or has no ANE ops."""
    if not summary.get("available"):
        return False, f"compute plan unavailable: {summary.get('reason')}"
    if int(summary.get("ane_ops") or 0) <= 0:
        supported = int(summary.get("supported_ane_ops") or 0)
        return (
            False,
            (
                "CPU fallback suspected: 0 non-const ops prefer ANE "
                f"(ANE listed as supported on {supported} ops)"
            ),
        )
    return True, f"{summary['ane_ops']} non-const ops prefer ANE"
