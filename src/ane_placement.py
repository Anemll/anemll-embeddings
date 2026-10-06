"""T7 helpers: classify Core ML compute-plan devices (no model load)."""

from __future__ import annotations

from collections import Counter
from typing import Any

from coremltools.models.compute_device import (
    MLCPUComputeDevice,
    MLGPUComputeDevice,
    MLNeuralEngineComputeDevice,
)


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


def summarize_compute_plan(plan: Any, *, skip_const: bool = True) -> dict[str, Any]:
    """Count non-const ML program ops by preferred compute device."""
    structure = getattr(plan, "model_structure", None)
    program = getattr(structure, "program", None) if structure is not None else None
    if program is None:
        return {
            "available": False,
            "reason": "no ML program structure on compute plan",
            "by_device": {},
            "ane_ops": 0,
            "cpu_ops": 0,
            "gpu_ops": 0,
            "other_ops": 0,
            "total_ops": 0,
            "top_cpu_ops": [],
            "top_ane_ops": [],
        }

    functions = getattr(program, "functions", None) or {}
    main = functions.get("main") if isinstance(functions, dict) else None
    if main is None:
        return {
            "available": False,
            "reason": "no main function in program",
            "by_device": {},
            "ane_ops": 0,
            "cpu_ops": 0,
            "gpu_ops": 0,
            "other_ops": 0,
            "total_ops": 0,
            "top_cpu_ops": [],
            "top_ane_ops": [],
        }

    block = getattr(main, "block", None)
    operations = list(getattr(block, "operations", []) or [])
    by_device: Counter[str] = Counter()
    by_op_device: dict[str, Counter[str]] = {"ANE": Counter(), "CPU": Counter(), "GPU": Counter()}
    getter = getattr(plan, "get_compute_device_usage_for_mlprogram_operation")
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

    def _top(counter: Counter[str], n: int = 8) -> list[dict[str, Any]]:
        return [{"op": name, "count": int(c)} for name, c in counter.most_common(n)]

    ane = int(by_device.get("ANE", 0))
    cpu = int(by_device.get("CPU", 0))
    gpu = int(by_device.get("GPU", 0))
    total = int(sum(by_device.values()))
    return {
        "available": True,
        "reason": None,
        "by_device": {k: int(v) for k, v in sorted(by_device.items())},
        "ane_ops": ane,
        "cpu_ops": cpu,
        "gpu_ops": gpu,
        "other_ops": total - ane - cpu - gpu,
        "total_ops": total,
        "top_cpu_ops": _top(by_op_device["CPU"]),
        "top_ane_ops": _top(by_op_device["ANE"]),
    }


def placement_verdict(summary: dict[str, Any]) -> tuple[bool, str]:
    """Return (ok, reason). Fail closed if the plan is missing or has no ANE ops."""
    if not summary.get("available"):
        return False, f"compute plan unavailable: {summary.get('reason')}"
    if int(summary.get("ane_ops") or 0) <= 0:
        return False, "CPU fallback suspected: 0 non-const ops prefer ANE"
    return True, f"{summary['ane_ops']} non-const ops prefer ANE"
