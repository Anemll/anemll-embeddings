#!/usr/bin/env python3
"""Unit checks for Core AI smoke helpers (no .aimodel, no coreai runtime)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.coreai_smoke import (  # noqa: E402
    TEXT_SEQ_LEN,
    classify_device_runs,
    dummy_numpy_inputs,
    extract_devices_from_debug,
    output_is_finite,
    parse_mpsgraph_regions,
    percentile_ms,
    placement_from_cache_manifest,
    tower_smoke_io,
)


def _fail(msg: str) -> None:
    raise AssertionError(msg)


def test_dummy_shapes() -> None:
    v = dummy_numpy_inputs("vision")
    if tuple(v["pixel_values"].shape) != (1, 2520, 768):
        _fail(str(v["pixel_values"].shape))
    if tuple(v["pixel_position_ids"].shape) != (1, 2520, 2):
        _fail(str(v["pixel_position_ids"].shape))
    if v["pixel_position_ids"].dtype != np.float16:
        _fail(str(v["pixel_position_ids"].dtype))
    if v["pixel_values"].dtype != np.float16:
        _fail(str(v["pixel_values"].dtype))
    a = dummy_numpy_inputs("audio")
    if tuple(a["input_features"].shape) != (1, 280, 128):
        _fail(str(a["input_features"].shape))
    t = dummy_numpy_inputs("text")
    if tuple(t["input_ids"].shape) != (1, TEXT_SEQ_LEN):
        _fail(str(t["input_ids"].shape))
    if t["input_ids"].dtype != np.int32 or t["attention_mask"].dtype != np.int32:
        _fail(f"{t['input_ids'].dtype} {t['attention_mask'].dtype}")
    if a["input_features_mask"].dtype != np.float16:
        _fail(str(a["input_features_mask"].dtype))
    e = dummy_numpy_inputs("text_embeds")
    if tuple(e["inputs_embeds"].shape) != (1, 320, 512):
        _fail(str(e["inputs_embeds"].shape))
    if e["inputs_embeds"].dtype != np.float16 or e["attention_mask"].dtype != np.float16:
        _fail(f"{e['inputs_embeds'].dtype} {e['attention_mask'].dtype}")


def test_io_text_is_s128() -> None:
    spec = tower_smoke_io("text")
    if spec["inputs"]["input_ids"] != [1, 128]:
        _fail(str(spec))
    if spec["outputs"]["embedding"] != [1, 768]:
        _fail(str(spec))


def test_finite_helper() -> None:
    if not output_is_finite(np.zeros((1, 3), dtype=np.float16)):
        _fail("zeros should be finite")
    if output_is_finite(np.array([np.nan], dtype=np.float32)):
        _fail("nan should fail")
    if not output_is_finite(np.arange(4, dtype=np.int32)):
        _fail("ints are finite")


def test_begin_cpu_ok() -> None:
    got = classify_device_runs(["CPU", "CPU", "ANE", "ANE"])
    if not got["begin_end_switches_only"] or got["mid_graph_cpu_islands"]:
        _fail(str(got))


def test_end_cpu_ok() -> None:
    got = classify_device_runs(["ANE", "ANE", "CPU"])
    if not got["begin_end_switches_only"] or got["mid_graph_cpu_islands"]:
        _fail(str(got))


def test_mid_cpu_flagged() -> None:
    got = classify_device_runs(["ANE", "CPU", "ANE"])
    if got["begin_end_switches_only"] or not got["mid_graph_cpu_islands"]:
        _fail(str(got))


def test_cache_manifest_labels() -> None:
    if placement_from_cache_manifest(b"xx mps.fullyPlacedOnANE yy") != "ANE":
        _fail("ANE")
    if placement_from_cache_manifest(b"_ANE_region_ gpu") != "ANE+GPU":
        _fail("mixed")
    if placement_from_cache_manifest(b"gpu only") != "GPU":
        _fail("gpu")


def test_debug_extract() -> None:
    blob = {"ops": [{"preferred_compute_device": "NeuralEngine"}, {"device": "CPU"}]}
    got = extract_devices_from_debug(blob)
    if got != ["ANE", "CPU"]:
        _fail(str(got))


def test_percentile_ms() -> None:
    got = percentile_ms([0.010, 0.020, 0.030, 0.040, 0.050], 50.0)
    if abs(got - 30.0) > 1e-6:
        _fail(str(got))
    if abs(percentile_ms([0.010], 90.0) - 10.0) > 1e-6:
        _fail("single")


def test_parse_mpsgraph_regions() -> None:
    blob = (
        b"foo_ANE_region_0_0 #placement.region_type<ANE> "
        b"foo_GPU_region_0 foo_GPU_region_0_nested_1 "
        b"#placement.region_type<GPU> ane_io_cast mpsx.parallelGPURegion"
    )
    got = parse_mpsgraph_regions(blob)
    if got["gpu_top_regions"] != ["foo_GPU_region_0"]:
        _fail(str(got["gpu_top_regions"]))
    if "ane_io_cast" not in got["named_ops_present"]:
        _fail(str(got["named_ops_present"]))
    if not got["has_gpu_region_type"]:
        _fail("gpu type")


def main() -> int:
    tests = [
        test_dummy_shapes,
        test_io_text_is_s128,
        test_finite_helper,
        test_begin_cpu_ok,
        test_end_cpu_ok,
        test_mid_cpu_flagged,
        test_cache_manifest_labels,
        test_debug_extract,
        test_percentile_ms,
        test_parse_mpsgraph_regions,
    ]
    for fn in tests:
        fn()
        print(f"OK {fn.__name__}")
    print(f"OK: {len(tests)} coreai-smoke checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
