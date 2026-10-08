#!/usr/bin/env python3
"""Hardware-gated placement and parity checks (real Core AI packages).

Skipped unless ``ANEMLL_HW_TESTS=1``. Needs the downloaded packages and a Core
AI interpreter (see README "Core AI runtime"). Run on each target Mac::

    ANEMLL_HW_TESTS=1 python -m pytest tests/test_hardware_ane.py -q -s

By default every tower must load and the audio tower must be fully on the
Neural Engine (true on M3 Ultra, M4 Pro, and M5). On Macs where vision and text
are expected on the ANE too (M3/M4, macOS 27.0) also set
``ANEMLL_HW_EXPECT_ANE=1``; that is the strict ``warmup.py --require-ane`` gate.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

pytestmark = pytest.mark.skipif(
    os.environ.get("ANEMLL_HW_TESTS") != "1",
    reason="hardware test: set ANEMLL_HW_TESTS=1 on a Mac with the Core AI packages",
)

FIXTURES = REPO_ROOT / "tests" / "fixtures"
MIN_TEXT_COSINE = 0.999


@pytest.fixture(scope="module")
def embedder():
    from api import Embedder

    emb = Embedder(compute="ane")
    try:
        yield emb
    finally:
        emb.close()


def test_placement(embedder) -> None:
    report = embedder.warmup()
    towers = {row["name"]: row for row in report.get("towers") or []}
    assert len(towers) == 3, report
    print("placement:", json.dumps({k: v.get("placement") for k, v in towers.items()}))
    audio = [info for name, info in towers.items() if name.startswith("audio")]
    assert audio and audio[0].get("placement") == "fullyOnANE", report
    if os.environ.get("ANEMLL_HW_EXPECT_ANE") == "1":
        off = [name for name, info in towers.items() if info.get("placement") != "fullyOnANE"]
        assert not off, f"not fully on the Neural Engine: {off}"


def test_text_parity_with_reference(embedder) -> None:
    prompts = json.loads((FIXTURES / "prompts.json").read_text())["prompts"]
    ref = np.load(FIXTURES / "embeddings.npy")
    worst = 1.0
    for row, prompt in zip(ref, prompts[:4]):
        vec = embedder.embed_text(prompt["text"], role=prompt.get("prompt_name") or "query")
        cos = float(np.dot(vec, row) / (np.linalg.norm(vec) * np.linalg.norm(row)))
        worst = min(worst, cos)
    print(f"min text cosine vs reference fixtures: {worst:.6f}")
    assert worst >= MIN_TEXT_COSINE
