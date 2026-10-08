"""T6 metrics: cosine, rel-L2, pairwise agreement (no Core ML import)."""

from __future__ import annotations

from typing import Any

import numpy as np

# PLAN starting gate for an FP path on short prompts.
COSINE_GATE = 0.999
# Unit-vector rel-L2 ≈ sqrt(2 - 2*cos). cos=0.999 → ~0.0447.
REL_L2_GATE = 0.05
PAIR_DELTA_GATE = 0.01

# Ranking-relevant pairs from tests/fixtures/prompts.json.
FIXTURE_PAIRS: tuple[tuple[str, str], ...] = (
    ("sq_short", "doc_short"),
    ("sq_medium", "doc_medium"),
    ("code_q", "code_doc"),
    ("sts_a", "sts_b"),
)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a64 = np.asarray(a, dtype=np.float64).ravel()
    b64 = np.asarray(b, dtype=np.float64).ravel()
    denom = float(np.linalg.norm(a64) * np.linalg.norm(b64))
    if denom == 0:
        return float("nan")
    return float(np.dot(a64, b64) / denom)


def rel_l2(pred: np.ndarray, ref: np.ndarray) -> float:
    """||pred - ref||_2 / ||ref||_2 (forge: cosine hides magnitude)."""
    p = np.asarray(pred, dtype=np.float64).ravel()
    r = np.asarray(ref, dtype=np.float64).ravel()
    ref_n = float(np.linalg.norm(r))
    if ref_n == 0:
        return float("nan")
    return float(np.linalg.norm(p - r) / ref_n)


def l2_norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(x, dtype=np.float64).ravel()))


def finite_counts(x: np.ndarray) -> dict[str, int]:
    arr = np.asarray(x)
    return {
        "n": int(arr.size),
        "finite": int(np.isfinite(arr).sum()),
        "nan": int(np.isnan(arr).sum()),
        "inf": int(np.isinf(arr).sum()),
    }


def pairwise_cosines(
    vectors: np.ndarray,
    ids: list[str],
    pairs: tuple[tuple[str, str], ...] = FIXTURE_PAIRS,
) -> list[dict[str, Any]]:
    index = {name: i for i, name in enumerate(ids)}
    rows: list[dict[str, Any]] = []
    for left, right in pairs:
        if left not in index or right not in index:
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "cosine": None,
                    "skipped": True,
                }
            )
            continue
        rows.append(
            {
                "left": left,
                "right": right,
                "cosine": cosine(vectors[index[left]], vectors[index[right]]),
                "skipped": False,
            }
        )
    return rows


def pair_deltas(
    st_pairs: list[dict[str, Any]],
    cml_pairs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for st, cml in zip(st_pairs, cml_pairs, strict=True):
        row = {
            "left": st["left"],
            "right": st["right"],
            "st_cosine": st.get("cosine"),
            "coreml_cosine": cml.get("cosine"),
            "abs_delta": None,
            "skipped": bool(st.get("skipped") or cml.get("skipped")),
        }
        if (
            not row["skipped"]
            and st.get("cosine") is not None
            and cml.get("cosine") is not None
        ):
            row["abs_delta"] = abs(float(st["cosine"]) - float(cml["cosine"]))
        out.append(row)
    return out
