"""Cosine scoring for the camera-alert rules.

Vectors are L2-normalized, so cosine is a dot product. A text rule with a
baseline frame (the empty street) reports the margin over that frame: the
street itself is 0 and stays quiet while the threshold is positive. Other
rules report cosine directly.

Once every item in a rule's scope has a score, the suggested threshold is:

- baseline rules: half the smallest positive margin
- every other rule: the midpoint between the best score and the second best
"""

from __future__ import annotations

from typing import Any

import numpy as np

from demo.alert_catalog import MARGIN_PLACEHOLDER, PHOTO_PLACEHOLDER, TEXT_PLACEHOLDER
from demo.types import DIM


def l2(vector: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    if arr.shape != (DIM,):
        raise ValueError(f"expected a {DIM}-d vector, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("embedding is not finite")
    norm = float(np.linalg.norm(arr))
    if norm < 1e-8:
        raise ValueError("embedding norm is zero")
    return (arr / norm).astype(np.float32, copy=False)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(np.asarray(left, dtype=np.float32), np.asarray(right, dtype=np.float32)))


def average_unit(vectors: list[np.ndarray]) -> np.ndarray:
    if not vectors:
        raise ValueError("photo rule needs at least one reference photo")
    if len(vectors) > 3:
        raise ValueError("a photo rule accepts at most 3 reference photos")
    mean = np.mean(np.stack([l2(vec) for vec in vectors], axis=0), axis=0)
    return l2(mean)


def suggest_midpoint(scores: list[float]) -> float | None:
    """Midpoint of the best and second-best. Only the leader sits at or above it."""
    if len(scores) < 2:
        return None
    ordered = sorted((float(score) for score in scores), reverse=True)
    return (ordered[0] + ordered[1]) / 2.0


def suggest_margin(margins: list[float]) -> float | None:
    """Half the smallest positive margin, so a 0 baseline stays below the line."""
    positive = [float(score) for score in margins if float(score) > 1e-6]
    if not positive:
        return None
    return min(positive) / 2.0


def placeholder_threshold(rule: dict[str, Any]) -> float:
    if rule.get("baseline_id"):
        return MARGIN_PLACEHOLDER
    if rule.get("type") == "photo":
        return PHOTO_PLACEHOLDER
    return TEXT_PLACEHOLDER


def rule_value(raw: float, baseline_raw: float | None) -> float:
    if baseline_raw is None:
        return float(raw)
    return float(raw) - float(baseline_raw)


def fires(score: float | None, threshold: float) -> bool:
    if score is None:
        return False
    return float(score) >= float(threshold)


def scope_ids(items: list[dict[str, Any]], scope: str) -> list[str]:
    if scope not in {"image", "audio", "all"}:
        raise ValueError(f"scope must be image, audio, or all, got {scope!r}")
    chosen = []
    for item in items:
        if item.get("kind") == "reference":
            continue
        modality = item.get("modality")
        if scope == "all" or modality == scope:
            chosen.append(str(item["id"]))
    return chosen
