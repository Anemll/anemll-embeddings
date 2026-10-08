"""Cosine scoring for the camera-alert rules.

Vectors are L2-normalized, so cosine is a dot product. The significant rule
is not a text match: it is how far a frame has moved from the empty-street
photo, ``1 - cosine(frame, street)``. The street itself is 0.

Once every item in a rule's scope has a score, the suggested threshold is
the midpoint between the lowest true positive and the highest negative.
That sits in the gap (UPS vs FedEx, Sparky vs the next cat, bark vs meow).
A text rule that only names a baseline frame, with no positive ids, still
uses half the smallest positive margin.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from demo.alert_catalog import (
    DOG_THRESHOLD,
    MEOW_THRESHOLD,
    PHOTO_THRESHOLD,
    SIGNIFICANT_THRESHOLD,
    UPS_THRESHOLD,
)
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


def suggest_split(scores: dict[str, float], positive_ids: list[str]) -> float | None:
    """Midpoint of the lowest true positive and the highest negative.

    Returns None until both classes have a score. The line is halfway across
    the gap, so a thin margin (bark vs meow, UPS vs FedEx) stays visible.
    """
    wanted = {str(item_id) for item_id in positive_ids}
    positives: list[float] = []
    negatives: list[float] = []
    for item_id, score in scores.items():
        value = float(score)
        if str(item_id) in wanted:
            positives.append(value)
        else:
            negatives.append(value)
    if not positives or not negatives:
        return None
    return (min(positives) + max(negatives)) / 2.0


def change_score(similarity: float, *, baseline: bool) -> float:
    """1 - cosine against the empty street. The street frame itself is 0."""
    if baseline:
        return 0.0
    return max(0.0, 1.0 - float(similarity))


def placeholder_threshold(rule: dict[str, Any]) -> float:
    """Shipped M4 lines, used when a request does not send a threshold."""
    known = {
        "significant": SIGNIFICANT_THRESHOLD,
        "ups": UPS_THRESHOLD,
        "sparky": PHOTO_THRESHOLD,
        "dog": DOG_THRESHOLD,
        "meow-query": MEOW_THRESHOLD,
    }
    rule_id = str(rule.get("id") or "")
    if rule_id in known:
        return known[rule_id]
    if rule.get("type") == "change":
        return SIGNIFICANT_THRESHOLD
    if rule.get("type") == "photo":
        return PHOTO_THRESHOLD
    return UPS_THRESHOLD


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
