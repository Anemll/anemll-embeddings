"""Host side of short-text buckets and packed batches (numpy only).

The Core AI text towers have fixed shapes. ``text_embeds_s320`` covers any
text up to 320 tokens; the optional bucket towers (``text_embeds_s32`` ...
``text_embeds_s256``) do the same work on a shorter sequence, and the
optional packed tower (``text_pack_256x8``) runs up to 8 short texts in one
forward. This module picks the tower and builds the feeds. It has no
torch or Core AI dependency, so it is tested on any machine.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from api.types import DIM

PACK_NEG = -1.0e4
# The model card lists these Matryoshka (MRL) sizes: keep the first N
# values of the 768-d vector and L2-normalize again.
MATRYOSHKA_DIMS = (128, 256, 512, 768)

_BUCKET_RE = re.compile(r"^text_embeds_s(\d+)$")
_PACK_RE = re.compile(r"^text_pack_(\d+)x(\d+)$")


def parse_text_function(name: str) -> tuple[str, int, int] | None:
    """``("bucket", S, 1)`` or ``("pack", N, T)`` for a text function name."""
    match = _BUCKET_RE.match(name)
    if match:
        return "bucket", int(match.group(1)), 1
    match = _PACK_RE.match(name)
    if match:
        return "pack", int(match.group(1)), int(match.group(2))
    return None


def extra_text_packages(coreai_dir: Path) -> list[Path]:
    """Optional bucket / pack packages next to ``text_embeds_s320.aimodel``.

    A combined ``text_buckets.aimodel`` (one package, several functions)
    wins over the one-function packages so the same weights are not loaded
    twice.
    """
    root = Path(coreai_dir)
    combined = root / "text_buckets.aimodel"
    if combined.is_dir():
        return [combined]
    found = []
    for path in sorted(root.glob("text_*.aimodel")):
        info = parse_text_function(path.name[: -len(".aimodel")])
        if info is None or (info[0] == "bucket" and info[1] == 320):
            continue
        found.append(path)
    return found


def pick_bucket(n_tokens: int, buckets: Iterable[int]) -> int | None:
    """Smallest bucket that holds ``n_tokens``, or ``None`` if none does."""
    fits = [int(b) for b in buckets if int(b) >= int(n_tokens)]
    return min(fits) if fits else None


def plan_packs(lengths: Sequence[int], *, n_tokens: int, max_texts: int) -> list[list[int]]:
    """Group text indices into packs of at most ``n_tokens`` / ``max_texts``.

    First-fit decreasing: long texts are placed first, each into the first
    pack with room. Texts longer than ``n_tokens`` are left out (the caller
    runs them on a bucket). Each returned group keeps ascending index order.
    """
    order = sorted(
        (i for i, n in enumerate(lengths) if 0 < int(n) <= int(n_tokens)),
        key=lambda i: (-int(lengths[i]), i),
    )
    packs: list[list[int]] = []
    used: list[int] = []
    for idx in order:
        need = int(lengths[idx])
        for slot, group in enumerate(packs):
            if len(group) < max_texts and used[slot] + need <= n_tokens:
                group.append(idx)
                used[slot] += need
                break
        else:
            packs.append([idx])
            used.append(need)
    return [sorted(group) for group in packs]


# Fixed cost of one worker round trip (IPC + feed copies), measured on an
# M4 Pro at 1.3 ms for a bucket and 2.8 ms for a packed feed.
CALL_OVERHEAD_MS = 1.5


def plan_cost(
    lengths: Sequence[int],
    bucket_ms: dict[int, float],
    pack: tuple[int, int, float] | None,
    *,
    overhead_ms: float = CALL_OVERHEAD_MS,
) -> tuple[float, list[list[int]], list[int]]:
    """Estimated time to embed ``lengths`` with one plan.

    ``bucket_ms`` maps bucket S to its warm latency, ``pack`` is
    ``(N, T, latency)`` or ``None`` for one text per call. Returns
    ``(cost, packed_groups, single_indices)``; a group of one runs as a
    single. Texts longer than every bucket count at the largest one.
    """
    sizes = sorted(bucket_ms)
    groups: list[list[int]] = []
    singles: list[int] = []
    cost = 0.0
    if pack is not None:
        n_tokens, max_texts, pack_ms = pack
        for group in plan_packs(lengths, n_tokens=n_tokens, max_texts=max_texts):
            if len(group) >= 2:
                groups.append(group)
                cost += pack_ms + overhead_ms
            else:
                singles.extend(group)
        placed = {i for group in groups for i in group}
        singles = sorted(set(singles) | {i for i in range(len(lengths)) if i not in placed})
    else:
        singles = list(range(len(lengths)))
    for idx in singles:
        size = pick_bucket(lengths[idx], sizes) or sizes[-1]
        cost += bucket_ms[size] + overhead_ms
    return cost, groups, singles


def default_tower_ms(size: int) -> float:
    """Rough warm latency when the worker did not report one (M4 Pro scale)."""
    return 1.0 + 0.06 * float(size)


def bucket_feed(rows: np.ndarray, seq_len: int) -> dict[str, np.ndarray]:
    """``[L, 512]`` token rows → right-padded f16 feed for ``text_embeds_s{S}``."""
    rows = np.asarray(rows, dtype=np.float32)
    length, hidden = int(rows.shape[0]), int(rows.shape[1])
    if length > seq_len:
        raise ValueError(f"{length} tokens do not fit S={seq_len}")
    embeds = np.zeros((1, seq_len, hidden), dtype=np.float16)
    embeds[0, :length] = rows
    mask = np.zeros((1, seq_len), dtype=np.float16)
    mask[0, :length] = 1
    return {"inputs_embeds": embeds, "attention_mask": mask}


def pack_feed(
    rows: Sequence[np.ndarray], *, n_tokens: int, max_texts: int, neg: float = PACK_NEG
) -> dict[str, np.ndarray]:
    """Lay texts end to end in one ``[1, N, 512]`` packed feed (all f16).

    ``attention_bias`` is 0 inside each text's block and ``neg`` elsewhere.
    Padding tokens attend only to themselves, so no row is fully masked.
    ``positions`` restart at 0 per text. ``pool`` row ``t`` averages text t.
    """
    if not rows:
        raise ValueError("pack_feed needs at least one text")
    if len(rows) > max_texts:
        raise ValueError(f"{len(rows)} texts > max_texts={max_texts}")
    hidden = int(np.asarray(rows[0]).shape[1])
    embeds = np.zeros((1, n_tokens, hidden), dtype=np.float16)
    bias = np.full((1, 1, n_tokens, n_tokens), neg, dtype=np.float16)
    positions = np.zeros((n_tokens, 1), dtype=np.float16)
    pool = np.zeros((max_texts, n_tokens), dtype=np.float16)
    start = 0
    for slot, row in enumerate(rows):
        row = np.asarray(row, dtype=np.float32)
        length = int(row.shape[0])
        end = start + length
        if length < 1 or end > n_tokens:
            raise ValueError(f"texts need {end} tokens > N={n_tokens}")
        embeds[0, start:end] = row
        bias[0, 0, start:end, start:end] = 0
        positions[start:end, 0] = np.arange(length, dtype=np.float16)
        pool[slot, start:end] = 1.0 / float(length)
        start = end
    pad = np.arange(start, n_tokens)
    bias[0, 0, pad, pad] = 0
    return {
        "inputs_embeds": embeds,
        "attention_bias": bias,
        "positions": positions,
        "pool": pool,
    }


def unit_rows(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize each row in float32; raise on a zero or non-finite row."""
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    if not np.isfinite(arr).all() or (norms < 1e-8).any():
        raise RuntimeError("bad embedding row (zero or non-finite)")
    return (arr / norms).astype(np.float32, copy=False)


def truncate_dim(vectors: np.ndarray, dim: int | None) -> np.ndarray:
    """Matryoshka truncation: keep the first ``dim`` values, re-normalize.

    ``dim`` must be one of :data:`MATRYOSHKA_DIMS` (the sizes the model was
    trained for). ``None`` or 768 returns the input unchanged.
    """
    if dim is None or int(dim) == DIM:
        return np.asarray(vectors, dtype=np.float32)
    if int(dim) not in MATRYOSHKA_DIMS:
        raise ValueError(f"dim must be one of {MATRYOSHKA_DIMS}, got {dim}")
    arr = np.asarray(vectors, dtype=np.float32)
    return unit_rows(arr[..., : int(dim)]).reshape(*arr.shape[:-1], int(dim))


@dataclass
class TextBatchStats:
    """What one ``embed_texts`` call ran (for logs and benchmarks)."""

    texts: int = 0
    calls: int = 0
    tower_ms: float = 0.0
    wall_ms: float = 0.0
    by_tower: dict[str, int] = field(default_factory=dict)
    truncated: int = 0

    def count(self, tower: str, ms: float) -> None:
        self.calls += 1
        self.tower_ms += float(ms)
        self.by_tower[tower] = self.by_tower.get(tower, 0) + 1

    def as_dict(self) -> dict:
        return {
            "texts": self.texts,
            "calls": self.calls,
            "tower_ms": round(self.tower_ms, 3),
            "wall_ms": round(self.wall_ms, 3),
            "by_tower": dict(self.by_tower),
            "truncated": self.truncated,
        }
