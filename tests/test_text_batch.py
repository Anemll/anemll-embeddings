#!/usr/bin/env python3
"""Text buckets, packed batches and Matryoshka dims. No Core AI needed."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from api import Embedder  # noqa: E402
from api.embedder import CoreAIBackend  # noqa: E402
from api.text_batch import (  # noqa: E402
    PACK_NEG,
    bucket_feed,
    extra_text_packages,
    pack_feed,
    parse_text_function,
    pick_bucket,
    plan_cost,
    plan_packs,
    truncate_dim,
    unit_rows,
)
from tests.test_demo_coreai_masks import _Proc, _Text  # noqa: E402


def test_parse_text_function() -> None:
    assert parse_text_function("text_embeds_s32") == ("bucket", 32, 1)
    assert parse_text_function("text_pack_256x8") == ("pack", 256, 8)
    assert parse_text_function("vision_s280") is None
    assert parse_text_function("text_embeds_s32x") is None


def test_pick_bucket() -> None:
    buckets = [32, 64, 128, 256, 320]
    assert pick_bucket(1, buckets) == 32
    assert pick_bucket(32, buckets) == 32
    assert pick_bucket(33, buckets) == 64
    assert pick_bucket(320, buckets) == 320
    assert pick_bucket(321, buckets) is None
    assert pick_bucket(10, [320]) == 320


def test_plan_packs_limits_and_order() -> None:
    lengths = [30, 200, 10, 300, 60, 60, 5, 5, 5, 5, 5, 5, 5]
    packs = plan_packs(lengths, n_tokens=256, max_texts=8)
    seen = sorted(i for group in packs for i in group)
    # Index 3 (300 tokens) does not fit and is left for a bucket.
    assert seen == [i for i in range(len(lengths)) if i != 3]
    for group in packs:
        assert group == sorted(group)
        assert len(group) <= 8
        assert sum(lengths[i] for i in group) <= 256


def test_plan_packs_skips_empty_and_handles_none() -> None:
    assert plan_packs([], n_tokens=256, max_texts=8) == []
    assert plan_packs([0, 4], n_tokens=256, max_texts=8) == [[1]]


def test_plan_cost_prefers_the_cheaper_plan() -> None:
    bucket_ms = {32: 3.0, 320: 35.0}
    pack = (256, 8, 15.0)
    # Eight short texts: one 15 ms pack beats eight 3 ms calls.
    cost_pack, groups, singles = plan_cost([10] * 8, bucket_ms, pack)
    cost_one, _, one_singles = plan_cost([10] * 8, bucket_ms, None)
    assert groups == [list(range(8))] and singles == []
    assert one_singles == list(range(8)) and cost_pack < cost_one
    # Two short texts: two s32 calls beat one pack.
    two_pack, _, _ = plan_cost([10, 10], bucket_ms, pack)
    two_one, _, _ = plan_cost([10, 10], bucket_ms, None)
    assert two_one < two_pack
    # Too long for the pack and every bucket: counted at the largest bucket.
    _cost, groups, singles = plan_cost([400, 10], bucket_ms, pack)
    assert groups == [] and singles == [0, 1]


def test_pack_feed_is_ane_legal() -> None:
    rows = [np.full((3, 512), 0.5, np.float32), np.full((5, 512), -1.0, np.float32)]
    feed = pack_feed(rows, n_tokens=16, max_texts=4)
    assert set(feed) == {"inputs_embeds", "attention_bias", "positions", "pool"}
    for value in feed.values():
        assert value.dtype == np.float16
        assert np.isfinite(value).all()
    bias = feed["attention_bias"][0, 0]
    assert bias.shape == (16, 16)
    assert (bias[:3, :3] == 0).all() and (bias[3:8, 3:8] == 0).all()
    assert (bias[:3, 3:] == np.float16(PACK_NEG)).all()
    assert (bias[3:8, :3] == np.float16(PACK_NEG)).all()
    # Padding rows see only themselves; no row is fully masked.
    assert all((bias[i] == 0).any() for i in range(16))
    assert feed["positions"][:, 0].tolist()[:9] == [0, 1, 2, 0, 1, 2, 3, 4, 0]
    pool = feed["pool"].astype(np.float32)
    assert np.allclose(pool[0, :3], 1 / 3, atol=1e-3) and pool[0, 3:].sum() == 0
    assert np.allclose(pool[1, 3:8], 1 / 5, atol=1e-3)
    assert pool[2:].sum() == 0


def test_pack_feed_rejects_overflow() -> None:
    rows = [np.zeros((10, 512), np.float32)] * 2
    with pytest.raises(ValueError):
        pack_feed(rows, n_tokens=16, max_texts=4)
    with pytest.raises(ValueError):
        pack_feed(rows, n_tokens=64, max_texts=1)


def test_bucket_feed() -> None:
    feed = bucket_feed(np.ones((5, 512), np.float32), 32)
    assert feed["inputs_embeds"].shape == (1, 32, 512)
    assert feed["attention_mask"].dtype == np.float16
    assert feed["attention_mask"].sum() == 5
    with pytest.raises(ValueError):
        bucket_feed(np.ones((40, 512), np.float32), 32)


def test_truncate_dim() -> None:
    rng = np.random.default_rng(0)
    vecs = unit_rows(rng.standard_normal((3, 768)))
    out = truncate_dim(vecs, 256)
    assert out.shape == (3, 256)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
    assert np.allclose(out[0], vecs[0, :256] / np.linalg.norm(vecs[0, :256]), atol=1e-6)
    assert truncate_dim(vecs[0], 128).shape == (128,)
    assert truncate_dim(vecs, None) is not None and truncate_dim(vecs, 768).shape == (3, 768)
    with pytest.raises(ValueError):
        truncate_dim(vecs, 100)


def test_extra_text_packages(tmp_path: Path) -> None:
    for name in ("text_embeds_s320", "text_embeds_s32", "text_pack_256x8", "vision_s280"):
        (tmp_path / f"{name}.aimodel").mkdir()
    found = [p.name for p in extra_text_packages(tmp_path)]
    assert found == ["text_embeds_s32.aimodel", "text_pack_256x8.aimodel"]
    (tmp_path / "text_buckets.aimodel").mkdir()
    assert [p.name for p in extra_text_packages(tmp_path)] == ["text_buckets.aimodel"]


def test_mock_embed_texts_order_and_dim() -> None:
    embedder = Embedder(backend="mock")
    texts = ["red fox", "blue whale", "red fox"]
    out = embedder.embed_texts(texts, role="document")
    assert out.shape == (3, 768) and out.dtype == np.float32
    assert np.allclose(out[0], out[2])
    single = embedder.embed_text("blue whale", role="document")
    assert np.allclose(out[1], single)
    small = embedder.embed_texts(texts, role="document", dim=128)
    assert small.shape == (3, 128)
    assert np.allclose(np.linalg.norm(small, axis=1), 1.0, atol=1e-5)
    assert embedder.embed_text("red fox", dim=256).shape == (256,)
    assert embedder.embed_texts([]).shape == (0, 768)
    assert embedder.last_batch is not None and embedder.last_batch.texts == 3


class _MeanRunner:
    """Fake towers: embedding = mean token row (first 512 dims), unnormalized.

    A bucket and a packed forward give the same vector for the same text,
    so the test checks that the host routes and orders rows correctly.
    """

    def __init__(self, extras: bool = True) -> None:
        self.calls: list[str] = []
        self.extras = extras

    def startup(self) -> dict:
        towers = {
            "text_embeds_s320": {"loaded": True, "placement": "fullyOnANE"},
            "vision_s280": {"loaded": True, "placement": "fullyOnANE"},
            "audio_s280": {"loaded": True, "placement": "fullyOnANE"},
        }
        if self.extras:
            for name in ("text_embeds_s32", "text_embeds_s64", "text_pack_64x4"):
                towers[name] = {"loaded": True, "placement": "fullyOnANE", "extra": True}
            towers["text_embeds_s128"] = {"loaded": True, "placement": "GPU", "extra": True}
        return {"placement": "fullyOnANE", "towers": towers}

    def forward(self, tower: str, feed: dict[str, np.ndarray]):
        self.calls.append(tower)
        embeds = feed["inputs_embeds"][0].astype(np.float32)
        if "pool" in feed:
            assert tower.startswith("text_pack_")
            pooled = feed["pool"].astype(np.float32) @ embeds
        else:
            mask = feed["attention_mask"][0].astype(np.float32)
            assert embeds.shape[0] == {"text": 320}.get(tower, None) or tower.endswith(
                f"s{embeds.shape[0]}"
            )
            pooled = (mask @ embeds / mask.sum())[None, :]
        out = np.zeros((pooled.shape[0], 768), dtype=np.float32)
        out[:, :512] = pooled
        return out, 1.0

    def close(self) -> None:
        return None


def _backend(runner) -> CoreAIBackend:
    torch.manual_seed(0)
    backend = CoreAIBackend(
        artifacts=None,
        model_path=None,
        coreai_python=None,
        compute="ane",
        runner=runner,
        processor=_Proc(),
        text_model=_Text(),
        prompts={},
    )
    backend.warmup()
    return backend


def test_coreai_embed_texts_packs_and_buckets() -> None:
    runner = _MeanRunner()
    backend = _backend(runner)
    towers = backend.text_towers()
    # The GPU-placed s128 bucket is ignored.
    assert sorted(towers["buckets"]) == [32, 64, 320]
    assert towers["packs"] == [(64, 4, "text_pack_64x4")]
    texts = ["a b", "c d e", "long " * 50, "f", "g h i j k", "z " * 90, "q"]
    vecs, stats = backend.embed_texts(texts, role="document", pack=True)
    assert vecs.shape == (len(texts), 768)
    assert stats.by_tower.get("text_pack_64x4", 0) >= 1
    # "z " * 90 is 92 tokens: too long for the pack and s64, so it runs on s320.
    assert stats.by_tower.get("text", 0) == 1
    runner.calls.clear()
    single, single_stats = backend.embed_texts(texts, role="document", pack=False)
    assert "text_pack_64x4" not in single_stats.by_tower
    assert np.allclose(vecs, single, atol=2e-3)
    for i, text in enumerate(texts):
        one = backend.embed_text(text, role="document").vector
        assert np.allclose(vecs[i], one, atol=2e-3)


def test_coreai_embed_texts_without_extras_uses_s320() -> None:
    runner = _MeanRunner(extras=False)
    backend = _backend(runner)
    vecs, stats = backend.embed_texts(["one", "two words"], role="query")
    assert vecs.shape == (2, 768)
    assert stats.by_tower == {"text": 2}
