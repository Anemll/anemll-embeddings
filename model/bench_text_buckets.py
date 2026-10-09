#!/usr/bin/env python3
"""Placement, cosine and speed of the text buckets and the packed tower.

Runs the real host path (``api.embedder.CoreAIBackend`` and its Core AI
worker). Reference for cosine is the shipped ``text_embeds_s320`` tower.

    CFFIXED_USER_HOME=/tmp/fresh python model/bench_text_buckets.py \\
        --artifacts DIR --model CKPT --lines-from api --json out.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from api.coreai_host import TEXT_EMBEDS_S  # noqa: E402
from api.embedder import CoreAIBackend, worker_python  # noqa: E402
from api.text_batch import bucket_feed, pack_feed, unit_rows  # noqa: E402

FIXTURES = REPO_ROOT / "tests" / "fixtures" / "prompts.json"


def _rows(backend: CoreAIBackend, text: str, prompt_name: str | None) -> np.ndarray:
    prefix = backend._prompts.get(prompt_name or "", "")
    ids, _ = backend._tokenize(prefix + text)
    ids = ids[:, :TEXT_EMBEDS_S]
    embeds = backend._scatter(ids, image_soft=None, audio_soft=None)
    return embeds[0].detach().cpu().numpy().astype(np.float32)


def _p(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def grep_lines(root: Path, limit: int) -> list[str]:
    out: list[str] = []
    for path in sorted(root.rglob("*.py")):
        for line in path.read_text(errors="replace").splitlines():
            text = line.strip()
            if len(text) >= 8:
                out.append(text)
            if len(out) >= limit:
                return out
    return out


POSTS = (
    "Just shipped a new release, the ANE path is finally fully on device.",
    "Anyone else seeing the M5 run hot when compiling large models?",
    "Coffee first, then debugging the tokenizer.",
    "Hot take: fixed shapes are a feature, not a bug.",
    "Looking for a good on-device embedding model for code search.",
    "The northern lights were visible from Seattle last night!",
    "Reminder: mean pooling needs the attention mask.",
    "Benchmarks without warmup are just noise.",
    "What is the best way to batch short queries on the Neural Engine?",
    "Our grep tool now understands what you mean, not just what you type.",
    "Cats sleeping on keyboards is a universal constant.",
    "Is 768 dims overkill for a small index? Try Matryoshka 256.",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--coreai-python", type=Path, default=None)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--lines-from", type=Path, default=REPO_ROOT / "api")
    parser.add_argument("--lines", type=int, default=512)
    parser.add_argument("--repeat", type=int, default=3, help="best of N for throughput")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    report: dict = {}
    backend = CoreAIBackend(
        artifacts=args.artifacts,
        model_path=args.model,
        coreai_python=worker_python(args.coreai_python),
        compute="ane",
    )
    t0 = time.perf_counter()
    health = backend.warmup()
    report["warmup_s"] = round(time.perf_counter() - t0, 1)
    report["towers"] = {row["name"]: row.get("placement") for row in health["towers"]}
    report["text_towers"] = backend.text_towers()
    print(json.dumps({"towers": report["towers"], "text": report["text_towers"]}, default=str))
    runner = backend._runner
    buckets = report["text_towers"]["buckets"]
    packs = report["text_towers"]["packs"]

    # 1) Cosine vs s320 on the fixtures: every bucket that fits, and packed.
    prompts = json.loads(FIXTURES.read_text())["prompts"]
    fixture_rows = [_rows(backend, p["text"], p.get("prompt_name")) for p in prompts]
    ref = []
    for rows in fixture_rows:
        raw, _ = runner.forward("text", bucket_feed(rows, TEXT_EMBEDS_S))
        ref.append(unit_rows(raw)[0])
    cos: dict[str, list[float]] = {}
    for size, tower in buckets.items():
        if size == TEXT_EMBEDS_S:
            continue
        vals = []
        for rows, r in zip(fixture_rows, ref, strict=True):
            if rows.shape[0] <= size:
                raw, _ = runner.forward(tower, bucket_feed(rows, size))
                vals.append(float(unit_rows(raw)[0] @ r))
        cos[tower] = vals
    for n_tokens, max_texts, tower in packs:
        groups: list[list[int]] = [[]]
        used = 0
        for i, rows in enumerate(fixture_rows):
            need = int(rows.shape[0])
            if need > n_tokens:
                continue
            if len(groups[-1]) == max_texts or used + need > n_tokens:
                groups.append([])
                used = 0
            groups[-1].append(i)
            used += need
        vals = []
        for group in (g for g in groups if g):
            feed = pack_feed(
                [fixture_rows[i] for i in group], n_tokens=n_tokens, max_texts=max_texts
            )
            raw, _ = runner.forward(tower, feed)
            vecs = unit_rows(np.asarray(raw)[: len(group)])
            vals.extend(float(vecs[slot] @ ref[i]) for slot, i in enumerate(group))
        cos[tower] = vals
    report["cosine_vs_s320"] = {
        k: {"n": len(v), "min": min(v) if v else None, "mean": statistics.fmean(v) if v else None}
        for k, v in cos.items()
    }
    report["fixture_tokens"] = [int(r.shape[0]) for r in fixture_rows]
    print(json.dumps(report["cosine_vs_s320"], indent=1))

    # 2) Warm tower latency per bucket (worker-measured, one text).
    probe = _rows(backend, "What causes the northern lights?", "SearchQuery")
    lat: dict[str, dict] = {}
    for size, tower in sorted(buckets.items()):
        feed = bucket_feed(probe, size)
        for _ in range(3):
            runner.forward(tower, feed)
        ms = [runner.forward(tower, feed)[1] for _ in range(args.iters)]
        lat[f"s{size}"] = {"p50": _p(ms, 0.5), "p90": _p(ms, 0.9)}
    for n_tokens, max_texts, tower in packs:
        fill = max(1, min(max_texts, n_tokens // int(probe.shape[0])))
        feed = pack_feed([probe] * fill, n_tokens=n_tokens, max_texts=max_texts)
        for _ in range(3):
            runner.forward(tower, feed)
        ms = [runner.forward(tower, feed)[1] for _ in range(args.iters)]
        lat[tower] = {"p50": _p(ms, 0.5), "p90": _p(ms, 0.9)}
    report["tower_latency_ms"] = lat
    report["probe_tokens"] = int(probe.shape[0])
    print(json.dumps(lat, indent=1))

    # 3) End-to-end single query (tokenize + lookup + IPC + tower).
    e2e = {}
    saved = dict(backend._buckets)
    for label, table in (("s320_only", {TEXT_EMBEDS_S: "text"}), ("buckets", saved)):
        backend._buckets = table
        for _ in range(3):
            backend.embed_text("What causes the northern lights?", role="query")
        ms = []
        for _ in range(args.iters):
            t = time.perf_counter()
            backend.embed_text("What causes the northern lights?", role="query")
            ms.append((time.perf_counter() - t) * 1000.0)
        e2e[label] = {"p50": _p(ms, 0.5), "p90": _p(ms, 0.9)}
    backend._buckets = saved
    report["query_e2e_ms"] = e2e
    print(json.dumps(e2e, indent=1))

    # 4) Throughput on short texts: packed vs one at a time.
    corpora = {
        "grep_lines": grep_lines(args.lines_from, args.lines),
        "posts": list(POSTS) * 16,
    }
    thr = {}
    saved_packs = list(backend._packs)
    plans = [
        ("s320_one_at_a_time", {TEXT_EMBEDS_S: "text"}, [], False),
        ("buckets_one_at_a_time", saved, [], False),
        ("auto", saved, saved_packs, True),
    ] + [(f"only_{p[2]}", saved, [p], True) for p in saved_packs]
    for name, texts in corpora.items():
        lengths = [int(_rows(backend, t, "Document").shape[0]) for t in texts]
        row = {"texts": len(texts), "median_tokens": statistics.median(lengths)}
        base = None
        for label, table, pack_list, pack in plans:
            backend._buckets, backend._packs = table, pack_list
            backend.embed_texts(texts[:16], role="document", pack=pack)
            best = None
            for _ in range(args.repeat):
                t = time.perf_counter()
                vecs, stats = backend.embed_texts(texts, role="document", pack=pack)
                secs = time.perf_counter() - t
                if best is None or secs < best[0]:
                    best = (secs, vecs, stats)
            secs, vecs, stats = best
            row[label] = {
                "texts_per_s": round(len(texts) / secs, 1),
                "tower_texts_per_s": round(len(texts) / (stats.tower_ms / 1000.0), 1),
                "calls": stats.calls,
                "by_tower": stats.by_tower,
            }
            if base is None:
                base = vecs
            else:
                row[label]["min_cos_vs_s320"] = float(np.min(np.sum(vecs * base, axis=1)))
        backend._buckets, backend._packs = saved, saved_packs
        thr[name] = row
    report["throughput"] = thr
    print(json.dumps(thr, indent=1, default=str))
    backend.close()
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
