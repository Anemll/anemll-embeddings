"""HTTP routes for the camera-alert page.

``GET /alert/catalog`` describes the preset rules and whether each sample
file is on disk. ``POST /alert/score`` embeds the text (and any reference
photos) and dots them with cached image and audio vectors.
"""

from __future__ import annotations

import asyncio
import re
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from PIL import Image

from demo.alert_catalog import (
    CHANGE_LABELS,
    FRAMES,
    MEOW_COMPARE,
    REFERENCES,
    SOUNDS,
    default_rules,
)
from demo.alert_library import (
    AlertLibrary,
    decode_audio_upload,
    embed_path,
    valid_id,
)
from demo.alert_score import (
    average_unit,
    change_score,
    cosine,
    fires,
    placeholder_threshold,
    rule_value,
    scope_ids,
    suggest_margin,
    suggest_midpoint,
    suggest_split,
)
from demo.media_io import AUDIO_SR, load_image_bytes

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_AUDIO_BYTES = 30 * 1024 * 1024
MAX_RULES = 12


def mount_alert(app: FastAPI, alert_dir: Path) -> None:
    library = AlertLibrary(alert_dir)
    app.state.alert = library

    @app.get("/alert/catalog")
    async def alert_catalog() -> dict[str, Any]:
        return _catalog_payload(library)

    @app.get("/alert/media/{item_id}")
    async def alert_media(item_id: str) -> FileResponse:
        path = library.media_path(valid_id(item_id))
        if path is None:
            raise HTTPException(404, "sample not downloaded yet")
        media_type = "audio/wav" if path.suffix.lower() == ".wav" else "image/jpeg"
        return FileResponse(path, media_type=media_type)

    @app.get("/alert/reference/{rule_id}")
    async def alert_reference(rule_id: str, index: int = 0) -> FileResponse:
        paths = library.reference_paths(valid_id(rule_id))
        if index < 0 or index >= len(paths):
            raise HTTPException(404, "no reference photo")
        return FileResponse(paths[index], media_type="image/jpeg")

    @app.post("/alert/score")
    async def alert_score(request: Request) -> dict[str, Any]:
        body = await request.json()
        rules = _parse_rules(body.get("rules"))
        wanted = body.get("item_ids")
        compare_on = bool(body.get("include_compare", True))
        # A click sends fresh=true: the clicked frame/sound is embedded again on
        # the Neural Engine instead of read from the cache.
        fresh = bool(body.get("fresh", False))
        return await _score(app, library, rules, wanted, compare_on, fresh=fresh)

    @app.post("/alert/frames/{item_id}")
    async def replace_frame(item_id: str, request: Request) -> dict[str, Any]:
        item_id = _known_frame(item_id)
        data, filename = await _one_file(request)
        if len(data) > MAX_IMAGE_BYTES:
            raise HTTPException(413, "image is too large")
        try:
            image = load_image_bytes(data)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        library.save_frame(item_id, image)
        return {"ok": True, "id": item_id, "media_url": f"/alert/media/{item_id}"}

    @app.post("/alert/sounds/{item_id}")
    async def replace_sound(item_id: str, request: Request) -> dict[str, Any]:
        item_id = _known_sound(item_id)
        data, filename = await _one_file(request)
        if len(data) > MAX_AUDIO_BYTES:
            raise HTTPException(413, "audio is too large")
        try:
            samples = decode_audio_upload(data, filename)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        library.save_sound(item_id, samples, AUDIO_SR)
        return {"ok": True, "id": item_id, "media_url": f"/alert/media/{item_id}"}

    @app.post("/alert/references/{rule_id}")
    async def replace_references(rule_id: str, request: Request) -> dict[str, Any]:
        rule_id = valid_id(rule_id)
        images = await _image_list(request)
        library.save_references(rule_id, images)
        return {"ok": True, "id": rule_id, "count": len(images), "media_url": f"/alert/reference/{rule_id}"}

    @app.delete("/alert/overrides")
    async def clear_overrides() -> dict[str, bool]:
        library.clear_overrides()
        return {"ok": True}


def _catalog_payload(library: AlertLibrary) -> dict[str, Any]:
    frames = [_public_item(library, item) for item in FRAMES]
    sounds = [_public_item(library, item) for item in SOUNDS]
    references = []
    for ref in REFERENCES:
        paths = library.reference_paths(str(ref["rule_id"]))
        row = _public_item(library, ref)
        row["rule_id"] = ref["rule_id"]
        row["count"] = len(paths)
        if paths:
            row["available"] = True
            row["media_url"] = f"/alert/reference/{ref['rule_id']}"
        references.append(row)
    ready = all(row["available"] for row in [*frames, *sounds, *references])
    return {
        "ready": ready,
        "fetch": "python samples/fetch_alert.py",
        "note": (
            "The model compares meaning, so describe the scene rather than giving a command. "
            "Sparky is visual similarity to the reference photo, not identity verification. "
            "Another black cat may also match."
        ),
        "frames": frames,
        "sounds": sounds,
        "references": references,
        "rules": default_rules(),
        "compare": dict(MEOW_COMPARE),
    }


def _public_item(library: AlertLibrary, item: dict[str, Any]) -> dict[str, Any]:
    path = library.media_path(item["id"])
    credit = library.credit_for(item["id"])
    return {
        "id": item["id"],
        "caption": item["caption"],
        "modality": item["modality"],
        "kind": item["kind"],
        "baseline": bool(item.get("baseline")),
        "available": path is not None,
        "media_url": f"/alert/media/{item['id']}" if path is not None else None,
        **credit,
    }


def _parse_rules(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return default_rules()
    if not isinstance(raw, list) or not raw:
        raise HTTPException(400, "rules must be a non-empty list")
    if len(raw) > MAX_RULES:
        raise HTTPException(400, f"at most {MAX_RULES} rules")
    parsed = []
    seen: set[str] = set()
    for row in raw:
        if not isinstance(row, dict):
            raise HTTPException(400, "each rule must be an object")
        rule_id = valid_id(str(row.get("id") or ""))
        if rule_id in seen:
            raise HTTPException(400, f"duplicate rule {rule_id}")
        seen.add(rule_id)
        rule_type = str(row.get("type") or "")
        if rule_type not in {"text", "photo", "change"}:
            raise HTTPException(400, f"rule {rule_id} type must be text, photo, or change")
        name = str(row.get("name") or "").strip()
        if not name or len(name) > 80:
            raise HTTPException(400, f"rule {rule_id} needs a name")
        scope = str(row.get("scope") or "all")
        if scope not in {"image", "audio", "all"}:
            raise HTTPException(400, f"rule {rule_id} scope must be image, audio, or all")
        text = str(row.get("text") or "").strip()
        if rule_type == "text" and (not text or len(text) > 500):
            raise HTTPException(400, f"rule {rule_id} needs a text description")
        baseline = row.get("baseline_id")
        if baseline:
            baseline = valid_id(str(baseline))
        if rule_type == "change" and not baseline:
            raise HTTPException(400, f"rule {rule_id} needs a baseline frame")
        positive_ids = _positive_ids(row.get("positive_ids"))
        threshold = row.get("threshold")
        if threshold is None or threshold == "":
            threshold_value = None
        else:
            try:
                threshold_value = float(threshold)
            except (TypeError, ValueError) as exc:
                raise HTTPException(400, f"rule {rule_id} threshold must be a number") from exc
        chip = str(row.get("chip") or name).strip()[:40]
        color = str(row.get("color") or "#e39a45").strip()
        # Echoed into a CSS custom property in the page: hex colours only.
        if not re.fullmatch(r"#[0-9a-fA-F]{3,8}", color):
            color = "#e39a45"
        parsed.append(
            {
                "id": rule_id,
                "type": rule_type,
                "name": name,
                "chip": chip,
                "text": text or None,
                "label": str(row.get("label") or "").strip() or None,
                "baseline_id": baseline,
                "scope": scope,
                "threshold": threshold_value,
                "positive_ids": positive_ids,
                "color": color,
            }
        )
    return parsed


def _positive_ids(raw: Any) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, list) or len(raw) > 24:
        raise HTTPException(400, "positive_ids must be a list of sample ids")
    return [valid_id(str(item_id)) for item_id in raw]


def _known_frame(item_id: str) -> str:
    item_id = valid_id(item_id)
    if item_id not in {row["id"] for row in FRAMES}:
        raise HTTPException(404, "unknown frame")
    return item_id


def _known_sound(item_id: str) -> str:
    item_id = valid_id(item_id)
    if item_id not in {row["id"] for row in SOUNDS}:
        raise HTTPException(404, "unknown sound")
    return item_id


async def _one_file(request: Request) -> tuple[bytes, str]:
    form = await request.form()
    upload = form.get("file")
    if upload is None or not hasattr(upload, "read"):
        raise HTTPException(400, "file is required")
    data = await upload.read()
    filename = getattr(upload, "filename", None) or ""
    if not data:
        raise HTTPException(400, "file is empty")
    return data, filename


async def _image_list(request: Request) -> list[Image.Image]:
    form = await request.form()
    uploads = form.getlist("file") if hasattr(form, "getlist") else []
    if not uploads:
        one = form.get("file")
        uploads = [one] if one is not None else []
    images: list[Image.Image] = []
    for upload in uploads:
        if not hasattr(upload, "read"):
            continue
        data = await upload.read()
        if len(data) > MAX_IMAGE_BYTES:
            raise HTTPException(413, "image is too large")
        try:
            images.append(load_image_bytes(data))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
    if not images or len(images) > 3:
        raise HTTPException(400, "upload 1 to 3 reference photos")
    return images


async def _score(
    app: FastAPI,
    library: AlertLibrary,
    rules: list[dict[str, Any]],
    wanted: Any,
    compare_on: bool,
    *,
    fresh: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    media = [_public_item(library, item) for item in [*FRAMES, *SOUNDS]]
    available = [row for row in media if row["available"]]
    by_id = {row["id"]: row for row in available}
    if wanted is None:
        target_ids = [row["id"] for row in available]
    else:
        if not isinstance(wanted, list) or not wanted:
            raise HTTPException(400, "item_ids must be a list of ids")
        target_ids = []
        for raw in wanted:
            item_id = valid_id(str(raw))
            if item_id not in by_id:
                raise HTTPException(400, f"{item_id} is not downloaded")
            target_ids.append(item_id)

    needed = set(target_ids)
    fresh_ids = set(target_ids) if fresh else set()
    for rule in rules:
        if rule.get("baseline_id"):
            base = str(rule["baseline_id"])
            if base not in by_id:
                raise HTTPException(400, f"baseline {base} is not downloaded")
            needed.add(base)

    latency_ms = 0.0
    fresh_embeds = 0
    placement = None
    backend_name = None
    vectors: dict[str, Any] = {}
    item_timing: list[dict[str, Any]] = []

    def _remember(spent: float) -> None:
        nonlocal latency_ms, fresh_embeds, placement, backend_name
        latency_ms += spent
        if spent > 0:
            fresh_embeds += 1
        health = app.state.backend.health()
        placement = health.get("placement")
        backend_name = health.get("backend")

    async def embed_media(item_id: str) -> Any:
        path = library.media_path(item_id)
        if path is None:
            raise HTTPException(400, f"{item_id} is not downloaded")
        modality = by_id[item_id]["modality"]
        split: dict[str, Any] = {}
        renew = item_id in fresh_ids
        vector, spent = await asyncio.to_thread(
            partial(library.embed_file, fresh=renew), path, modality, _embedder(app, split)
        )
        _remember(spent)
        if renew:
            item_timing.append(
                {
                    "id": item_id,
                    "modality": modality,
                    "embed_ms": round(float(spent), 3),
                    **{key: round(float(value), 3) for key, value in split.items() if key.endswith("_ms")},
                }
            )
        return vector

    async def cached_media(item_id: str) -> Any | None:
        if item_id in vectors:
            return vectors[item_id]
        path = library.media_path(item_id)
        if path is None or not library.is_cached_file(path):
            return None
        vectors[item_id] = await embed_media(item_id)
        return vectors[item_id]

    for item_id in needed:
        vectors[item_id] = await embed_media(item_id)

    text_vectors: dict[str, Any] = {}

    async def embed_query(text: str) -> Any:
        cached = text_vectors.get(text)
        if cached is not None:
            return cached

        def _run(query: str, role: str) -> tuple[Any, float]:
            vector = app.state.backend.embed_text(query, role=role)
            last = getattr(app.state.backend, "last", None)
            latency = float(last.latency_ms) if last is not None else 0.0
            return vector, latency

        vector, spent = await asyncio.to_thread(library.embed_text, text, "query", _run)
        _remember(spent)
        text_vectors[text] = vector
        return vector

    rule_vectors: dict[str, Any] = {}
    rule_notes: dict[str, str | None] = {}
    for rule in rules:
        if rule["type"] == "text":
            rule_vectors[rule["id"]] = await embed_query(str(rule["text"]))
            rule_notes[rule["id"]] = None
        elif rule["type"] == "photo":
            paths = library.reference_paths(rule["id"])
            if not paths:
                rule_notes[rule["id"]] = "no reference photo yet"
                continue
            refs = []
            for path in paths:
                vector, spent = await asyncio.to_thread(library.embed_file, path, "image", _embedder(app))
                _remember(spent)
                refs.append(vector)
            rule_vectors[rule["id"]] = average_unit(refs)
            rule_notes[rule["id"]] = None
        elif rule["type"] == "change":
            # Compared with the baseline frame below, not with a text query.
            rule_notes[rule["id"]] = None
        else:
            raise HTTPException(400, f"unknown rule type {rule['type']}")

    compare_vec = None
    if compare_on:
        compare_vec = await embed_query(str(MEOW_COMPARE["text"]))

    label_vectors: list[tuple[str, Any]] = []
    if any(rule["type"] == "change" for rule in rules):
        for label in CHANGE_LABELS:
            label_vectors.append((str(label["caption"]), await embed_query(str(label["text"]))))

    match_started = time.perf_counter()
    raw_by_rule: dict[str, dict[str, float]] = {}
    cosine_by_rule: dict[str, dict[str, float]] = {}
    hints: dict[str, dict[str, str]] = {}
    for rule in rules:
        raw_by_rule[rule["id"]] = {}
        cosine_by_rule[rule["id"]] = {}
        hints[rule["id"]] = {}
        ids = scope_ids(available, str(rule["scope"]))
        for item_id in ids:
            # A single click embeds that frame (and the empty-street baseline).
            # Other samples affect the suggested threshold only once they are
            # cached, or when this request asked for them.
            if item_id not in vectors:
                await cached_media(item_id)
        if rule["type"] == "change":
            baseline_id = str(rule["baseline_id"])
            base_vec = vectors.get(baseline_id)
            if base_vec is None:
                continue
            for item_id in ids:
                vector = vectors.get(item_id)
                if vector is None:
                    continue
                similarity = cosine(vector, base_vec)
                cosine_by_rule[rule["id"]][item_id] = similarity
                raw_by_rule[rule["id"]][item_id] = change_score(similarity, baseline=item_id == baseline_id)
                if label_vectors:
                    caption = max(label_vectors, key=lambda pair: cosine(vector, pair[1]))[0]
                    hints[rule["id"]][item_id] = f"Looks most like {caption}"
            continue
        probe = rule_vectors.get(rule["id"])
        if probe is None:
            continue
        for item_id, vector in vectors.items():
            if item_id not in ids and item_id not in target_ids:
                continue
            similarity = cosine(vector, probe)
            cosine_by_rule[rule["id"]][item_id] = similarity
            raw_by_rule[rule["id"]][item_id] = similarity

    suggestions: dict[str, float] = {}
    thresholds: dict[str, float] = {}
    for rule in rules:
        raws = raw_by_rule.get(rule["id"]) or {}
        scope = scope_ids(available, str(rule["scope"]))
        complete = bool(scope) and all(item_id in raws for item_id in scope)
        positives = [str(item_id) for item_id in (rule.get("positive_ids") or [])]
        if rule["type"] == "change" and not positives and rule.get("baseline_id"):
            positives = [item_id for item_id in scope if item_id != rule["baseline_id"]]
        scoped = {item_id: raws[item_id] for item_id in scope if item_id in raws}
        if rule["type"] != "change" and rule.get("baseline_id") and rule["baseline_id"] in raws:
            base_raw = raws[str(rule["baseline_id"])]
            scoped = {
                item_id: rule_value(raws[item_id], base_raw)
                for item_id in scope
                if item_id in raws
            }
        if positives and complete:
            suggested = suggest_split(scoped, positives)
        elif rule.get("baseline_id") and rule["type"] != "change" and complete:
            suggested = suggest_margin([value for item_id, value in scoped.items() if item_id != rule["baseline_id"]])
        else:
            suggested = suggest_midpoint(list(scoped.values())) if complete else None
        if suggested is not None:
            suggestions[rule["id"]] = _num(suggested)
        sent = rule.get("threshold")
        thresholds[rule["id"]] = float(sent) if sent is not None else float(
            suggestions.get(rule["id"], placeholder_threshold(rule))
        )

    compare_scores: dict[str, float] = {}
    compare_threshold = float(MEOW_COMPARE["threshold"])
    compare_suggestion = None
    if compare_vec is not None:
        audio_ids = scope_ids(available, "audio")
        for item_id in audio_ids:
            if item_id not in vectors:
                await cached_media(item_id)
            if item_id in vectors:
                compare_scores[item_id] = cosine(vectors[item_id], compare_vec)
        if audio_ids and all(item_id in compare_scores for item_id in audio_ids):
            compare_suggestion = suggest_split(compare_scores, list(MEOW_COMPARE.get("positive_ids") or []))
            if compare_suggestion is not None:
                compare_suggestion = _num(compare_suggestion)

    items_out = []
    for item_id in target_ids:
        row = by_id[item_id]
        rule_rows = []
        for rule in rules:
            # A barking-dog rule does not fire on a photograph, and a UPS
            # rule does not fire on a sound. Scope "all" still scores both.
            if item_id not in scope_ids(available, str(rule["scope"])):
                continue
            note = rule_notes.get(rule["id"])
            raws = raw_by_rule.get(rule["id"]) or {}
            if note or item_id not in raws:
                rule_rows.append(
                    {
                        "id": rule["id"],
                        "name": rule["name"],
                        "chip": rule["chip"],
                        "color": rule["color"],
                        "score": None,
                        "raw": None,
                        "threshold": _num(thresholds[rule["id"]]),
                        "high": False,
                        "detail": note or "not scored",
                    }
                )
                continue
            if rule["type"] == "change":
                value = float(raws[item_id])
                raw_value = cosine_by_rule.get(rule["id"], {}).get(item_id, value)
                detail = rule.get("label") or "different from the usual empty street"
                hint = hints.get(rule["id"], {}).get(item_id)
            else:
                baseline_id = rule.get("baseline_id")
                baseline_raw = raws.get(baseline_id) if baseline_id else None
                value = rule_value(raws[item_id], baseline_raw if baseline_id else None)
                raw_value = raws[item_id]
                detail = "margin over the empty street" if baseline_id else None
                hint = None
            threshold = thresholds[rule["id"]]
            rule_rows.append(
                {
                    "id": rule["id"],
                    "name": rule["name"],
                    "chip": rule["chip"],
                    "color": rule["color"],
                    "score": _num(value),
                    "raw": _num(raw_value),
                    "threshold": _num(threshold),
                    "margin": _num(value - threshold),
                    "high": fires(value, threshold),
                    "detail": detail,
                    "hint": hint,
                }
            )
        comparisons = []
        if compare_vec is not None and row["modality"] == "audio" and item_id in compare_scores:
            used = compare_suggestion if compare_suggestion is not None else compare_threshold
            score = compare_scores[item_id]
            comparisons.append(
                {
                    "id": MEOW_COMPARE["id"],
                    "label": MEOW_COMPARE["label"],
                    "score": _num(score),
                    "threshold": _num(used),
                    "high": fires(score, float(used)),
                }
            )
        photo_high = any(entry["high"] and _rule_type(rules, entry["id"]) == "photo" for entry in rule_rows)
        unknown_cat = row["kind"] == "cat" and not photo_high
        items_out.append(
            {
                "id": item_id,
                "caption": row["caption"],
                "modality": row["modality"],
                "kind": row["kind"],
                "rules": rule_rows,
                "comparisons": comparisons,
                "unknown_cat": unknown_cat,
                "fired": [entry["chip"] for entry in rule_rows if entry["high"]],
            }
        )

    match_ms = (time.perf_counter() - match_started) * 1000.0
    towers: dict[str, float] = {}
    for row in item_timing:
        for key in ("vision_ms", "audio_ms", "text_ms"):
            if key in row:
                towers[key[: -len("_ms")]] = round(towers.get(key[: -len("_ms")], 0.0) + row[key], 3)
    server_ms = (time.perf_counter() - started) * 1000.0
    timing = {
        "items": item_timing,
        "towers": towers,
        "match_ms": round(match_ms, 3),
        "server_ms": round(server_ms, 3),
    }
    tower_text = " ".join(f"{name} {value:.1f} ms" for name, value in towers.items()) or "cached"
    print(
        f"[alert] score items={','.join(target_ids)} fresh={','.join(sorted(fresh_ids)) or '-'} "
        f"{tower_text} match {match_ms:.2f} ms server {server_ms:.1f} ms",
        file=sys.stderr,
        flush=True,
    )
    return {
        "items": items_out,
        "timing": timing,
        "latency_ms": round(latency_ms, 3),
        "backend": backend_name,
        "placement": placement,
        "suggested_thresholds": suggestions,
        "compare_threshold": compare_suggestion,
        "thresholds": {key: _num(value) for key, value in thresholds.items()},
        "fresh_embeds": fresh_embeds,
    }


def _num(value: float) -> float:
    """Keep enough precision that a tiny negative score does not become zero."""
    return round(float(value), 6)


def _embedder(app: FastAPI, timing: dict[str, Any] | None = None):
    def _run(file_path: Path, kind: str) -> tuple[Any, float]:
        return embed_path(file_path, kind, app.state.backend, timing)

    return _run


def _rule_type(rules: list[dict[str, Any]], rule_id: str) -> str:
    for rule in rules:
        if rule["id"] == rule_id:
            return str(rule["type"])
    return ""

