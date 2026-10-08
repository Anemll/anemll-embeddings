#!/usr/bin/env python3
"""API checks for the showcase server (mock backend, no Core AI)."""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from _coreai_run_npy import parity_compute  # noqa: E402
import demo.coreai_worker as coreai_worker  # noqa: E402
from demo.backends.coreai import apply_compute_env  # noqa: E402
from demo.coreai_worker import (  # noqa: E402
    _manifest_label,
    _placement,
    manifest_label_for_package,
    package_main_hash_hex,
)
from demo.feeds import audio_tower_feed  # noqa: E402
from demo.scripts.fetch_corpus import _trim_with_wave, audio_skip_reason, fetch  # noqa: E402
from demo.alert_catalog import default_rules  # noqa: E402
from demo.alert_score import suggest_margin, suggest_midpoint  # noqa: E402
from demo.scripts.fetch_alert import license_ok  # noqa: E402
from demo.server import create_app  # noqa: E402
from demo.settings import DEFAULT_PORT, server_compute  # noqa: E402


def _fail(msg: str) -> None:
    raise AssertionError(msg)


def _wav_bytes(seconds: float = 0.25, freq: float = 440.0, rate: int = 16000) -> bytes:
    n = int(seconds * rate)
    buf = io.BytesIO()
    with wave.open(buf, "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        frames = bytearray()
        for i in range(n):
            sample = 0.2 * np.sin(2 * np.pi * freq * i / rate)
            frames += np.int16(np.clip(sample, -1, 1) * 32767).tobytes()
        handle.writeframes(bytes(frames))
    return buf.getvalue()


def _png(color: tuple[int, int, int]) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (24, 24), color).save(buf, "PNG")
    return buf.getvalue()


def _unit(vec: list[float]) -> float:
    arr = np.asarray(vec, dtype=np.float64)
    return float(np.linalg.norm(arr))


def test_compute_defaults() -> None:
    saved_demo = os.environ.pop("ANEMLL_DEMO_COMPUTE", None)
    saved_core = os.environ.pop("ANEMLL_COREAI_COMPUTE", None)
    try:
        if parity_compute() != "cpu":
            _fail(f"parity default {parity_compute()}")
        if server_compute() != "ane":
            _fail(f"server default {server_compute()}")
        os.environ["ANEMLL_COREAI_COMPUTE"] = "cpu"
        if server_compute() != "ane":
            _fail("server followed the parity CPU variable")
        if parity_compute() != "cpu":
            _fail("parity ignored ANEMLL_COREAI_COMPUTE")
        os.environ["ANEMLL_COREAI_COMPUTE"] = "ane"
        if parity_compute() != "ane":
            _fail("parity did not honor ane")
        if apply_compute_env("ane", {})["ANEMLL_COREAI_COMPUTE"] != "ane":
            _fail("worker env")
        if server_compute("cpu") != "cpu":
            _fail("explicit cpu")
    finally:
        if saved_demo is None:
            os.environ.pop("ANEMLL_DEMO_COMPUTE", None)
        else:
            os.environ["ANEMLL_DEMO_COMPUTE"] = saved_demo
        if saved_core is None:
            os.environ.pop("ANEMLL_COREAI_COMPUTE", None)
        else:
            os.environ["ANEMLL_COREAI_COMPUTE"] = saved_core


def test_audio_keep_mask_not_all_ones() -> None:
    feat = np.ones((1, 99, 128), dtype=np.float32)
    mask = np.ones((1, 99), dtype=np.float32)
    mask[:, -3:] = 0
    feed = audio_tower_feed(feat, mask)
    packed = feed["input_features_mask"]
    if packed.shape != (1, 1, 280, 1):
        _fail(str(packed.shape))
    kept = int((packed.reshape(-1) != 0).sum())
    if kept != 96:
        _fail(f"kept {kept}, expected 96 (pad frames must stay off)")
    if kept == packed.size:
        _fail("audio mask was forced to all ones")


def test_api(tmp: Path) -> None:
    app = create_app(backend="mock", data_dir=tmp, compute="ane")
    with TestClient(app) as client:
        health = client.get("/health")
        if health.status_code != 200:
            _fail(health.text)
        body = health.json()
        names = {row["name"] for row in body["towers"]}
        if names != {"vision_s280", "audio_s280", "text_embeds_s320"}:
            _fail(str(names))
        if body["backend"] != "mock" or not body["warmup"]:
            _fail(str(body))
        if any(row["warmup_ms"] is None for row in body["towers"]):
            _fail("missing warmup")

        empty = client.post("/embed", json={})
        if empty.status_code != 400:
            _fail(str(empty.status_code))

        both = client.post(
            "/embed",
            data={"text": "hello"},
            files={"file": ("a.png", _png((1, 2, 3)), "image/png")},
        )
        if both.status_code != 400:
            _fail("text+file should be rejected")

        text = client.post("/embed", json={"text": "northern lights"})
        if text.status_code != 200:
            _fail(text.text)
        vec = text.json()["vector"]
        if len(vec) != 768 or abs(_unit(vec) - 1.0) > 1e-4:
            _fail(f"norm {_unit(vec)} dim {len(vec)}")
        again = client.post("/embed", json={"text": "northern lights"})
        if again.json()["vector"] != vec:
            _fail("mock text embedding is not stable")

        page = client.get("/")
        if page.status_code != 200 or b"latency-badge" not in page.content:
            _fail("search page")
        if b"latency-badge" not in client.get("/heatmap").content:
            _fail("heatmap page")
        if b"latency-badge" not in client.get("/heard").content:
            _fail("heard page")
        alert_page = client.get("/alert")
        if alert_page.status_code != 200 or b"Score everything" not in alert_page.content:
            _fail("alert page")
        if b'href="/alert"' not in client.get("/").content:
            _fail("alert nav")
        js = (REPO_ROOT / "demo" / "static" / "common.js").read_text()
        if "fullyOnANE" not in js or "on ${where} · ${ms} ms" not in js:
            _fail("badge format missing")

        client.post(
            "/index",
            json={"text": "the northern lights are a green glow", "label": "aurora"},
        )
        client.post(
            "/index",
            json={"text": "quarterly payroll ledger", "label": "payroll"},
        )
        found = client.post("/search", json={"text": "northern lights", "k": 2})
        hits = found.json()["results"]
        if not hits or hits[0]["label"] != "aurora":
            _fail(str([(h["label"], h["score"]) for h in hits]))
        if "latency_ms" not in found.json()["query"]:
            _fail("query timing")

        red = client.post(
            "/index",
            files={"file": ("red.png", _png((220, 30, 30)), "image/png")},
            data={"label": "red"},
        )
        blue = client.post(
            "/index",
            files={"file": ("blue.png", _png((30, 40, 210)), "image/png")},
            data={"label": "blue"},
        )
        if red.status_code != 200 or blue.status_code != 200:
            _fail(red.text + blue.text)
        ranked = client.post(
            "/search",
            files={"file": ("q.png", _png((220, 30, 30)), "image/png")},
            data={"k": "4", "filter_modality": "image"},
        )
        image_hits = ranked.json()["results"]
        if image_hits[0]["label"] != "red":
            _fail(str([(h["label"], round(h["score"], 3)) for h in image_hits]))

        tone = _wav_bytes()
        indexed = client.post(
            "/index",
            files={"file": ("tone.wav", tone, "audio/wav")},
            data={"label": "a4", "session": "s1", "t_start": "0", "t_end": "0.25"},
        )
        if indexed.status_code != 200:
            _fail(indexed.text)
        other = client.post(
            "/index",
            files={"file": ("other.wav", _wav_bytes(freq=900), "audio/wav")},
            data={"label": "other", "session": "s2"},
        )
        audio_hits = client.post(
            "/search",
            files={"file": ("q.wav", tone, "audio/wav")},
            data={"session": "s1", "filter_modality": "audio", "k": "5"},
        ).json()["results"]
        if len(audio_hits) != 1 or audio_hits[0]["label"] != "a4":
            _fail(str(audio_hits))
        if audio_hits[0].get("media_url") is None:
            _fail("audio media")
        media = client.get(audio_hits[0]["media_url"])
        if media.status_code != 200 or media.headers["content-type"] != "audio/wav":
            _fail("media bytes")

        ids = [red.json()["item"]["id"], blue.json()["item"]["id"]]
        matrix = client.post("/compare", json={"ids": ids})
        if matrix.status_code != 200:
            _fail(matrix.text)
        scores = matrix.json()["matrix"]
        if abs(scores[0][0] - 1.0) > 1e-3 or abs(scores[1][1] - 1.0) > 1e-3:
            _fail(str(scores))
        if "latency_ms" not in matrix.json():
            _fail("compare timing")

        listed = client.get("/items").json()["items"]
        if len(listed) < 4:
            _fail(str(len(listed)))
        gone = client.delete(f"/items/{other.json()['item']['id']}")
        if gone.status_code != 200:
            _fail("delete")

        too_many = client.post("/search", json={"text": "lights", "k": 0})
        if too_many.status_code != 400:
            _fail("k")

    reloaded = create_app(backend="mock", data_dir=tmp, compute="ane")
    with TestClient(reloaded) as client:
        again = client.post("/search", json={"text": "northern lights", "k": 1, "filter_modality": "text"})
        if again.json()["results"][0]["label"] != "aurora":
            _fail("index did not persist")


def test_compressed_audio(tmp: Path) -> None:
    if shutil.which("ffmpeg") is None:
        print("SKIP test_compressed_audio (ffmpeg not installed)")
        return
    app = create_app(backend="mock", data_dir=tmp / "enc", compute="cpu")
    webm = tmp / "tone.webm"
    ogg = tmp / "tone.ogg"
    webm_ok = subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=0.3",
            "-c:a", "libopus", str(webm),
        ],
        check=False,
    ).returncode == 0
    ogg_ok = subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "sine=frequency=523:duration=0.3",
            "-c:a", "libvorbis", str(ogg),
        ],
        check=False,
    ).returncode == 0
    if not webm_ok and not ogg_ok:
        _fail("ffmpeg could not encode webm or ogg")
    with TestClient(app) as client:
        if webm_ok:
            response = client.post(
                "/embed",
                files={"file": ("mic.webm", webm.read_bytes(), "audio/webm")},
            )
            if response.status_code != 200 or response.json()["modality"] != "audio":
                _fail(response.text)
        if ogg_ok:
            response = client.post(
                "/embed",
                files={"file": ("clip.ogg", ogg.read_bytes(), "audio/ogg")},
            )
            if response.status_code != 200:
                _fail(response.text)


def _write_tone(path: Path, seconds: float, rate: int = 16000) -> None:
    n = int(seconds * rate)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * n)


def test_wave_trim_and_missing_converter(tmp: Path) -> None:
    src = tmp / "long.wav"
    dest = tmp / "short.wav"
    _write_tone(src, 10.0)
    _trim_with_wave(src, dest, seconds=8)
    with wave.open(str(dest), "r") as handle:
        if handle.getframerate() != 16000 or handle.getnframes() != 16000 * 8:
            _fail(f"trim {handle.getframerate()} {handle.getnframes()}")
    if shutil.which("ffmpeg") or shutil.which("afconvert"):
        if audio_skip_reason() is not None:
            _fail("converter present but skip reason set")
    else:
        reason = audio_skip_reason() or ""
        if "afconvert" not in reason:
            _fail(reason)


def test_port_and_artifacts_guard(tmp: Path) -> None:
    if DEFAULT_PORT != 8766:
        _fail(f"default port {DEFAULT_PORT}")
    art = tmp / "artifacts"
    art.mkdir(parents=True)
    try:
        create_app(backend="mock", data_dir=art / "demo", artifacts=art, compute="cpu")
        _fail("data dir inside artifacts was accepted")
    except ValueError as exc:
        if "artifacts" not in str(exc):
            _fail(str(exc))
    if (art / "demo").exists():
        _fail("store was created inside artifacts")
    saved = os.environ.get("ANEMLL_EMBEDDINGS_ARTIFACTS")
    os.environ["ANEMLL_EMBEDDINGS_ARTIFACTS"] = str(art)
    try:
        fetch(art / "corpus", limit_images=0, limit_audio=0)
        _fail("corpus dest inside artifacts was accepted")
    except ValueError as exc:
        if "artifacts" not in str(exc):
            _fail(str(exc))
    finally:
        if saved is None:
            os.environ.pop("ANEMLL_EMBEDDINGS_ARTIFACTS", None)
        else:
            os.environ["ANEMLL_EMBEDDINGS_ARTIFACTS"] = saved
    if (art / "corpus").exists():
        _fail("corpus was created inside artifacts")
    outside = tmp / "demo-data"
    create_app(backend="mock", data_dir=outside, artifacts=art, compute="cpu")
    if not (outside / "index.json").exists() and not outside.is_dir():
        _fail("outside data dir was not created")


class _Silent:
    _debug_infos = b"{}"


def test_package_hash_manifest(tmp: Path) -> None:
    digest = "ab" * 32
    pkg = tmp / "vision_s280.aimodel"
    pkg.mkdir(parents=True)
    (pkg / "main.hash").write_bytes(bytes.fromhex(digest))
    if package_main_hash_hex(pkg) != digest:
        _fail(package_main_hash_hex(pkg) or "no hash")
    cache = tmp / "coreai-cache"
    own = cache / "aa" / "bb" / "cc" / digest / "model.aimodelx" / "spec" / "manifest.plist"
    other = cache / "11" / "22" / "33" / ("cd" * 32) / "model.aimodelx" / "spec" / "manifest.plist"
    own.parent.mkdir(parents=True)
    other.parent.mkdir(parents=True)
    own.write_bytes(b"mps.fullyPlacedOnANE")
    other.write_bytes(b"gpu only")
    old = time.time() - 10_000
    os.utime(own, (old, old))
    os.utime(other, None)
    label = manifest_label_for_package(pkg, cache)
    if label != "fullyOnANE":
        _fail(f"hash lookup {label}")
    saved = coreai_worker.CACHE
    coreai_worker.CACHE = cache
    try:
        placed = _placement(_Silent(), time.time(), pkg)
    finally:
        coreai_worker.CACHE = saved
    if placed != "fullyOnANE":
        _fail(f"placement {placed}")
    bare = tmp / "nohash.aimodel"
    bare.mkdir()
    if package_main_hash_hex(bare) is not None:
        _fail("missing main.hash")
    fallback = _manifest_label(time.time() - 5, cache)
    if fallback != "GPU":
        _fail(f"fallback {fallback}")


def _jpeg(path: Path, color: tuple[int, int, int], mark: tuple[int, int, int] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (48, 48), color)
    if mark is not None:
        image.putpixel((6, 6), mark)
        image.putpixel((12, 20), mark)
        image.putpixel((30, 8), mark)
    image.save(path, "JPEG", quality=95)


def _alert_fixture(root: Path) -> None:
    colors = {
        "frames/ups.jpg": (90, 50, 20),
        "frames/fedex.jpg": (40, 40, 140),
        "frames/ginger.jpg": (220, 120, 30),
        "frames/door.jpg": (40, 90, 160),
        "frames/street.jpg": (180, 180, 180),
    }
    for rel, color in colors.items():
        _jpeg(root / rel, color)
    sparky = root / "frames" / "sparky.jpg"
    _jpeg(sparky, (8, 8, 12), (240, 200, 40))
    ref = root / "references" / "sparky.jpg"
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_bytes(sparky.read_bytes())
    sounds = root / "sounds"
    sounds.mkdir(parents=True, exist_ok=True)
    (sounds / "bark.wav").write_bytes(_wav_bytes(freq=180))
    (sounds / "meow.wav").write_bytes(_wav_bytes(freq=900))


def _rule_row(item: dict, rule_id: str) -> dict:
    for row in item["rules"]:
        if row["id"] == rule_id:
            return row
    _fail(f"missing rule {rule_id} on {item['id']}")
    return {}


def test_alert(tmp: Path) -> None:
    if abs(suggest_midpoint([0.4, 0.2, 0.1]) - 0.3) > 1e-9:
        _fail("midpoint")
    if abs(suggest_margin([0.0, 0.04, 0.2]) - 0.02) > 1e-9:
        _fail("margin threshold")
    if license_ok("CC BY-SA 4.0") and license_ok("CC0") and license_ok("Public domain"):
        pass
    else:
        _fail("open licenses should pass")
    if license_ok("CC BY-NC 2.0") or license_ok("CC BY-ND 4.0") or license_ok("GFDL"):
        _fail("nc/nd/gfdl should fail")

    _alert_fixture(tmp)
    app = create_app(backend="mock", data_dir=tmp / "index", alert_dir=tmp, compute="cpu")
    with TestClient(app) as client:
        catalog = client.get("/alert/catalog")
        if catalog.status_code != 200:
            _fail(catalog.text)
        body = catalog.json()
        if len(body["frames"]) != 6 or len(body["sounds"]) != 2:
            _fail("catalog size")
        if not body["ready"]:
            _fail("fixture should be complete")
        names = [rule["name"] for rule in body["rules"]]
        if names != ["Anything significant", "UPS truck", "Sparky", "Dog barking"]:
            _fail(str(names))
        if body["compare"]["text"] != "a cat meowing":
            _fail("meow compare")
        sparky_ref = next(row for row in body["references"] if row["rule_id"] == "sparky")
        if not sparky_ref["available"] or client.get(sparky_ref["media_url"]).status_code != 200:
            _fail("sparky reference")
        page = client.get("/alert").text
        for snippet in (
            "These are your alerts.",
            "Click what the camera sees or hears.",
            "Watch which alerts fire.",
            "Score everything",
            "Your alerts",
            "Camera feed",
            "Sounds",
            "Advanced / customize",
            "green = alert fires",
            "Front door cam: UPS truck",
            "Sparky (test photo)",
            "Neighbor's cat",
            "Sound: dog barking",
        ):
            if snippet not in page and snippet not in (REPO_ROOT / "demo" / "static" / "alert.js").read_text():
                # Captions are rendered by alert.js from the catalog, not baked into the HTML.
                if snippet in page or snippet in (REPO_ROOT / "demo" / "static" / "alert.js").read_text() or snippet in json.dumps(body):
                    continue
                _fail(f"missing {snippet}")

        rules = default_rules()
        for rule in rules:
            if rule["id"] == "sparky":
                rule["threshold"] = 0.99
        scored = client.post("/alert/score", json={"rules": rules, "include_compare": True})
        if scored.status_code != 200:
            _fail(scored.text)
        payload = scored.json()
        if payload["fresh_embeds"] < 1:
            _fail("expected fresh embeds")
        by_id = {row["id"]: row for row in payload["items"]}
        street = _rule_row(by_id["street"], "significant")
        if abs(street["score"]) > 1e-4 or street["high"]:
            _fail(f"empty street should sit on the baseline, got {street}")
        sparky = _rule_row(by_id["sparky"], "sparky")
        ginger = _rule_row(by_id["ginger"], "sparky")
        if abs(sparky["score"] - 1.0) > 1e-3 or not sparky["high"]:
            _fail(f"sparky photo match {sparky}")
        if ginger["high"] or ginger["score"] >= sparky["score"]:
            _fail(f"ginger should lose to sparky {ginger}")
        if by_id["ginger"]["unknown_cat"] is not True or by_id["sparky"]["unknown_cat"]:
            _fail("unknown cat labels")
        if any(row["id"] == "dog" for row in by_id["ups"]["rules"]):
            _fail("dog rule should not score a camera frame")
        if any(row["id"] == "ups" for row in by_id["bark"]["rules"]):
            _fail("UPS rule should not score a sound")
        if "Dog barking" not in [row["chip"] for row in by_id["bark"]["rules"]]:
            _fail("bark rules")
        meow_cmp = by_id["meow"]["comparisons"]
        bark_cmp = by_id["bark"]["comparisons"]
        if not meow_cmp or meow_cmp[0]["label"] != "a cat meowing":
            _fail(str(meow_cmp))
        if "dog" not in payload["suggested_thresholds"] or payload["compare_threshold"] is None:
            _fail(str(payload["suggested_thresholds"]))
        fitted = default_rules()
        for rule in fitted:
            if rule["id"] in payload["suggested_thresholds"]:
                rule["threshold"] = payload["suggested_thresholds"][rule["id"]]
        again = client.post(
            "/alert/score",
            json={"item_ids": ["bark", "meow"], "rules": fitted, "include_compare": True},
        )
        if again.status_code != 200:
            _fail(again.text)
        if again.json()["fresh_embeds"] != 0:
            _fail(f"cache miss {again.json()['fresh_embeds']}")
        sound_rows = {row["id"]: row for row in again.json()["items"]}
        dog_high = [key for key, row in sound_rows.items() if _rule_row(row, "dog")["high"]]
        meow_high = [key for key, row in sound_rows.items() if row["comparisons"][0]["high"]]
        if len(dog_high) != 1 or len(meow_high) != 1:
            _fail(f"sound split dog={dog_high} meow={meow_high}")

        replaced = client.post(
            "/alert/frames/sparky",
            files={"file": ("cat.png", _png((220, 120, 30)), "image/png")},
        )
        if replaced.status_code != 200:
            _fail(replaced.text)
        moved = client.post(
            "/alert/score",
            json={"item_ids": ["sparky"], "rules": rules, "include_compare": False},
        )
        if moved.status_code != 200:
            _fail(moved.text)
        if _rule_row(moved.json()["items"][0], "sparky")["score"] > 0.99:
            _fail("replaced frame should no longer match the reference exactly")

        bad = client.post("/alert/score", json={"rules": []})
        if bad.status_code != 400:
            _fail("empty rules")

    art = tmp / "artifacts"
    art.mkdir()
    try:
        create_app(backend="mock", data_dir=tmp / "safe-data", alert_dir=art / "alert", artifacts=art, compute="cpu")
        _fail("alert dir inside artifacts was accepted")
    except ValueError as exc:
        if "artifacts" not in str(exc):
            _fail(str(exc))


def main() -> int:
    tests = [test_compute_defaults, test_audio_keep_mask_not_all_ones]
    with tempfile.TemporaryDirectory(prefix="anemll-demo-") as raw:
        tmp = Path(raw)
        tests.append(lambda: test_api(tmp / "api"))
        tests.append(lambda: test_compressed_audio(tmp))
        tests.append(lambda: test_wave_trim_and_missing_converter(tmp / "trim"))
        tests.append(lambda: test_port_and_artifacts_guard(tmp / "guard"))
        tests.append(lambda: test_package_hash_manifest(tmp / "hash"))
        tests.append(lambda: test_alert(tmp / "alert"))
        for fn in tests:
            fn()
            print(f"OK {getattr(fn, '__name__', 'case')}")
    print(f"OK: {len(tests)} demo api checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
