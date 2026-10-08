#!/usr/bin/env python3
"""API checks for the showcase server (mock backend, no Core AI)."""

from __future__ import annotations

import io
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


def main() -> int:
    tests = [test_compute_defaults, test_audio_keep_mask_not_all_ones]
    with tempfile.TemporaryDirectory(prefix="anemll-demo-") as raw:
        tmp = Path(raw)
        tests.append(lambda: test_api(tmp / "api"))
        tests.append(lambda: test_compressed_audio(tmp))
        tests.append(lambda: test_wave_trim_and_missing_converter(tmp / "trim"))
        tests.append(lambda: test_port_and_artifacts_guard(tmp / "guard"))
        tests.append(lambda: test_package_hash_manifest(tmp / "hash"))
        for fn in tests:
            fn()
            print(f"OK {getattr(fn, '__name__', 'case')}")
    print(f"OK: {len(tests)} demo api checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
