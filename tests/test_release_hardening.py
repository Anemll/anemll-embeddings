#!/usr/bin/env python3
"""Release-readiness checks: runtime paths, IPC cleanup, warmup flags,
request limits, bind warning, checksums, and packaging (no Core AI)."""

from __future__ import annotations

import asyncio
import io
import os
import sys
import tomllib
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import api.runtime_paths as runtime_paths  # noqa: E402
from api.embedder import CoreAIWorkerClient, worker_python  # noqa: E402
from api.runtime_paths import (  # noqa: E402
    CoreAIPythonNotFound,
    default_artifacts,
    default_model,
    resolve_coreai_python,
)
from demo.limits import BodySizeLimit  # noqa: E402
from demo.media_io import load_image_bytes  # noqa: E402
from demo.server import create_app  # noqa: E402
from demo.settings import bind_warning, is_loopback, server_host  # noqa: E402
from scripts import warmup  # noqa: E402
from scripts.download_common import (  # noqa: E402
    HOST_SHA256,
    TOWERS,
    env_exports,
    host_checksums,
    sha256_file,
    verify_sha256,
)

_ENV = (
    "ANEMLL_COREAI_PYTHON",
    "ANEMLL_EMBEDDINGS_ARTIFACTS",
    "ANEMLL_EMBEDDINGS_MODEL",
    "ANEMLL_EMBEDDINGS_HOME",
    "ANEMLL_DEMO_HOST",
    "CFFIXED_USER_HOME",
)


@pytest.fixture
def clean_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for key in _ENV:
        monkeypatch.delenv(key, raising=False)
    home = tmp_path / "anemll-home"
    monkeypatch.setenv("ANEMLL_EMBEDDINGS_HOME", str(home))
    # No forge checkout next to the repo / in $HOME during these tests.
    monkeypatch.setattr(
        runtime_paths, "coreai_python_candidates", lambda: (home / "coreai-venv" / "bin" / "python",)
    )
    return home


# ---------- Core AI interpreter + default paths ----------

def test_coreai_python_env_missing_is_actionable(clean_env: Path, monkeypatch) -> None:
    monkeypatch.setenv("ANEMLL_COREAI_PYTHON", str(clean_env / "nope" / "python"))
    with pytest.raises(CoreAIPythonNotFound) as err:
        resolve_coreai_python()
    text = str(err.value)
    assert "ANEMLL_COREAI_PYTHON" in text and "coreai-core" in text and "README" in text


def test_coreai_python_none_found_lists_locations(clean_env: Path) -> None:
    with pytest.raises(CoreAIPythonNotFound) as err:
        resolve_coreai_python()
    assert "looked in" in str(err.value)
    assert "ANEMLL_COREAI_PYTHON" in str(err.value)
    assert resolve_coreai_python(required=False) is None


def test_coreai_python_candidate_used_when_env_unset(clean_env: Path) -> None:
    py = clean_env / "coreai-venv" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("#!/bin/sh\n")
    assert resolve_coreai_python() == py


def test_worker_python_refuses_interpreter_without_coreai(clean_env: Path) -> None:
    with patch("api.embedder.current_python_has_coreai", return_value=False):
        with pytest.raises(CoreAIPythonNotFound) as err:
            worker_python(None)
    assert "cannot import coreai.runtime" in str(err.value)
    with patch("api.embedder.current_python_has_coreai", return_value=True):
        assert worker_python(None) is None


def test_no_user_specific_paths_in_runtime_code() -> None:
    for rel in ("api", "scripts", "model", "demo", "samples"):
        for path in (REPO_ROOT / rel).rglob("*"):
            if path.suffix not in {".py", ".sh"}:
                continue
            text = path.read_text(encoding="utf-8")
            assert "/Users/anemll" not in text, path
            assert "/Volumes/" not in text, path


def test_default_paths_follow_download_layout(clean_env: Path) -> None:
    assert default_artifacts() is None and default_model() is None
    (clean_env / "artifacts" / "coreai").mkdir(parents=True)
    (clean_env / "embeddinggemma-2").mkdir(parents=True)
    assert default_artifacts() == clean_env / "artifacts"
    assert default_model() == clean_env / "embeddinggemma-2"


def test_env_beats_default_paths(clean_env: Path, monkeypatch) -> None:
    (clean_env / "artifacts" / "coreai").mkdir(parents=True)
    monkeypatch.setenv("ANEMLL_EMBEDDINGS_ARTIFACTS", "/tmp/elsewhere")
    assert default_artifacts() == Path("/tmp/elsewhere")


# ---------- IPC temp files ----------

def _client(tmp_path: Path) -> CoreAIWorkerClient:
    return CoreAIWorkerClient({}, compute="ane", python=None)


def test_forward_removes_ipc_files(tmp_path: Path) -> None:
    client = _client(tmp_path)
    folder = Path(client._tmp.name)

    def fake_request(payload):
        np.save(payload["out"], np.ones((1, 768), dtype=np.float16))
        return {"ok": True, "latency_ms": 1.5}

    client._request = fake_request
    for _ in range(3):
        out, ms = client.forward("text", {"x": np.zeros((2, 2), dtype=np.float16)})
        assert out.shape == (1, 768) and ms == 1.5
    assert list(folder.iterdir()) == []
    client._tmp.cleanup()


def test_forward_removes_ipc_files_on_error(tmp_path: Path) -> None:
    client = _client(tmp_path)
    folder = Path(client._tmp.name)

    def failing(payload):
        np.save(payload["out"], np.zeros(1))
        raise RuntimeError("worker failed")

    client._request = failing
    with pytest.raises(RuntimeError):
        client.forward("vision", {"x": np.zeros(4, dtype=np.float16)})
    assert list(folder.iterdir()) == []
    client._tmp.cleanup()


# ---------- warmup flags ----------

def test_warmup_flags_parse() -> None:
    args = warmup.build_parser().parse_args(["--require-ane", "--compute", "ane"])
    assert args.require_ane and args.cache_dir is None and args.coreai_home is None
    with pytest.raises(SystemExit):
        warmup.build_parser().parse_args(["--cache-dir", "/a", "--coreai-home", "/b"])


def test_warmup_cache_dir_rejects_unsupported(tmp_path: Path) -> None:
    args = warmup.build_parser().parse_args(["--cache-dir", str(tmp_path / "my-cache")])
    with pytest.raises(SystemExit) as err:
        warmup.resolve_coreai_home(args)
    assert "Library/Caches/coreai-cache" in str(err.value)


def test_warmup_cache_dir_and_home_map_to_cffixed(tmp_path: Path) -> None:
    shaped = tmp_path / "h" / "Library" / "Caches" / "coreai-cache"
    args = warmup.build_parser().parse_args(["--cache-dir", str(shaped)])
    assert warmup.resolve_coreai_home(args) == (tmp_path / "h").resolve()
    args = warmup.build_parser().parse_args(["--coreai-home", str(tmp_path / "x")])
    assert warmup.resolve_coreai_home(args) == (tmp_path / "x").resolve()


def test_towers_not_on_ane() -> None:
    report = {
        "towers": {
            "vision_s280": {"placement": "GPU"},
            "audio_s280": {"placement": "fullyOnANE"},
            "text_embeds_s320": {"placement": "ANE+GPU"},
        }
    }
    assert warmup.towers_not_on_ane(report) == ["vision_s280", "text_embeds_s320"]
    assert warmup.towers_not_on_ane({"towers": {"a": {"placement": "fullyOnANE"}}}) == []
    assert warmup.towers_not_on_ane({}) == ["<no towers reported>"]


@pytest.mark.parametrize("placement,code", [("fullyOnANE", 0), ("GPU", warmup.EXIT_NOT_ON_ANE)])
def test_warmup_require_ane_exit_code(tmp_path: Path, clean_env: Path, placement: str, code: int) -> None:
    coreai = tmp_path / "artifacts" / "coreai"
    for name in TOWERS:
        (coreai / f"{name}.aimodel").mkdir(parents=True)
    py = tmp_path / "py"
    py.write_text("#!/bin/sh\n")

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def startup(self):
            return {
                "towers": {name: {"placement": placement, "load_ms": 1.0, "warmup_ms": 2.0} for name in TOWERS},
                "placement": placement,
            }

        def close(self):
            pass

    with patch.object(warmup, "CoreAIWorkerClient", FakeClient):
        rc = warmup.main(
            ["--artifacts", str(tmp_path / "artifacts"), "--coreai-python", str(py), "--require-ane",
             "--coreai-home", str(tmp_path / "home")]
        )
    assert rc == code
    assert os.environ.get("CFFIXED_USER_HOME") == str((tmp_path / "home").resolve())
    os.environ.pop("CFFIXED_USER_HOME", None)


# ---------- demo: limits, bind host ----------

def _app(tmp_path: Path, **kw):
    return create_app(backend="mock", data_dir=tmp_path / "data", alert_dir=tmp_path / "alert", **kw)


def test_request_size_limits(tmp_path: Path) -> None:
    app = _app(tmp_path, max_upload=4096, max_json=256)
    with TestClient(app) as client:
        ok = client.post("/embed", json={"text": "a red fox"})
        assert ok.status_code == 200, ok.text
        big = client.post("/embed", json={"text": "x" * 1000})
        assert big.status_code == 413
        buf = io.BytesIO()
        Image.new("RGB", (8, 8)).save(buf, "PNG")
        small = client.post("/embed", files={"file": ("a.png", buf.getvalue(), "image/png")})
        assert small.status_code == 200, small.text
        huge = client.post("/embed", files={"file": ("a.png", b"\0" * 10_000, "image/png")})
        assert huge.status_code == 413
        assert "too large" in huge.json()["detail"]


def test_streamed_body_without_length_is_capped() -> None:
    seen: dict = {}

    async def app(scope, receive, send):
        body = b""
        while True:
            message = await receive()
            body += message.get("body", b"")
            if not message.get("more_body"):
                break
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    chunks = [{"type": "http.request", "body": b"x" * 100, "more_body": True} for _ in range(5)]
    chunks.append({"type": "http.request", "body": b"", "more_body": False})

    async def receive():
        return chunks.pop(0)

    async def send(message):
        if message["type"] == "http.response.start":
            seen["status"] = message["status"]

    scope = {"type": "http", "method": "POST", "headers": [(b"content-type", b"application/json")]}
    asyncio.run(BodySizeLimit(app, max_upload=10_000, max_other=250)(scope, receive, send))
    assert seen["status"] == 413


def test_bind_host_default_and_warning(monkeypatch) -> None:
    monkeypatch.delenv("ANEMLL_DEMO_HOST", raising=False)
    assert server_host(None) == "127.0.0.1"
    monkeypatch.setenv("ANEMLL_DEMO_HOST", "0.0.0.0")
    assert server_host(None) == "0.0.0.0"
    assert server_host("127.0.0.1") == "127.0.0.1"
    for host in ("127.0.0.1", "localhost", "::1"):
        assert is_loopback(host) and bind_warning(host, 8766) is None
    text = bind_warning("0.0.0.0", 8766)
    assert text and "WARNING" in text and "no authentication" in text


def test_decoded_image_pixel_cap() -> None:
    buf = io.BytesIO()
    Image.new("RGB", (20, 20)).save(buf, "PNG")
    with pytest.raises(ValueError, match="at most"):
        load_image_bytes(buf.getvalue(), max_pixels=100)
    assert load_image_bytes(buf.getvalue()).size == (20, 20)


# ---------- downloads: quoting + checksums ----------

def test_env_exports_are_shell_quoted() -> None:
    text = env_exports(artifacts=Path("/tmp/my dir/a"), model=Path("/tmp/m$x"), coreai_python=None)
    assert "export ANEMLL_EMBEDDINGS_ARTIFACTS='/tmp/my dir/a'" in text
    assert "export ANEMLL_EMBEDDINGS_MODEL='/tmp/m$x'" in text
    assert "# ANEMLL_COREAI_PYTHON" in text


def test_verify_sha256_and_host_conflicts(tmp_path: Path) -> None:
    good = tmp_path / "a.bin"
    good.write_bytes(b"hello")
    digest = sha256_file(good)
    assert verify_sha256({good: digest}) == []
    assert "mismatch" in verify_sha256({good: "0" * 64})[0]
    assert "missing" in verify_sha256({tmp_path / "nope": digest})[0]
    (tmp_path / "SHA256SUMS").write_text(f"{'1' * 64}  config.json\n{digest}  LICENSE\n")
    files, conflicts = host_checksums(tmp_path)
    assert files[tmp_path / "config.json"] == HOST_SHA256["config.json"]
    assert files[tmp_path / "LICENSE"] == digest
    assert conflicts and "config.json" in conflicts[0]


def test_download_ane_rejects_bad_checksum(tmp_path: Path, monkeypatch) -> None:
    from scripts import download_models

    monkeypatch.delenv("ANEMLL_ANE_REVISION", raising=False)

    def fake(**kwargs):
        root = Path(kwargs["local_dir"])
        for name in TOWERS:
            bundle = root / name / f"{name}.aimodel"
            bundle.mkdir(parents=True, exist_ok=True)
            for filename in ("metadata.json", "main.hash", "main.mlirb"):
                (bundle / filename).write_text("tampered\n")

    with patch("scripts.download_common._hf_snapshot_download", fake):
        with pytest.raises(SystemExit) as err:
            download_models.download_ane(tmp_path / "ane", force=True)
    assert "checksum" in str(err.value)
    assert not (tmp_path / "ane" / ".anemll-revision").exists()


def test_prepare_host_temp_source_is_removed(tmp_path: Path) -> None:
    from scripts import prepare_hf_host_folder as prep

    seen: list[Path] = []

    def fake_snapshot(repo, rev, local_dir, **kw):
        seen.append(Path(local_dir))

    with patch.object(prep, "snapshot", fake_snapshot):
        with prep.resolve_src(None) as (src, kind):
            assert kind == "downloaded" and src.is_dir()
    assert seen and not seen[0].exists()


# ---------- samples: bounded download ----------

class _FakeResponse(io.BytesIO):
    def __init__(self, data: bytes, length: str | None = None):
        super().__init__(data)
        self.headers = {} if length is None else {"Content-Length": length}

    def geturl(self):
        return "https://example.org/final"

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_sample_download_cap_and_provenance(tmp_path: Path) -> None:
    from samples import download_utils

    with patch.object(download_utils.urllib.request, "urlopen", lambda *a, **k: _FakeResponse(b"abc")):
        rec = download_utils.download("https://example.org/x", tmp_path / "x.bin", user_agent="t", max_bytes=10)
    assert rec["final_url"] == "https://example.org/final" and rec["bytes"] == 3
    assert rec["sha256"] == sha256_file(tmp_path / "x.bin") and rec["fetched_at"].endswith("Z")
    with patch.object(download_utils.urllib.request, "urlopen", lambda *a, **k: _FakeResponse(b"x" * 50)):
        with pytest.raises(download_utils.DownloadTooLarge):
            download_utils.download("https://example.org/y", tmp_path / "y.bin", user_agent="t", max_bytes=10)
    assert not (tmp_path / "y.bin").exists()
    assert not list(tmp_path.glob("*.part"))


# ---------- packaging ----------

def test_pyproject_pins_extras_and_package_data() -> None:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = data["project"]["optional-dependencies"]
    runtime = " ".join(extras["runtime"])
    assert "transformers>=5.19,<5.20" in runtime
    assert "torch>=2.14,<2.15" in runtime and "torchvision>=0.29,<0.30" in runtime
    assert extras["reference"] == ["sentence-transformers>=6.1,<6.2"]
    assert any("pytest" in item for item in extras["test"])
    assert any("coremltools" in item for item in extras["test"])
    assert "static/*" in data["tool"]["setuptools"]["package-data"]["demo"]
    constraints = (REPO_ROOT / "constraints.txt").read_text(encoding="utf-8")
    for pin in ("torch==2.14.1", "torchvision==0.29.1", "transformers==5.19.0", "sentence-transformers==6.1.0"):
        assert pin in constraints
