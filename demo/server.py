"""Local showcase server for the three Core AI embedding towers.

Loads the chosen backend once at startup. ``coreai`` keeps the ``.aimodel``
packages resident and runs them with the parity runner's ANE/CPU switch,
defaulting to the Neural Engine. ``mock`` is a deterministic stand-in for
machines that cannot run Core AI.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid
import wave
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from api.runtime_paths import default_artifacts, default_model
from api.types import EmbedResult
from demo.backends import open_backend
from demo.media_io import (
    AUDIO_SR,
    load_audio_bytes,
    load_image_bytes,
    read_wav_bytes,
    resample_linear,
    save_jpeg,
    sniff_modality,
)
from demo.alert_routes import mount_alert
from demo.limits import BodySizeLimit
from demo.settings import (
    DEFAULT_PORT,
    assert_outside_artifacts,
    bind_warning,
    default_alert_dir,
    default_data_dir,
    env_path,
    max_json_bytes,
    max_upload_bytes,
    server_compute,
    server_host,
)
from demo.store import VectorStore

REPO_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_AUDIO_BYTES = 30 * 1024 * 1024
PAGES = {
    "/": "index.html",
    "/heatmap": "heatmap.html",
    "/heard": "heard.html",
    "/alert": "alert.html",
}


def create_app(
    *,
    backend: str = "mock",
    data_dir: Path | None = None,
    alert_dir: Path | None = None,
    artifacts: Path | None = None,
    model: Path | None = None,
    coreai_python: Path | None = None,
    compute: str | None = None,
    embedder: Any | None = None,
    max_upload: int | None = None,
    max_json: int | None = None,
) -> FastAPI:
    # Same defaults as ``Embedder``: env var, then ~/.anemll-embeddings/... if downloaded.
    art = artifacts if artifacts is not None else default_artifacts()
    data = assert_outside_artifacts(
        Path(data_dir) if data_dir is not None else default_data_dir(),
        art,
    )
    alert = assert_outside_artifacts(
        Path(alert_dir) if alert_dir is not None else default_alert_dir(),
        art,
    )
    chosen = server_compute(compute)
    store = VectorStore(data)
    if embedder is None:
        # Same public class as ``from api import Embedder``.
        embedder = open_backend(
            backend,
            artifacts=art,
            model=model if model is not None else default_model(),
            coreai_python=(
                coreai_python
                if coreai_python is not None
                else env_path("ANEMLL_COREAI_PYTHON")
            ),
            compute=chosen,
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.ready = await asyncio.to_thread(app.state.backend.warmup)
        try:
            yield
        finally:
            app.state.backend.close()

    app = FastAPI(title="anemll embeddings demo", lifespan=lifespan)
    app.add_middleware(
        BodySizeLimit,
        max_upload=max_upload if max_upload is not None else max_upload_bytes(),
        max_other=max_json if max_json is not None else max_json_bytes(),
    )
    app.state.backend = embedder
    app.state.store = store
    app.state.compute = chosen
    app.state.ready = None
    mount_alert(app, alert)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        report = app.state.backend.health()
        towers = report.get("towers") or []
        return {
            "ok": True,
            "backend": report.get("backend"),
            "placement": report.get("placement"),
            "compute": report.get("compute", app.state.compute if report.get("backend") == "coreai" else None),
            "dim": report.get("dim"),
            "index_size": len(app.state.store),
            "towers": towers,
            "warmup": {
                row["name"]: row.get("warmup_ms")
                for row in towers
                if row.get("warmup_ms") is not None
            },
        }

    @app.post("/embed")
    async def embed(request: Request) -> dict[str, Any]:
        incoming = await _read_input(request, default_role="query")
        result = await _embed_incoming(app.state.backend, incoming)
        return result.as_dict()

    @app.post("/index")
    async def index(request: Request) -> dict[str, Any]:
        incoming = await _read_input(request, default_role="document")
        result = await _embed_incoming(app.state.backend, incoming)
        filename = _store_media(app.state.store, incoming)
        item = app.state.store.add(
            result.vector,
            modality=incoming["modality"],
            label=incoming.get("label"),
            text=incoming.get("text") if incoming["modality"] == "text" else None,
            session=incoming.get("session"),
            t_start=incoming.get("t_start"),
            t_end=incoming.get("t_end"),
            credit=incoming.get("credit"),
            source=incoming.get("source"),
            filename=filename,
        )
        return {
            "item": item,
            "latency_ms": result.as_dict()["latency_ms"],
            "backend": result.backend,
            "placement": result.placement,
            "slices": result.slices,
        }

    @app.post("/search")
    async def search(request: Request) -> dict[str, Any]:
        incoming = await _read_input(request, default_role="query")
        raw_k = incoming.get("k")
        k = 8 if raw_k is None else int(raw_k)
        if k < 1 or k > 50:
            raise HTTPException(400, "k must be between 1 and 50")
        result = await _embed_incoming(app.state.backend, incoming)
        hits = app.state.store.search(
            result.vector,
            k,
            modality=incoming.get("filter_modality"),
            session=incoming.get("session"),
        )
        payload = result.as_dict()
        payload.pop("vector", None)
        return {"query": payload, "results": hits}

    @app.post("/compare")
    async def compare(request: Request) -> dict[str, Any]:
        body = await request.json()
        ids = list(body.get("ids") or [])
        if not ids:
            raise HTTPException(400, "ids is required")
        if len(ids) > 12:
            raise HTTPException(400, "compare accepts at most 12 items")
        rows = []
        vectors = []
        per_item = []
        for item_id in ids:
            item = app.state.store.get(str(item_id))
            if item is None:
                raise HTTPException(404, f"unknown id {item_id}")
            result = await _embed_stored(app.state.backend, app.state.store, item)
            vectors.append(np.asarray(result.vector, dtype=np.float32))
            per_item.append(round(float(result.latency_ms), 3))
            rows.append(app.state.store.public(item))
        mat = np.stack(vectors, axis=0)
        scores = (mat @ mat.T).astype(np.float64)
        placement = app.state.backend.placement
        return {
            "items": rows,
            "matrix": scores.round(6).tolist(),
            "per_item_ms": per_item,
            "latency_ms": round(float(sum(per_item)), 3),
            "backend": app.state.backend.name,
            "placement": placement,
        }

    @app.get("/items")
    async def items(modality: str | None = None, session: str | None = None) -> dict[str, Any]:
        return {"items": app.state.store.list_items(modality=modality, session=session)}

    @app.delete("/items/{item_id}")
    async def delete_item(item_id: str) -> dict[str, bool]:
        if not app.state.store.delete(item_id):
            raise HTTPException(404, "unknown id")
        return {"ok": True}

    @app.get("/media/{item_id}")
    async def media(item_id: str) -> FileResponse:
        path = app.state.store.media_path(item_id)
        if path is None:
            raise HTTPException(404, "no media for this item")
        media_type = "audio/wav" if path.suffix == ".wav" else "image/jpeg"
        return FileResponse(path, media_type=media_type)

    for route, name in PAGES.items():
        app.add_api_route(
            route,
            _page_handler(name),
            methods=["GET"],
            include_in_schema=False,
        )

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.exception_handler(ValueError)
    async def _value_error(_request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    return app


def _page_handler(filename: str):
    def page() -> FileResponse:
        return FileResponse(STATIC_DIR / filename)

    page.__name__ = f"page_{Path(filename).stem}"
    return page


def _as_result(embedder: Any, value: Any, *, modality: str) -> EmbedResult:
    """Accept ``Embedder`` (vector + ``.last``) or a backend that returns ``EmbedResult``."""
    if isinstance(value, EmbedResult):
        return value
    last = getattr(embedder, "last", None)
    if isinstance(last, EmbedResult):
        return last
    return EmbedResult(
        vector=np.asarray(value, dtype=np.float32),
        latency_ms=0.0,
        backend=getattr(embedder, "name", "unknown"),
        modality=modality,
        placement=getattr(embedder, "placement", None),
    )


def _call_embed(embedder: Any, incoming: dict[str, Any]) -> EmbedResult:
    modality = incoming["modality"]
    if modality == "text":
        out = embedder.embed_text(incoming["text"], role=incoming.get("role") or "query")
        return _as_result(embedder, out, modality="text")
    if modality == "image":
        out = embedder.embed_image(incoming["image"])
        return _as_result(embedder, out, modality="image")
    if modality == "audio":
        out = embedder.embed_audio(incoming["wav"], incoming.get("sample_rate") or AUDIO_SR)
        return _as_result(embedder, out, modality="audio")
    raise HTTPException(400, f"unknown modality {modality}")


async def _embed_incoming(backend: Any, incoming: dict[str, Any]) -> EmbedResult:
    return await asyncio.to_thread(_call_embed, backend, incoming)


async def _embed_stored(backend: Any, store: VectorStore, item: dict[str, Any]) -> EmbedResult:
    modality = item.get("modality")
    if modality == "text":
        text = item.get("text") or item.get("label") or ""

        def _text() -> Any:
            return backend.embed_text(text, role="document")

        out = await asyncio.to_thread(_text)
        return _as_result(backend, out, modality="text")
    path = store.media_path(item["id"])
    if path is None:
        raise HTTPException(400, f"item {item['id']} has no media to re-embed")
    if modality == "image":
        image = Image.open(path).convert("RGB")
        out = await asyncio.to_thread(backend.embed_image, image)
        return _as_result(backend, out, modality="image")
    if modality == "audio":
        samples, rate = read_wav_bytes(path.read_bytes())
        if samples.ndim > 1:
            samples = samples.mean(axis=-1)
        if rate != AUDIO_SR:
            samples = resample_linear(samples, rate, AUDIO_SR)
            rate = AUDIO_SR
        out = await asyncio.to_thread(backend.embed_audio, samples, rate)
        return _as_result(backend, out, modality="audio")
    raise HTTPException(400, f"unknown modality {modality}")


def _store_media(store: VectorStore, incoming: dict[str, Any]) -> str | None:
    """Write a JPEG or 16 kHz wav next to the index. Returns the filename."""
    modality = incoming["modality"]
    if modality == "text":
        return None
    # The id is assigned in the store; write under a temporary name then the
    # caller only has the filename. Use a content hash of the bytes so the
    # file exists before add() returns an id — the store records the name.
    if modality == "image":
        name = uuid.uuid4().hex[:16] + ".jpg"
        save_jpeg(incoming["image"], store.media_dir / name, max_edge=1024)
        return name
    name = uuid.uuid4().hex[:16] + ".wav"
    _write_wav(store.media_dir / name, incoming["wav"], AUDIO_SR)
    return name


def _write_wav(path: Path, samples: np.ndarray, rate: int) -> None:
    pcm = np.clip(np.asarray(samples, dtype=np.float32).reshape(-1), -1.0, 1.0)
    ints = (pcm * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(rate))
        handle.writeframes(ints.tobytes())


async def _read_input(request: Request, *, default_role: str) -> dict[str, Any]:
    ctype = request.headers.get("content-type", "")
    if "application/json" in ctype:
        body = await request.json()
        text = str(body.get("text") or "").strip()
        if not text:
            raise HTTPException(400, "text is required")
        return {
            "modality": "text",
            "text": text,
            "role": body.get("role") or default_role,
            "label": body.get("label"),
            "session": body.get("session"),
            "t_start": body.get("t_start"),
            "t_end": body.get("t_end"),
            "credit": body.get("credit"),
            "source": body.get("source"),
            "k": body.get("k"),
            "filter_modality": body.get("filter_modality"),
        }
    form = await request.form()
    text = _form_text(form, "text")
    upload = _form_file(form)
    label = _form_text(form, "label")
    session = _form_text(form, "session")
    credit = _form_text(form, "credit")
    source = _form_text(form, "source")
    role = _form_text(form, "role") or default_role
    filt = _form_text(form, "filter_modality")
    k = _form_text(form, "k")
    t_start = _form_float(form, "t_start")
    t_end = _form_float(form, "t_end")
    common = {
        "label": label,
        "session": session,
        "t_start": t_start,
        "t_end": t_end,
        "credit": credit,
        "source": source,
        "role": role,
        "k": int(k) if k else None,
        "filter_modality": filt,
    }
    if upload is not None and text:
        raise HTTPException(400, "send text or one file, not both")
    if text:
        return {"modality": "text", "text": text, **common}
    if upload is None:
        raise HTTPException(400, "text or a file is required")
    data = await upload.read()
    filename = getattr(upload, "filename", None) or ""
    content_type = getattr(upload, "content_type", None)
    kind = sniff_modality(filename, content_type)
    if kind == "image":
        if len(data) > MAX_IMAGE_BYTES:
            raise HTTPException(413, "image is too large")
        return {"modality": "image", "image": load_image_bytes(data), **common}
    if kind == "audio":
        if len(data) > MAX_AUDIO_BYTES:
            raise HTTPException(413, "audio is too large")
        suffix = Path(filename).suffix.lower() or ".webm"
        wav = load_audio_bytes(data, suffix)
        return {"modality": "audio", "wav": wav, "sample_rate": AUDIO_SR, **common}
    raise HTTPException(400, "expected an image or audio file")


def _form_text(form, key: str) -> str | None:
    value = form.get(key)
    if value is None or hasattr(value, "read"):
        return None
    text = str(value).strip()
    return text or None


def _form_float(form, key: str) -> float | None:
    text = _form_text(form, key)
    if text is None:
        return None
    try:
        return float(text)
    except ValueError as exc:
        raise HTTPException(400, f"{key} must be a number") from exc


def _form_file(form):
    for key in ("file", "image", "audio"):
        value = form.get(key)
        if value is not None and hasattr(value, "read"):
            return value
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default=os.environ.get("ANEMLL_DEMO_BACKEND", "mock"))
    parser.add_argument(
        "--host",
        default=None,
        help="bind address (default: $ANEMLL_DEMO_HOST or 127.0.0.1). "
        "Use 0.0.0.0 to reach the demo from other machines on a trusted LAN; "
        "there is no authentication.",
    )
    parser.add_argument("--port", type=int, default=int(os.environ.get("ANEMLL_DEMO_PORT", str(DEFAULT_PORT))))
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--alert-dir", type=Path, default=None)
    parser.add_argument("--artifacts", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--coreai-python", type=Path, default=None)
    parser.add_argument(
        "--compute",
        default=None,
        help="coreai device: ane (default) or cpu. Reads --compute / ANEMLL_DEMO_COMPUTE, not shell ANEMLL_COREAI_COMPUTE.",
    )
    args = parser.parse_args(argv)
    app = create_app(
        backend=args.backend,
        data_dir=args.data_dir,
        alert_dir=args.alert_dir,
        artifacts=args.artifacts,
        model=args.model,
        coreai_python=args.coreai_python,
        compute=args.compute,
    )
    host = server_host(args.host)
    warning = bind_warning(host, args.port)
    if warning:
        print(warning, file=sys.stderr, flush=True)
    uvicorn.run(app, host=host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
