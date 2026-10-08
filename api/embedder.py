"""Core AI runtime: three ANE towers, host embed + scatter, real masks.

Mirrors ``model/parity_coreai_host.py``:

* vision ``vision_s280`` — processor pixels and position ids (pads stay ``-1``)
* audio ``audio_s280`` — processor mel and the real keep-mask, not all ones
* text ``text_embeds_s320`` — host token lookup, soft-token scatter, real mask

The ``.aimodel`` packages stay loaded in a resident worker
(``api/coreai_worker.py``). This module is the host side only.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from PIL import Image

from api.feeds import assert_text_mask_preserved, audio_tower_feed, vision_tower_feed
from api.host_embed import load_slim_host
from api.mock import MockBackend
from api.types import DIM, EmbedResult, TowerHealth
from api.coreai_host import (
    AUDIO_FRAMES,
    TEXT_EMBEDS_S,
    crop_audio_soft_to_src,
    hf_audio_slots_from_frames,
    crop_vision_soft_to_valid,
    expand_media_placeholders,
    interleaved_inputs_embeds,
    pad_embeds_to_package,
)

AUDIO_SR = 16000


def _unit(vector: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if arr.shape != (DIM,) or not np.isfinite(arr).all() or norm < 1e-8:
        raise RuntimeError(f"bad embedding shape={arr.shape} norm={norm}")
    return (arr / norm).astype(np.float32, copy=False)


def package_paths(artifacts: Path) -> dict[str, Path]:
    root = Path(artifacts) / "coreai"
    return {
        "vision": root / "vision_s280.aimodel",
        "audio": root / "audio_s280.aimodel",
        "text": root / "text_embeds_s320.aimodel",
    }


class CoreAIWorkerClient:
    """One long-lived ``api/coreai_worker.py`` process."""

    def __init__(self, packages: dict[str, Path], *, compute: str, python: Path | None) -> None:
        self.packages = packages
        self.compute = compute
        self.python = python
        self._proc: subprocess.Popen[str] | None = None
        self._tmp = tempfile.TemporaryDirectory(prefix="anemll-coreai-")
        self._seq = 0
        self.report: dict[str, Any] = {}

    def startup(self) -> dict[str, Any]:
        if self._proc is not None:
            return self.report
        worker = Path(__file__).resolve().parent / "coreai_worker.py"
        exe = str(self.python) if self.python else sys.executable
        cmd = [exe, str(worker)]
        env = apply_compute_env(self.compute)
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        assert self._proc.stderr is not None
        threading.Thread(target=self._drain, args=(self._proc.stderr,), daemon=True).start()
        self.report = self._request(
            {
                "cmd": "load",
                "compute": self.compute,
                "packages": {key: str(path) for key, path in self.packages.items()},
            }
        )
        return self.report

    def forward(self, tower: str, feed: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
        self._seq += 1
        folder = Path(self._tmp.name)
        npz = folder / f"in_{self._seq}.npz"
        out = folder / f"out_{self._seq}.npy"
        np.savez(npz, **{key: np.ascontiguousarray(val) for key, val in feed.items()})
        reply = self._request(
            {"cmd": "forward", "tower": tower, "npz": str(npz), "out": str(out)}
        )
        elapsed = float(reply.get("latency_ms") or 0.0)
        # One line per Neural Engine forward, so a click's inferences can be counted.
        print(f"[ane] forward tower={tower} {elapsed:.1f} ms", file=sys.stderr, flush=True)
        return np.load(out), elapsed

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            self._request({"cmd": "shutdown"})
        except Exception:
            pass
        if self._proc.poll() is None:
            self._proc.terminate()
        self._tmp.cleanup()
        self._proc = None

    def _drain(self, pipe) -> None:
        for line in pipe:
            sys.stderr.write(f"[coreai] {line}")

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        proc = self._proc
        if proc is None or proc.stdin is None or proc.stdout is None:
            raise RuntimeError("coreai worker is not running")
        proc.stdin.write(json_line(payload))
        proc.stdin.flush()
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("coreai worker closed stdout")
        reply = json_loads(line)
        if not reply.get("ok"):
            raise RuntimeError(reply.get("error") or "coreai worker failed")
        return reply


def apply_compute_env(compute: str, base: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for the worker. Parity stays CPU unless this is set.

    The showcase default is ``ane``. ``model/_coreai_run_npy.py`` still
    treats an unset ``ANEMLL_COREAI_COMPUTE`` as CPU; the worker sets it.
    """
    key = compute.strip().lower()
    if key not in {"ane", "cpu"}:
        raise ValueError(f"compute must be ane or cpu, got {compute!r}")
    env = dict(os.environ if base is None else base)
    env["ANEMLL_COREAI_COMPUTE"] = key
    return env


def json_line(payload: dict[str, Any]) -> str:
    return json.dumps(payload) + "\n"


def json_loads(line: str) -> dict[str, Any]:
    return json.loads(line)


class CoreAIBackend:
    """Host interleave in this process, tower forwards in the worker."""

    name = "coreai"
    dim = DIM

    def __init__(
        self,
        *,
        artifacts: Path | None,
        model_path: Path | None,
        coreai_python: Path | None,
        compute: str = "ane",
        runner: Any | None = None,
        processor: Any | None = None,
        text_model: Any | None = None,
        prompts: dict[str, str] | None = None,
    ) -> None:
        self.artifacts = artifacts
        self.model_path = model_path
        self.coreai_python = coreai_python
        self.compute = compute
        self._runner = runner
        self._proc = processor
        self._text = text_model
        self._prompts = dict(prompts or {})
        self._injected = runner is not None
        self._lock = threading.Lock()
        self._max_samples: int | None = None
        self._min_audio_ms: int | None = None
        self.placement: str | None = None
        self._towers: list[TowerHealth] = []
        self._ready = False
        self._owns_runner = False

    def warmup(self) -> dict:
        if self._ready:
            return self.health()
        if self._runner is None:
            if self.artifacts is None:
                raise FileNotFoundError(
                    "coreai backend needs ANEMLL_EMBEDDINGS_ARTIFACTS "
                    "(directory that contains coreai/vision_s280.aimodel)"
                )
            packages = package_paths(self.artifacts)
            missing = [str(path) for path in packages.values() if not path.exists()]
            if missing:
                raise FileNotFoundError("missing Core AI packages: " + ", ".join(missing))
            self._runner = CoreAIWorkerClient(
                packages, compute=self.compute, python=self.coreai_python
            )
            self._owns_runner = True
        report = self._runner.startup()
        self.placement = report.get("placement")
        towers = report.get("towers") or {}
        self._towers = [
            TowerHealth(
                name=name,
                loaded=bool(row.get("loaded", True)),
                warmup_ms=row.get("warmup_ms"),
                placement=row.get("placement"),
                simulated=False,
            )
            for name, row in towers.items()
        ]
        if self._proc is None or self._text is None:
            self._load_host()
        self._ready = True
        return self.health()

    def health(self) -> dict:
        return {
            "backend": self.name,
            "placement": self.placement,
            "dim": self.dim,
            "compute": self.compute,
            "towers": [row.as_dict() for row in self._towers],
        }

    def close(self) -> None:
        if self._owns_runner and self._runner is not None:
            self._runner.close()

    def embed_text(self, text: str, *, role: str) -> EmbedResult:
        with self._lock:
            return self._embed_text(text, role=role)

    def embed_image(self, image) -> EmbedResult:
        with self._lock:
            return self._embed_image(image)

    def embed_audio(self, wav: np.ndarray, sample_rate: int) -> EmbedResult:
        with self._lock:
            return self._embed_audio(wav, sample_rate)

    def _load_host(self) -> None:
        # Checkpoint stack is the Mac host venv. Tests inject a processor and
        # text module, so this import stays off the mock/CI path.
        from transformers import AutoProcessor

        if self.model_path is None or not Path(self.model_path).is_dir():
            raise FileNotFoundError(
                "coreai backend needs ANEMLL_EMBEDDINGS_MODEL "
                "(EmbeddingGemma 2 checkpoint for the host embed lookup and processor)"
            )
        slim = load_slim_host(self.model_path)
        if slim is not None:
            self._text, self._prompts = slim
            self._proc = AutoProcessor.from_pretrained(
                str(self.model_path), trust_remote_code=True
            )
            return
        from model.embed_wrapper import EmbeddingGemma2Wrapper
        from model.load_text_model import load_sentence_transformer

        st, _meta = load_sentence_transformer(
            self.model_path, dtype=torch.float32, device="cpu"
        )
        wrapper = EmbeddingGemma2Wrapper.from_sentence_transformer(st, normalize=True).eval()
        self._text = wrapper.text_model
        self._prompts = {str(k): str(v) for k, v in (getattr(st, "prompts", {}) or {}).items()}
        self._proc = AutoProcessor.from_pretrained(str(self.model_path), trust_remote_code=True)

    def _prompt(self, text: str, role: str) -> str:
        if role == "query":
            name = "SearchQuery"
        elif role == "document":
            name = "Document"
        else:
            raise ValueError(f"unknown text role {role!r}")
        prefix = self._prompts.get(name) or ""
        if prefix and not text.startswith(prefix):
            return prefix + text
        return text

    def _tokenize(self, text: str):
        tok = self._proc.tokenizer
        batch = tok(text, return_tensors="pt", padding=False, truncation=False)
        return batch["input_ids"], batch["attention_mask"]

    def _run_text(self, embeds, mask) -> tuple[np.ndarray, float]:
        valid = int(mask.sum())
        embeds, mask = pad_embeds_to_package(embeds, mask, seq_len=TEXT_EMBEDS_S)
        packed = mask.detach().cpu().numpy()
        assert_text_mask_preserved(valid, packed)
        feed = {
            "inputs_embeds": np.ascontiguousarray(
                embeds.detach().cpu().numpy().astype(np.float16, copy=False)
            ),
            "attention_mask": np.ascontiguousarray(packed.astype(np.float16, copy=False)),
        }
        raw, elapsed = self._runner.forward("text", feed)
        return _unit(raw), float(elapsed)

    def _scatter(self, ids, *, image_soft, audio_soft):
        tok = self._proc.tokenizer
        img = None if image_soft is None else torch.from_numpy(np.asarray(image_soft)).to(torch.float32)
        aud = None if audio_soft is None else torch.from_numpy(np.asarray(audio_soft)).to(torch.float32)
        return interleaved_inputs_embeds(
            self._text,
            ids,
            image_token_id=int(tok.image_token_id),
            audio_token_id=int(tok.audio_token_id),
            pad_token_id=int(tok.pad_token_id),
            image_soft=img,
            audio_soft=aud,
        )

    def _embed_text(self, text: str, *, role: str) -> EmbedResult:
        prompted = self._prompt(text, role)
        ids, mask = self._tokenize(prompted)
        if int(ids.shape[-1]) > TEXT_EMBEDS_S:
            ids = ids[:, :TEXT_EMBEDS_S]
            mask = mask[:, :TEXT_EMBEDS_S]
        embeds = self._scatter(ids, image_soft=None, audio_soft=None)
        vector, elapsed = self._run_text(embeds, mask)
        return EmbedResult(
            vector=vector,
            latency_ms=elapsed,
            backend=self.name,
            modality="text",
            placement=self.placement,
        )

    def _embed_image(self, image) -> EmbedResult:
        vin = self._proc(text="<|image|>", images=image, return_tensors="pt")
        feed = vision_tower_feed(
            vin["pixel_values"].detach().cpu().numpy(),
            vin["image_position_ids"].detach().cpu().numpy(),
        )
        raw, vision_ms = self._runner.forward("vision", feed)
        soft = crop_vision_soft_to_valid(raw, feed["pixel_position_ids"])
        slots = int(soft.shape[1])
        tok = self._proc.tokenizer
        expanded = expand_media_placeholders(
            "<|image|>",
            image_token=tok.image_token,
            audio_token=tok.audio_token,
            image_slots=slots,
            audio_slots=0,
        )
        ids, mask = self._tokenize(expanded)
        embeds = self._scatter(ids, image_soft=soft, audio_soft=None)
        vector, text_ms = self._run_text(embeds, mask)
        return EmbedResult(
            vector=vector,
            latency_ms=float(vision_ms) + float(text_ms),
            backend=self.name,
            modality="image",
            placement=self.placement,
            extra={"vision_ms": round(float(vision_ms), 3), "text_ms": round(float(text_ms), 3)},
        )

    def _embed_audio(self, wav: np.ndarray, sample_rate: int) -> EmbedResult:
        samples = np.asarray(wav, dtype=np.float32).reshape(-1)
        if int(sample_rate) != AUDIO_SR:
            raise ValueError(f"audio sample rate {sample_rate} != {AUDIO_SR}")
        limit = self._window_samples()
        if samples.size <= limit:
            vector, audio_ms, text_ms = self._embed_audio_window(samples)
            return EmbedResult(
                vector=vector,
                latency_ms=audio_ms + text_ms,
                backend=self.name,
                modality="audio",
                placement=self.placement,
                slices=1,
                extra={"audio_ms": round(audio_ms, 3), "text_ms": round(text_ms, 3)},
            )
        vectors = []
        audio_total = 0.0
        text_total = 0.0
        for start in range(0, samples.size, limit):
            piece = samples[start : start + limit]
            if piece.size < AUDIO_SR // 10:
                break
            vec, audio_ms, text_ms = self._embed_audio_window(piece)
            vectors.append(vec)
            audio_total += audio_ms
            text_total += text_ms
        elapsed = audio_total + text_total
        if not vectors:
            raise ValueError(f"audio too short (min {self.min_audio_ms()} ms)")
        mean = np.mean(np.stack(vectors, axis=0), axis=0)
        return EmbedResult(
            vector=_unit(mean),
            latency_ms=elapsed,
            backend=self.name,
            modality="audio",
            placement=self.placement,
            slices=len(vectors),
            extra={"audio_ms": round(audio_total, 3), "text_ms": round(text_total, 3)},
        )

    def _embed_audio_window(self, wav: np.ndarray) -> tuple[np.ndarray, float, float]:
        ain = self._proc(text="<|audio|>", audio=np.asarray(wav, dtype=np.float32), return_tensors="pt")
        features = ain["input_features"].detach().cpu().numpy()
        src_mask = ain["input_features_mask"].detach().cpu().numpy()
        n_src = min(int(features.shape[1]), AUDIO_FRAMES)
        if hf_audio_slots_from_frames(n_src) < 1:
            raise ValueError(f"audio too short (min {self.min_audio_ms()} ms)")
        feed = audio_tower_feed(features, src_mask)
        raw, audio_ms = self._runner.forward("audio", feed)
        soft = crop_audio_soft_to_src(raw, n_src)
        tok = self._proc.tokenizer
        expanded = expand_media_placeholders(
            "<|audio|>",
            image_token=tok.image_token,
            audio_token=tok.audio_token,
            image_slots=0,
            audio_slots=int(soft.shape[1]),
        )
        ids, mask = self._tokenize(expanded)
        embeds = self._scatter(ids, image_soft=None, audio_soft=soft)
        vector, text_ms = self._run_text(embeds, mask)
        return vector, float(audio_ms), float(text_ms)

    def _frame_count(self, n_samples: int) -> int:
        wav = np.zeros(max(0, int(n_samples)), dtype=np.float32)
        ain = self._proc(text="<|audio|>", audio=wav, return_tensors="pt")
        return int(ain["input_features"].shape[1])

    def min_audio_ms(self) -> int:
        """Shortest 16 kHz clip that yields one mel frame, in milliseconds."""
        if self._min_audio_ms is not None:
            return self._min_audio_ms
        hi = AUDIO_SR * 2
        if self._frame_count(hi) < 1:
            self._min_audio_ms = 2000
            return self._min_audio_ms
        lo = 0
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if self._frame_count(mid) >= 1:
                hi = mid
            else:
                lo = mid
        self._min_audio_ms = max(1, (1000 * hi + AUDIO_SR - 1) // AUDIO_SR)
        return self._min_audio_ms

    def _window_samples(self) -> int:
        if self._max_samples is not None:
            return self._max_samples
        probe = np.zeros(AUDIO_SR, dtype=np.float32)
        ain = self._proc(text="<|audio|>", audio=probe, return_tensors="pt")
        frames = max(1, int(ain["input_features"].shape[1]))
        # Stay inside the 280-frame package. 0.95 leaves headroom for hop rounding.
        seconds = (float(AUDIO_FRAMES) / float(frames)) * 0.95
        self._max_samples = max(AUDIO_SR // 2, int(AUDIO_SR * seconds))
        return self._max_samples


def _env_path(name: str) -> Path | None:
    raw = os.environ.get(name)
    return Path(raw) if raw else None


def default_artifacts() -> Path | None:
    return _env_path("ANEMLL_EMBEDDINGS_ARTIFACTS")


def default_model() -> Path | None:
    return _env_path("ANEMLL_EMBEDDINGS_MODEL")


def default_coreai_python() -> Path | None:
    return _env_path("ANEMLL_COREAI_PYTHON")


TextRole = Literal["query", "document", "SearchQuery", "Document"]


def _public_role(role: str) -> str:
    key = (role or "query").strip()
    if key in {"query", "SearchQuery"}:
        return "query"
    if key in {"document", "Document"}:
        return "document"
    return key


class Embedder:
    """Load the embedding towers and return one 768-d unit vector per input.

    ``backend="coreai"`` (default) runs the ``.aimodel`` packages on the
    Neural Engine. ``backend="mock"`` is a deterministic stand-in when Core
    AI is not installed. ``backend="reference"`` is the full checkpoint on
    CPU.

        from api import Embedder, cosine
        embedder = Embedder(compute="ane")
        q = embedder.embed_text("a red fox")          # shape (768,)
        d = embedder.embed_text("a fox", role="document")
        print(cosine(q, d))
        embedder.close()

    See `api/README.md` for copy-paste examples. The demo uses this class
    for every page (`demo/server.py`, `demo/alert_routes.py`).
    """

    def __init__(
        self,
        artifacts: Path | str | None = None,
        model: Path | str | None = None,
        coreai_python: Path | str | None = None,
        compute: str = "ane",
        backend: str = "coreai",
        **kwargs: Any,
    ) -> None:
        """Create an embedder.

        Args:
            artifacts: Directory that contains ``coreai/vision_s280.aimodel``
                (and the audio / text-embeds packages). Defaults to
                ``ANEMLL_EMBEDDINGS_ARTIFACTS``.
            model: EmbeddingGemma 2 checkpoint directory (host tokenizer and
                embedding table). Defaults to ``ANEMLL_EMBEDDINGS_MODEL``.
            coreai_python: Interpreter that can ``import coreai.runtime``.
                Defaults to ``ANEMLL_COREAI_PYTHON``.
            compute: ``"ane"`` (Neural Engine) or ``"cpu"``.
            backend: ``"coreai"``, ``"mock"``, or ``"reference"``.
        """
        key = (backend or "coreai").strip().lower()
        self.name = key
        self.last: EmbedResult | None = None
        art = Path(artifacts) if artifacts is not None else default_artifacts()
        ckpt = Path(model) if model is not None else default_model()
        py = Path(coreai_python) if coreai_python is not None else default_coreai_python()
        if key == "mock":
            self._impl = MockBackend()
        elif key == "reference":
            # Sentence-Transformers lives in the optional reference backend;
            # importing it here would pull that stack into every Embedder().
            from demo.backends.reference import ReferenceBackend

            self._impl = ReferenceBackend(model_path=ckpt)
        elif key == "coreai":
            self._impl = CoreAIBackend(
                artifacts=art,
                model_path=ckpt,
                coreai_python=py,
                compute=compute,
                **kwargs,
            )
        else:
            raise ValueError(f"unknown backend {backend!r} (expected mock, reference, or coreai)")

    @property
    def placement(self) -> str | None:
        """Device label from the last warmup (``fullyOnANE``, ``mock``, …)."""
        return getattr(self._impl, "placement", None)

    def warmup(self) -> dict[str, Any]:
        """Load towers once. ``embed_*`` calls this automatically."""
        return self._impl.warmup()

    def health(self) -> dict[str, Any]:
        """Backend name, placement, and per-tower warmup (same shape as ``GET /health``)."""
        return self._impl.health()

    def close(self) -> None:
        """Release the worker process (no-op for mock)."""
        self._impl.close()

    def embed_text(self, text: str, *, role: TextRole | str = "query") -> np.ndarray:
        """Embed a sentence.

        Args:
            text: Raw string. Do not add the ``SearchQuery`` / ``Document``
                prefix yourself; ``role`` does that.
            role: ``"query"`` (search) or ``"document"`` (index). The
                Sentence-Transformers names ``SearchQuery`` / ``Document``
                are accepted too.

        Returns:
            L2-normalized ``float32`` vector of shape ``(768,)``.
        """
        self._impl.warmup()
        result = self._impl.embed_text(text, role=_public_role(str(role)))
        return self._take(result)

    def embed_image(self, image: Image.Image) -> np.ndarray:
        """Embed an RGB PIL image.

        Returns:
            L2-normalized ``float32`` vector of shape ``(768,)``.
        """
        self._impl.warmup()
        result = self._impl.embed_image(image)
        return self._take(result)

    def embed_audio(self, wav: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Embed a mono waveform.

        Args:
            wav: Samples as float32, typically in ``[-1, 1]``.
            sample_rate: Must be 16 kHz for the Core AI audio tower. The
                demo resamples uploads before calling this.

        Returns:
            L2-normalized ``float32`` vector of shape ``(768,)``.

        Raises:
            ValueError: clip too short to produce one mel frame (~9 ms).
        """
        self._impl.warmup()
        result = self._impl.embed_audio(np.asarray(wav, dtype=np.float32), int(sample_rate))
        return self._take(result)

    def _take(self, result: EmbedResult) -> np.ndarray:
        self.last = result
        vec = np.asarray(result.vector, dtype=np.float32).reshape(-1)
        if vec.shape != (DIM,):
            raise RuntimeError(f"embedding shape {vec.shape} != ({DIM},)")
        return vec


