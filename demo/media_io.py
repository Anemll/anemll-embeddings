"""Decode browser uploads into the shapes the embedder expects.

Images become RGB PIL. Audio becomes mono float32 at 16 kHz, which is the
rate ``src.multimodal_media.AUDIO_SR`` and the EmbeddingGemma processor use.
WAV is decoded in-process. WebM, Ogg, and other browser recordings go through
ffmpeg, then the same 16 kHz mono conversion.
"""

from __future__ import annotations

import io
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np
from PIL import Image

AUDIO_SR = 16000

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff"}
_AUDIO_SUFFIXES = {".wav", ".wave", ".webm", ".ogg", ".oga", ".mp3", ".m4a", ".mp4", ".flac", ".aac"}


def sniff_modality(filename: str | None, content_type: str | None) -> str | None:
    """Return ``image`` or ``audio`` from a filename / MIME type."""
    ctype = (content_type or "").lower()
    if ctype.startswith("image/"):
        return "image"
    if ctype.startswith("audio/") or ctype in {"video/webm", "video/ogg"}:
        return "audio"
    suffix = Path(filename or "").suffix.lower()
    if suffix in _IMAGE_SUFFIXES:
        return "image"
    if suffix in _AUDIO_SUFFIXES:
        return "audio"
    return None


def load_image_bytes(data: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception as exc:
        raise ValueError(f"could not read image: {exc}") from exc
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def save_jpeg(image: Image.Image, path: Path, *, max_edge: int = 1024, quality: int = 86) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = image.copy()
    out.thumbnail((int(max_edge), int(max_edge)))
    out.save(path, "JPEG", quality=int(quality))


def load_audio_bytes(data: bytes, suffix: str) -> np.ndarray:
    """Return mono float32 samples at 16 kHz."""
    suffix = suffix if suffix.startswith(".") else f".{suffix}"
    suffix = suffix.lower() or ".bin"
    if suffix in {".wav", ".wave"}:
        samples, sr = read_wav_bytes(data)
        if sr == AUDIO_SR and samples.ndim == 1:
            return samples
        try:
            return _ffmpeg_to_16k(data, suffix)
        except (OSError, subprocess.CalledProcessError, ValueError):
            mono = samples.mean(axis=-1) if samples.ndim > 1 else samples
            return resample_linear(mono, sr, AUDIO_SR)
    return _ffmpeg_to_16k(data, suffix)


def read_wav_bytes(data: bytes) -> tuple[np.ndarray, int]:
    """Mono-or-stereo WAV → float32 samples in [-1, 1] and sample rate."""
    try:
        with wave.open(io.BytesIO(data), "rb") as handle:
            channels = int(handle.getnchannels())
            width = int(handle.getsampwidth())
            rate = int(handle.getframerate())
            frames = handle.readframes(handle.getnframes())
    except wave.Error as exc:
        raise ValueError(f"could not read wav: {exc}") from exc
    if width == 2:
        pcm = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 1:
        pcm = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 4:
        pcm = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported wav sample width {width}")
    if channels > 1:
        pcm = pcm.reshape(-1, channels)
    if pcm.size == 0:
        raise ValueError("audio file is empty")
    return np.ascontiguousarray(pcm), rate


def resample_linear(samples: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    mono = np.asarray(samples, dtype=np.float32).reshape(-1)
    if int(src_sr) == int(dst_sr) or mono.size == 0:
        return mono
    n = max(1, int(round(mono.size * float(dst_sr) / float(src_sr))))
    src_x = np.linspace(0.0, 1.0, mono.size, endpoint=False)
    dst_x = np.linspace(0.0, 1.0, n, endpoint=False)
    return np.interp(dst_x, src_x, mono).astype(np.float32, copy=False)


def _ffmpeg_to_16k(data: bytes, suffix: str) -> np.ndarray:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise ValueError("ffmpeg is required to decode this audio upload")
    with tempfile.TemporaryDirectory(prefix="anemll-audio-") as tmp:
        src = Path(tmp) / f"in{suffix}"
        dst = Path(tmp) / "out.wav"
        src.write_bytes(data)
        proc = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(src),
                "-ac",
                "1",
                "-ar",
                str(AUDIO_SR),
                "-f",
                "wav",
                str(dst),
            ],
            check=False,
            capture_output=True,
        )
        if proc.returncode != 0 or not dst.is_file():
            err = proc.stderr.decode("utf-8", errors="replace").strip()
            raise ValueError(f"ffmpeg could not decode audio: {err or proc.returncode}")
        samples, sr = read_wav_bytes(dst.read_bytes())
    if sr != AUDIO_SR:
        raise ValueError(f"ffmpeg returned {sr} Hz, expected {AUDIO_SR}")
    if samples.ndim > 1:
        samples = samples.mean(axis=-1)
    return np.ascontiguousarray(samples, dtype=np.float32)
