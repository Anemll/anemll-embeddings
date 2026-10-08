"""Deterministic synthetic image / audio / video-frame writers (no ffmpeg)."""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

from PIL import Image

IMAGE_SIZE = 64
AUDIO_SR = 16000
AUDIO_SECONDS = 1.0
AUDIO_FREQ = 440.0


def write_png(path: Path, rgb: tuple[int, int, int], size: int = IMAGE_SIZE) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (int(size), int(size)), rgb).save(path)
    return path


def write_tone_wav(
    path: Path,
    *,
    seconds: float = AUDIO_SECONDS,
    sr: int = AUDIO_SR,
    freq: float = AUDIO_FREQ,
    amp: float = 0.2,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(seconds * sr)
    with wave.open(str(path), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        frames = bytearray()
        for i in range(n):
            sample = amp * math.sin(2.0 * math.pi * freq * i / sr)
            frames += struct.pack("<h", int(max(-1.0, min(1.0, sample)) * 32767.0))
        w.writeframes(bytes(frames))
    return path


def write_video_frames(
    directory: Path,
    *,
    count: int = 3,
    size: int = IMAGE_SIZE,
    stem: str = "video_f",
) -> list[Path]:
    """Three solid frames (green → teal → blue). Video encode uses this list."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    colors = [(32, 160, 80), (16, 120, 140), (20, 60, 180)]
    paths: list[Path] = []
    for i in range(int(count)):
        paths.append(write_png(directory / f"{stem}{i}.png", colors[i % len(colors)], size=size))
    return paths


def write_default_media(media_dir: Path) -> dict[str, Path | list[Path]]:
    """Canonical synthetic set used by multimodal fixtures."""
    media_dir = Path(media_dir)
    media_dir.mkdir(parents=True, exist_ok=True)
    frames = write_video_frames(media_dir)
    return {
        "aurora.png": write_png(media_dir / "aurora.png", (32, 160, 80)),
        "tone_a4.wav": write_tone_wav(media_dir / "tone_a4.wav"),
        "video_frames": frames,
    }
