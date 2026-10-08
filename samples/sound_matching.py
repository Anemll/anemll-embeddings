#!/usr/bin/env python3
"""Match a short clip against a few sound labels.

    python samples/sound_matching.py --backend mock
    python samples/sound_matching.py --wav bark.wav --backend coreai
"""

from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from api import Embedder  # noqa: E402

AUDIO_SR = 16000
DEFAULT_LABELS = ("a dog barking", "a cat meowing", "piano music", "rain")


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        n = handle.getnframes()
        raw = handle.readframes(n)
        width = handle.getsampwidth()
    if width == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
    return samples.reshape(-1), rate


def _tone(seconds: float = 0.4, freq: float = 440.0, rate: int = AUDIO_SR) -> np.ndarray:
    t = np.arange(int(seconds * rate), dtype=np.float32) / rate
    return (0.2 * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav", type=Path, help="16-bit WAV clip")
    parser.add_argument("--backend", default="coreai", choices=("coreai", "mock", "reference"))
    parser.add_argument("--label", action="append", dest="labels")
    args = parser.parse_args(argv)

    if args.wav is None:
        wav, rate = _tone(), AUDIO_SR
    else:
        wav, rate = _read_wav(args.wav)
    labels = tuple(args.labels) if args.labels else DEFAULT_LABELS

    embedder = Embedder(backend=args.backend)
    query = embedder.embed_audio(wav, rate)
    ranked = sorted(
        ((_cosine(query, embedder.embed_text(label, role="document")), label) for label in labels),
        reverse=True,
    )
    embedder.close()
    print("query=audio")
    for score, text in ranked:
        print(f"  {score:.4f}  {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
