#!/usr/bin/env python3
"""Core AI host path passes real masks. No .aimodel and no checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from demo.backends.coreai import CoreAIBackend  # noqa: E402


def _fail(msg: str) -> None:
    raise AssertionError(msg)


class _Tok:
    image_token = "<|image|>"
    audio_token = "<|audio|>"
    image_token_id = 7
    audio_token_id = 8
    pad_token_id = 0

    def __call__(self, text, return_tensors="pt", padding=False, truncation=False, max_length=None):
        ids = [1]
        i = 0
        specials = (("<|image|>", 7), ("<|audio|>", 8))
        while i < len(text):
            matched = False
            for token, token_id in specials:
                if text.startswith(token, i):
                    ids.append(token_id)
                    i += len(token)
                    matched = True
                    break
            if matched:
                continue
            if text[i].isspace():
                i += 1
                continue
            j = i
            while j < len(text) and not text[j].isspace() and not text.startswith("<|", j):
                j += 1
            word = text[i:j] or "x"
            ids.append(10 + (sum(word.encode()) % 40))
            i = j if j > i else i + 1
        ids.append(2)
        if max_length is not None:
            ids = ids[: int(max_length)]
        tensor = torch.tensor([ids], dtype=torch.long)
        return {"input_ids": tensor, "attention_mask": torch.ones_like(tensor)}


class _Proc:
    def __init__(self) -> None:
        self.tokenizer = _Tok()

    def __call__(self, text="", images=None, audio=None, return_tensors="pt"):
        if images is not None:
            pos = torch.zeros(1, 2520, 2)
            pos[:, 2304:, :] = -1
            pixels = torch.full((1, 2520, 768), 0.25)
            return {"pixel_values": pixels, "image_position_ids": pos}
        if audio is not None:
            n = max(8, int(round(len(np.asarray(audio)) / 16000 * 99)))
            feat = torch.zeros(1, n, 128)
            mask = torch.ones(1, n)
            mask[:, -3:] = 0
            return {"input_features": feat, "input_features_mask": mask}
        raise AssertionError("processor called without media")


class _Text(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb = torch.nn.Embedding(128, 512)

    def get_input_embeddings(self):
        return self.emb


class _Runner:
    def __init__(self) -> None:
        self.feeds: list[tuple[str, dict[str, np.ndarray]]] = []

    def startup(self) -> dict:
        towers = {}
        for name, ms in (("vision_s280", 350.0), ("audio_s280", 11.0), ("text_embeds_s320", 35.0)):
            towers[name] = {"loaded": True, "warmup_ms": ms, "placement": "fullyOnANE"}
        return {"placement": "fullyOnANE", "towers": towers}

    def forward(self, tower: str, feed: dict[str, np.ndarray]):
        self.feeds.append((tower, {key: np.array(value, copy=True) for key, value in feed.items()}))
        if tower == "vision":
            return np.zeros((1, 280, 512), dtype=np.float16), 350.0
        if tower == "audio":
            return np.zeros((1, 1, 70, 512), dtype=np.float16), 11.0
        vec = np.zeros((1, 768), dtype=np.float32)
        vec[0, 0] = 1.0
        return vec, 35.0

    def close(self) -> None:
        return None


def _feeds(runner: _Runner, tower: str) -> list[dict[str, np.ndarray]]:
    return [feed for name, feed in runner.feeds if name == tower]


def test_real_masks() -> None:
    runner = _Runner()
    backend = CoreAIBackend(
        artifacts=None,
        model_path=None,
        coreai_python=None,
        compute="ane",
        runner=runner,
        processor=_Proc(),
        text_model=_Text(),
        prompts={},
    )
    health = backend.warmup()
    if health["placement"] != "fullyOnANE":
        _fail(str(health))
    if backend.compute != "ane":
        _fail(backend.compute)

    text = backend.embed_text("northern lights", role="query")
    if text.placement != "fullyOnANE" or abs(float(np.linalg.norm(text.vector)) - 1) > 1e-5:
        _fail("text result")
    text_mask = _feeds(runner, "text")[-1]["attention_mask"]
    if text_mask.shape[-1] != 320:
        _fail(str(text_mask.shape))
    kept = int((text_mask.reshape(-1) != 0).sum())
    if kept >= 320 or kept < 1:
        _fail(f"text mask kept {kept}; pad must stay zero")

    image = backend.embed_image(Image.new("RGB", (8, 8), (12, 40, 90)))
    if image.placement != "fullyOnANE":
        _fail("image placement")
    pos = _feeds(runner, "vision")[-1]["pixel_position_ids"]
    if not np.any(np.asarray(pos) < 0):
        _fail("vision pad positions were dropped")
    image_mask = _feeds(runner, "text")[-1]["attention_mask"]
    if int((image_mask.reshape(-1) != 0).sum()) >= 320:
        _fail("image text mask is all ones")

    runner.feeds.clear()
    audio = backend.embed_audio(np.zeros(8000, dtype=np.float32), 16000)
    if audio.slices != 1:
        _fail(f"short clip slices {audio.slices}")
    amask = _feeds(runner, "audio")[0]["input_features_mask"]
    if amask.shape != (1, 1, 280, 1):
        _fail(str(amask.shape))
    if int((amask.reshape(-1) != 0).sum()) >= 280:
        _fail("short audio mask was all ones")

    runner.feeds.clear()
    long = backend.embed_audio(np.zeros(16000 * 4, dtype=np.float32), 16000)
    audio_feeds = _feeds(runner, "audio")
    if long.slices < 2 or len(audio_feeds) < 2:
        _fail(f"slices {long.slices} feeds {len(audio_feeds)}")
    for feed in audio_feeds:
        mask = feed["input_features_mask"]
        if int((mask.reshape(-1) != 0).sum()) >= int(mask.size):
            _fail("a long-audio slice was given an all-ones mask")
    backend.close()


def main() -> int:
    test_real_masks()
    print("OK test_real_masks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
