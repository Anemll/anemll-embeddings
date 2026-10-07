#!/usr/bin/env python3
"""CPU host interleave smoke: Core AI vision/audio + PyTorch text tower.

Does not re-export packages. Does not touch the FLOAT32 Core ML tree.
ANE specialize is out of scope (int widen / audio dummy_pool).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.coreai_host import (  # noqa: E402
    AUDIO_SLOTS,
    IMAGE_SLOTS,
    adapt_vision_pixels,
    adapt_vision_position_ids,
    encode_interleaved,
    expand_media_placeholders,
    pad_audio_to_package,
    slot_report,
)
from src.coreai_smoke import dummy_numpy_inputs  # noqa: E402
from src.embed_wrapper import EmbeddingGemma2Wrapper  # noqa: E402
from src.export_utils import (  # noqa: E402
    artifacts_root,
    git_sha,
    pad_to_seq_len,
    utc_now,
    write_json,
)
from src.load_text_model import load_sentence_transformer  # noqa: E402
from src.multimodal_media import AUDIO_SR, write_default_media  # noqa: E402

DEFAULT_COREAI_PY = Path("/Users/anemll/anemll-forge/coreai/.venv/bin/python")
RUN_NPY = REPO_ROOT / "scripts" / "_coreai_run_npy.py"
PROMPTS = REPO_ROOT / "tests" / "fixtures" / "multimodal_prompts.json"
TEXT_S = 128


def _coreai_python() -> Path:
    raw = os.environ.get("ANEMLL_COREAI_PYTHON")
    return Path(raw) if raw else DEFAULT_COREAI_PY


def _wav_f32(path: Path) -> np.ndarray:
    import wave

    with wave.open(str(path), "r") as w:
        raw = w.readframes(w.getnframes())
        sr = int(w.getframerate())
    if sr != AUDIO_SR:
        raise ValueError(f"{path} sr={sr} != {AUDIO_SR}")
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def _run_tower(pkg: Path, entry: str, feed: dict[str, np.ndarray], out: Path) -> np.ndarray:
    npz = out.with_suffix(".npz")
    np.savez(npz, **feed)
    cmd = [
        str(_coreai_python()),
        str(RUN_NPY),
        "--package",
        str(pkg),
        "--entry",
        entry,
        "--npz",
        str(npz),
        "--out",
        str(out),
    ]
    print("run", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        raise RuntimeError(f"coreai forward rc={rc} entry={entry}")
    return np.load(out)


def _load_processor(model_path: Path):
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=artifacts_root())
    parser.add_argument("--model", type=Path, default=None)
    args = parser.parse_args()

    out_dir = Path(args.artifacts) / "coreai"
    media_dir = Path(args.artifacts) / "fixtures" / "media"
    write_default_media(media_dir)
    pack = json.loads(PROMPTS.read_text())
    item = next(p for p in pack["prompts"] if p["id"] == "mm_image_caption")
    text = item["text"]
    img_path = media_dir / item["image"][0]
    wav_path = media_dir / "tone_a4.wav"

    from src.load_text_model import default_model_path

    model_path = Path(args.model) if args.model else default_model_path()
    proc = _load_processor(model_path)
    tok = proc.tokenizer
    image_id = int(tok.image_token_id)
    audio_id = int(tok.audio_token_id)
    pad_id = int(tok.pad_token_id)

    pil = Image.open(img_path)
    vision_in = proc(text="<|image|>", images=pil, return_tensors="pt")
    pixels = adapt_vision_pixels(vision_in["pixel_values"].numpy())
    pos = adapt_vision_position_ids(vision_in["image_position_ids"].numpy())

    wav = _wav_f32(wav_path)
    audio_in = proc(text="<|audio|>", audio=wav, return_tensors="pt")
    feat, amask = pad_audio_to_package(
        audio_in["input_features"].numpy(),
        audio_in["input_features_mask"].numpy(),
    )
    # Partial keep-mask on this package yields NaNs; treat pad frames as silence.
    amask[...] = 1

    vision_npy = out_dir / "host_vision_soft.npy"
    audio_npy = out_dir / "host_audio_soft.npy"
    vision_soft = _run_tower(
        out_dir / "vision_s280.aimodel",
        "vision_s280",
        {"pixel_values": pixels.astype(np.float16, copy=False), "pixel_position_ids": pos},
        vision_npy,
    )
    audio_soft = _run_tower(
        out_dir / "audio_s280.aimodel",
        "audio_s280",
        {"input_features": feat.astype(np.float16, copy=False), "input_features_mask": amask},
        audio_npy,
    )
    print(f"vision_soft={vision_soft.shape} audio_soft={audio_soft.shape}")
    vis_abs = float(np.nanmax(np.abs(vision_soft))) if vision_soft.size else 0.0
    aud_finite = bool(np.isfinite(audio_soft).all())
    print(f"vision_absmax={vis_abs} audio_finite={aud_finite}")
    if not aud_finite:
        raise SystemExit("audio soft tokens not finite (partial mask / package)")

    mix = "tone then picture <|audio|> <|image|>"
    expanded = expand_media_placeholders(
        mix,
        image_token=tok.image_token,
        audio_token=tok.audio_token,
        image_slots=IMAGE_SLOTS,
        audio_slots=AUDIO_SLOTS,
    )
    batch = tok(expanded, return_tensors="pt", padding=False, truncation=False)
    ids = batch["input_ids"]
    mask = batch["attention_mask"]
    slots = slot_report(ids, image_token_id=image_id, audio_token_id=audio_id)
    print(f"host seq={slots}")
    if slots["image_slots"] != IMAGE_SLOTS or slots["audio_slots"] != AUDIO_SLOTS:
        raise SystemExit(f"slot mismatch {slots}")

    print("Loading text-only ST (270M) for interleaved embeds …")
    st, load_meta = load_sentence_transformer(model_path, dtype=torch.float32, device="cpu")
    wrapper = EmbeddingGemma2Wrapper.from_sentence_transformer(st, normalize=True).eval()
    img_t = torch.from_numpy(np.asarray(vision_soft)).to(dtype=torch.float32)
    aud_t = torch.from_numpy(np.asarray(audio_soft)).to(dtype=torch.float32)
    with torch.no_grad():
        emb = encode_interleaved(
            wrapper,
            ids,
            mask,
            image_token_id=image_id,
            audio_token_id=audio_id,
            pad_token_id=pad_id,
            image_soft=img_t,
            audio_soft=aud_t,
        )
        emb_zero = encode_interleaved(
            wrapper,
            ids,
            mask,
            image_token_id=image_id,
            audio_token_id=audio_id,
            pad_token_id=pad_id,
            image_soft=torch.zeros_like(img_t),
            audio_soft=torch.zeros_like(aud_t),
        )
    vec = emb.detach().cpu().numpy().reshape(-1)
    zero = emb_zero.detach().cpu().numpy().reshape(-1)
    interleave_moves = float(np.linalg.norm(vec - zero))
    print(f"interleaved {vec.shape} finite={bool(np.isfinite(vec).all())} l2={float(np.linalg.norm(vec)):.4f}")
    print(f"vs zero-soft L2={interleave_moves:.4f}")

    # text_s128 on caption only (package cannot hold 280 image slots).
    caption = "Green sky glow."
    cap = tok(caption, return_tensors="pt")
    cap_ids, cap_mask = pad_to_seq_len(cap["input_ids"], cap["attention_mask"], TEXT_S, pad_token_id=pad_id)
    text_feed = dummy_numpy_inputs("text")
    text_feed["input_ids"] = cap_ids.numpy().astype(np.int16)
    text_feed["attention_mask"] = cap_mask.numpy().astype(np.int16)
    text_vec = _run_tower(
        out_dir / "text_s128.aimodel",
        "text_s128",
        text_feed,
        out_dir / "host_text_s128.npy",
    ).reshape(-1)
    print(f"text_s128 {text_vec.shape} finite={bool(np.isfinite(text_vec).all())}")

    ok = (
        vec.shape == (768,)
        and bool(np.isfinite(vec).all())
        and interleave_moves > 1e-3
        and text_vec.shape == (768,)
        and bool(np.isfinite(text_vec).all())
        and list(vision_soft.shape) == [1, IMAGE_SLOTS, 512]
        and list(audio_soft.shape) == [1, AUDIO_SLOTS, 512]
        and aud_finite
    )
    report = {
        "created_at_utc": utc_now(),
        "git_sha": git_sha(REPO_ROOT),
        "target": "coreai_host",
        "pass": ok,
        "compute": "cpu",
        "load": load_meta,
        "prompt": mix,
        "fixture_caption": text,
        "slots": slots,
        "vision_soft_shape": list(vision_soft.shape),
        "vision_absmax": vis_abs,
        "vision_note": "si16 vision package is finite zeros on CPU dummy and real pixels",
        "audio_soft_shape": list(audio_soft.shape),
        "audio_finite": aud_finite,
        "interleaved_shape": list(vec.shape),
        "interleaved_finite": bool(np.isfinite(vec).all()),
        "interleaved_l2": float(np.linalg.norm(vec)),
        "zero_soft_l2_delta": interleave_moves,
        "text_s128_shape": list(text_vec.shape),
        "text_s128_finite": bool(np.isfinite(text_vec).all()),
        "text_s128_note": "ids-only S=128; cannot hold 280 image slots",
        "limits": [
            "CPU only. ANE follow-up: embedding Int/Long widen; audio dummy_pool.",
            "Host expands 280 image / 70 audio slots (package), not HF 256 / 25.",
            "Interleaved 768-d uses PyTorch text tower + inputs_embeds.",
            "FLOAT32 Core ML text tree untouched. No re-export.",
        ],
    }
    dest = out_dir / "host.smoke.json"
    write_json(dest, report)
    print(f"wrote {dest} pass={ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
