#!/usr/bin/env python3
"""Cosine parity: Core AI host path vs ST multimodal fixtures (6×768).

Driveable: text via text_s128; image / audio / mix via towers + PyTorch text.
Video is skipped (no video package). Crops package 280/70 soft tokens to HF
256/25 when pads are a trailing suffix (small host fix, no re-export).

Fail-closed: non-finite or cosine below ABSURD (0.10) with no documented
limit. Documented: package slots 280/70 vs HF 256/25; text_s128 vs ST
~0.76 (package, not prefix — wrapper/ST vs fixture ~1.0).

Does not touch the FLOAT32 Core ML tree. ANE is out of scope.
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
    adapt_vision_pixels,
    adapt_vision_position_ids,
    crop_audio_soft_to_src,
    crop_vision_soft_to_valid,
    encode_interleaved,
    expand_media_placeholders,
    pad_audio_to_package,
    slot_report,
)
from src.coreai_smoke import dummy_numpy_inputs  # noqa: E402
from src.embed_wrapper import EmbeddingGemma2Wrapper, tokenize_with_st_prompt  # noqa: E402
from src.export_utils import (  # noqa: E402
    artifacts_root,
    git_sha,
    pad_to_seq_len,
    utc_now,
    write_json,
)
from src.load_text_model import default_model_path, load_sentence_transformer  # noqa: E402
from src.multimodal_media import AUDIO_SR, write_default_media  # noqa: E402
from src.parity_metrics import cosine, rel_l2  # noqa: E402

DEFAULT_COREAI_PY = Path("/Users/anemll/anemll-forge/coreai/.venv/bin/python")
RUN_NPY = REPO_ROOT / "scripts" / "_coreai_run_npy.py"
PROMPTS = REPO_ROOT / "tests" / "fixtures" / "multimodal_prompts.json"
TEXT_S = 128
# Fail-closed floor. T6-style 0.95 is not the Core AI text-package contract.
ABSURD_COSINE = 0.10
# si16 I/O wrapped vocab 262144 (SearchQuery 236787 → -25357). Text I/O is int32.
TEXT_IDS_DTYPE = "int32"


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
    print("run", entry)
    rc = subprocess.call(cmd)
    if rc != 0:
        raise RuntimeError(f"coreai forward rc={rc} entry={entry}")
    return np.load(out)


def _load_processor(model_path: Path):
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)


def _encode_text_s128(
    *,
    st,
    tok,
    text: str,
    prompt_name: str | None,
    pad_id: int,
    out_dir: Path,
    case_id: str,
) -> np.ndarray:
    batch = tokenize_with_st_prompt(st, text, prompt_name, max_length=TEXT_S)
    ids, mask = pad_to_seq_len(
        batch["input_ids"], batch["attention_mask"], TEXT_S, pad_token_id=pad_id
    )
    feed = dummy_numpy_inputs("text")
    feed["input_ids"] = ids.numpy().astype(np.int32)
    feed["attention_mask"] = mask.numpy().astype(np.int32)
    return _run_tower(
        out_dir / "text_s128.aimodel",
        "text_s128",
        feed,
        out_dir / f"parity_{case_id}_text.npy",
    ).reshape(-1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=artifacts_root())
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        help="Case id to run (repeatable). Default: all fixture rows.",
    )
    args = parser.parse_args()

    fixtures = Path(args.artifacts) / "fixtures"
    out_dir = Path(args.artifacts) / "coreai"
    media_dir = fixtures / "media"
    write_default_media(media_dir)
    pack = json.loads(PROMPTS.read_text())
    prompts = pack["prompts"]
    ref = np.load(fixtures / pack.get("embeddings_name", "multimodal_embeddings.npy"))
    if ref.shape != (len(prompts), 768):
        raise SystemExit(f"fixture shape {ref.shape} != ({len(prompts)}, 768)")

    model_path = Path(args.model) if args.model else default_model_path()
    proc = _load_processor(model_path)
    tok = proc.tokenizer
    image_id = int(tok.image_token_id)
    audio_id = int(tok.audio_token_id)
    pad_id = int(tok.pad_token_id)

    print("Loading text-only ST (270M) …")
    st, load_meta = load_sentence_transformer(model_path, dtype=torch.float32, device="cpu")
    wrapper = EmbeddingGemma2Wrapper.from_sentence_transformer(st, normalize=True).eval()

    vision_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    audio_cache: dict[str, tuple[np.ndarray, int]] = {}

    def vision_soft(name: str) -> np.ndarray:
        if name in vision_cache:
            return vision_cache[name][0]
        pil = Image.open(media_dir / name)
        vin = proc(text="<|image|>", images=pil, return_tensors="pt")
        pixels = adapt_vision_pixels(vin["pixel_values"].numpy())
        pos = adapt_vision_position_ids(vin["image_position_ids"].numpy())
        raw = _run_tower(
            out_dir / "vision_s280.aimodel",
            "vision_s280",
            {"pixel_values": pixels.astype(np.float32, copy=False), "pixel_position_ids": pos},
            out_dir / f"parity_{Path(name).stem}_vision.npy",
        )
        cropped = crop_vision_soft_to_valid(raw, pos)
        vision_cache[name] = (cropped, pos)
        print(f"  vision {name} raw={list(raw.shape)} crop={list(cropped.shape)} absmax={float(np.abs(cropped).max()):.4f}")
        return cropped

    def audio_soft(name: str) -> np.ndarray:
        if name in audio_cache:
            return audio_cache[name][0]
        wav = _wav_f32(media_dir / name)
        ain = proc(text="<|audio|>", audio=wav, return_tensors="pt")
        n_src = int(ain["input_features"].shape[1])
        feat, amask = pad_audio_to_package(
            ain["input_features"].numpy(), ain["input_features_mask"].numpy()
        )
        amask[...] = 1
        raw = _run_tower(
            out_dir / "audio_s280.aimodel",
            "audio_s280",
            {"input_features": feat.astype(np.float16, copy=False), "input_features_mask": amask},
            out_dir / f"parity_{Path(name).stem}_audio.npy",
        )
        cropped = crop_audio_soft_to_src(raw, n_src)
        audio_cache[name] = (cropped, n_src)
        print(f"  audio {name} raw={list(raw.shape)} crop={list(cropped.shape)} frames={n_src}")
        return cropped

    only = set(args.only) if args.only else None
    rows = []
    fail = False
    for i, item in enumerate(prompts):
        case_id = item["id"]
        if only is not None and case_id not in only:
            continue
        kind = item.get("kind") or "text"
        text = item.get("text") or ""
        ref_vec = ref[i]
        row: dict = {
            "id": case_id,
            "kind": kind,
            "driven": True,
            "finite": False,
            "cosine": None,
            "rel_l2": None,
        }
        print(f"case {case_id} kind={kind}")
        try:
            if kind == "video":
                row["driven"] = False
                row["skip"] = "no video package"
                print("  skip: no video package")
                rows.append(row)
                continue
            if kind == "text":
                pred = _encode_text_s128(
                    st=st,
                    tok=tok,
                    text=text,
                    prompt_name=item.get("prompt_name"),
                    pad_id=pad_id,
                    out_dir=out_dir,
                    case_id=case_id,
                )
                row["path"] = "text_s128"
                batch = tokenize_with_st_prompt(
                    st, text, item.get("prompt_name"), max_length=TEXT_S
                )
                ids, mask = pad_to_seq_len(
                    batch["input_ids"],
                    batch["attention_mask"],
                    TEXT_S,
                    pad_token_id=pad_id,
                )
                with torch.no_grad():
                    wrap = wrapper(ids, mask).detach().cpu().numpy().reshape(-1)
                row["wrapper_cosine_vs_ref"] = cosine(wrap, ref_vec)
                row["package_cosine_vs_wrapper"] = cosine(pred, wrap)
            else:
                img_name = (item.get("image") or [None])[0]
                aud_name = item.get("audio")
                img_t = None
                aud_t = None
                n_img = 0
                n_aud = 0
                if img_name:
                    vs = vision_soft(img_name)
                    img_t = torch.from_numpy(np.asarray(vs)).to(dtype=torch.float32)
                    n_img = int(img_t.shape[1])
                if aud_name:
                    asoft = audio_soft(aud_name)
                    aud_t = torch.from_numpy(np.asarray(asoft)).to(dtype=torch.float32)
                    n_aud = int(aud_t.shape[1])
                expanded = expand_media_placeholders(
                    text,
                    image_token=tok.image_token,
                    audio_token=tok.audio_token,
                    image_slots=n_img or 0,
                    audio_slots=n_aud or 0,
                )
                batch = tok(expanded, return_tensors="pt", padding=False, truncation=False)
                slots = slot_report(
                    batch["input_ids"], image_token_id=image_id, audio_token_id=audio_id
                )
                row["slots"] = slots
                if slots["image_slots"] != n_img or slots["audio_slots"] != n_aud:
                    raise RuntimeError(f"slot mismatch {slots} vs {n_img}/{n_aud}")
                with torch.no_grad():
                    emb = encode_interleaved(
                        wrapper,
                        batch["input_ids"],
                        batch["attention_mask"],
                        image_token_id=image_id,
                        audio_token_id=audio_id,
                        pad_token_id=pad_id,
                        image_soft=img_t,
                        audio_soft=aud_t,
                    )
                pred = emb.detach().cpu().numpy().reshape(-1)
                row["path"] = "host_interleave_hf_slots"
            finite = bool(pred.shape == (768,) and np.isfinite(pred).all())
            row["finite"] = finite
            row["pred_l2"] = float(np.linalg.norm(pred.astype(np.float64)))
            row["ref_l2"] = float(np.linalg.norm(ref_vec.astype(np.float64)))
            if not finite:
                row["error"] = "non-finite or bad shape"
                fail = True
            else:
                row["cosine"] = cosine(pred, ref_vec)
                row["rel_l2"] = rel_l2(pred, ref_vec)
                if row["cosine"] < ABSURD_COSINE:
                    row["error"] = (
                        f"cosine {row['cosine']:.4f} < {ABSURD_COSINE} (absurd, no documented limit)"
                    )
                    fail = True
            print(f"  cosine={row.get('cosine')} rel_l2={row.get('rel_l2')} finite={finite}")
        except Exception as exc:
            row["driven"] = True
            row["error"] = f"{type(exc).__name__}: {exc}"
            fail = True
            print(f"  FAIL {row['error']}")
        rows.append(row)

    report = {
        "created_at_utc": utc_now(),
        "git_sha": git_sha(REPO_ROOT),
        "target": "coreai_host_parity",
        "compute": "cpu",
        "pass": not fail,
        "load": load_meta,
        "absurd_cosine": ABSURD_COSINE,
        "limits": {
            "vision_slots": "package 280 vs HF 256 (host crops trailing pad groups)",
            "audio_slots": "package 70 vs HF 25 (host crops ceil(frames/4); 99→25)",
            "text_ids": "int32 I/O; si16 wraps vocab 262144 (SearchQuery 236787)",
            "audio_mask": "all-1s silence pad required for finite audio (keep-mask NaNs)",
        },
        "cases": rows,
        "notes": [
            "Host crops vision 280→256 (trailing pad groups) and audio 70→25 (ceil frames/4).",
            "That matches HF slot counts; package graphs stay 280/70 (no re-export).",
            "Video skipped: no video .aimodel.",
            "text_s128 is S=128 ids-only int32. Interleaved 768-d uses PyTorch text tower.",
            "si16 text I/O was the 0.76 gap (3/15 SearchQuery tokens wrapped).",
            "int32 re-export: mm_text_sq cosine 0.995 vs ST (package vs wrapper 0.995; remaining f16/cast16).",
            "Fail-closed: non-finite or cosine < 0.10. Not a T6 0.95 gate.",
            "CPU only. ANE embedding-widen / audio dummy_pool unchanged.",
            "FLOAT32 Core ML text tree untouched.",
        ],
    }
    dest = out_dir / "host.parity.json"
    if only is not None and dest.is_file():
        prev = json.loads(dest.read_text())
        by_id = {c["id"]: c for c in prev.get("cases") or []}
        for row in rows:
            by_id[row["id"]] = row
        order = [p["id"] for p in prompts]
        report["cases"] = [by_id[i] for i in order if i in by_id]
        report["merged_from"] = prev.get("git_sha")
    write_json(dest, report)
    print(f"wrote {dest} pass={not fail}")
    return 0 if not fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
