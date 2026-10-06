#!/usr/bin/env python3
"""Export vision / audio / text towers to Core AI ``.aimodel`` packages.

Does **not** call ``forge.py convert``. Host processor + interleave stay
outside the graph (PLAN Phase 2 separate packages).

Phase A (embeddings venv): load the matching ST slice, wrap, torch.export,
save ``.pt2``. Phase B (``ANEMLL_COREAI_PYTHON``): convert ``.pt2`` → ``.aimodel``.

FLOAT32 text Core ML artifacts are not touched.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.coreai_towers import (  # noqa: E402
    AUDIO_FEAT,
    AUDIO_FRAMES,
    AudioSoftTokens,
    VISION_SOFT_TOKENS,
    VisionSoftTokens,
    audio_example,
    tower_io_spec,
    vision_example,
)
from src.embed_wrapper import EmbeddingGemma2Wrapper  # noqa: E402
from src.export_utils import (  # noqa: E402
    artifacts_root,
    example_trace_inputs,
    force_eager_attention,
    git_sha,
    host_versions,
    utc_now,
    write_json,
)
from src.load_multimodal_model import (  # noqa: E402
    get_auto_model,
    load_multimodal_sentence_transformer,
)
from src.load_text_model import load_sentence_transformer  # noqa: E402
from src.trace_patches import apply_fixed_shape_patches  # noqa: E402
from src.traceable_wrapper import TraceableEmbeddingGemma2  # noqa: E402

DEFAULT_COREAI_PY = Path("/Users/anemll/anemll-forge/coreai/.venv/bin/python")
CONVERT_SCRIPT = REPO_ROOT / "scripts" / "_coreai_convert_ep.py"


def _coreai_python() -> Path:
    raw = os.environ.get("ANEMLL_COREAI_PYTHON")
    return Path(raw) if raw else DEFAULT_COREAI_PY


def _export_program(module: torch.nn.Module, args: tuple, path: Path) -> None:
    module.eval()
    with torch.no_grad():
        ep = torch.export.export(module, args, strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(ep, str(path))
    print(f"wrote {path}")


def _convert(
    ep: Path,
    out: Path,
    *,
    entry: str,
    inputs: list[str],
    outputs: list[str],
    no_cast16: bool = False,
) -> int:
    py = _coreai_python()
    if not py.is_file():
        print(f"ERROR: Core AI python missing at {py}")
        return 2
    cmd = [
        str(py),
        str(CONVERT_SCRIPT),
        "--ep",
        str(ep),
        "--out",
        str(out),
        "--entry",
        entry,
        "--input-names",
        ",".join(inputs),
        "--output-names",
        ",".join(outputs),
    ]
    if no_cast16:
        cmd.append("--no-cast16")
    print("convert:", " ".join(cmd))
    rc = int(subprocess.call(cmd))
    if rc != 0 and not no_cast16:
        print("convert failed with cast16; retrying --no-cast16")
        return _convert(
            ep, out, entry=entry, inputs=inputs, outputs=outputs, no_cast16=True
        )
    return rc


def _export_vision(out_dir: Path, *, convert: bool) -> dict:
    print("Loading vision+text ST (audio off) on CPU FP32 …")
    st, load_meta = load_multimodal_sentence_transformer(
        dtype=torch.float32, device="cpu", vision=True, audio=False
    )
    auto = get_auto_model(st)
    if auto.vision_tower is None or auto.embed_vision is None:
        raise RuntimeError("vision tower missing")
    wrapped = VisionSoftTokens(auto.vision_tower, auto.embed_vision).eval()
    pixels, pos = vision_example()
    with torch.no_grad():
        soft = wrapped(pixels, pos)
    print(f"vision eager soft_tokens={tuple(soft.shape)}")
    if tuple(soft.shape) != (1, VISION_SOFT_TOKENS, 512):
        raise RuntimeError(f"unexpected vision out {tuple(soft.shape)}")
    ep_path = out_dir / "vision_s280.pt2"
    _export_program(wrapped, (pixels, pos), ep_path)
    aimodel = out_dir / "vision_s280.aimodel"
    rc = 0
    if convert:
        rc = _convert(
            ep_path,
            aimodel,
            entry="vision_s280",
            inputs=["pixel_values", "pixel_position_ids"],
            outputs=["soft_tokens"],
        )
    return {
        "tower": "vision",
        "load": load_meta,
        "eager_shape": list(soft.shape),
        "ep": str(ep_path),
        "aimodel": str(aimodel) if convert else None,
        "convert_rc": rc,
        "io": tower_io_spec("vision"),
        "finite": bool(torch.isfinite(soft).all()),
    }


def _export_audio(out_dir: Path, *, convert: bool) -> dict:
    print("Loading audio+text ST (vision off) on CPU FP32 …")
    st, load_meta = load_multimodal_sentence_transformer(
        dtype=torch.float32, device="cpu", vision=False, audio=True
    )
    auto = get_auto_model(st)
    if auto.audio_tower is None or auto.embed_audio is None:
        raise RuntimeError("audio tower missing")
    wrapped = AudioSoftTokens(auto.audio_tower, auto.embed_audio).eval()
    feat, mask = audio_example()
    with torch.no_grad():
        soft = wrapped(feat, mask)
    print(f"audio eager soft_tokens={tuple(soft.shape)} (subsampled from {AUDIO_FRAMES}x{AUDIO_FEAT})")
    ep_path = out_dir / "audio_s280.pt2"
    _export_program(wrapped, (feat, mask), ep_path)
    aimodel = out_dir / "audio_s280.aimodel"
    rc = 0
    if convert:
        rc = _convert(
            ep_path,
            aimodel,
            entry="audio_s280",
            inputs=["input_features", "input_features_mask"],
            outputs=["soft_tokens"],
        )
    return {
        "tower": "audio",
        "load": load_meta,
        "eager_shape": list(soft.shape),
        "ep": str(ep_path),
        "aimodel": str(aimodel) if convert else None,
        "convert_rc": rc,
        "io": tower_io_spec("audio"),
        "finite": bool(torch.isfinite(soft).all()),
    }


def _export_text(out_dir: Path, seq_len: int, *, convert: bool) -> dict:
    print(f"Loading text-only ST on CPU FP32; Core AI text S={seq_len} …")
    apply_fixed_shape_patches(seq_len, batch=1)
    st, load_meta = load_sentence_transformer(dtype=torch.float32, device="cpu")
    wrapper = EmbeddingGemma2Wrapper.from_sentence_transformer(st, normalize=True)
    force_eager_attention(wrapper)
    wrapper = wrapper.float().eval()
    traced = TraceableEmbeddingGemma2(wrapper, seq_len=seq_len, batch=1).eval()
    ids, mask = example_trace_inputs(seq_len)
    ids = ids.to(dtype=torch.int32)
    mask = mask.to(dtype=torch.int32)
    with torch.no_grad():
        emb = traced(ids, mask)
    print(f"text eager embedding={tuple(emb.shape)}")
    ep_path = out_dir / f"text_s{seq_len}.pt2"
    _export_program(traced, (ids, mask), ep_path)
    aimodel = out_dir / f"text_s{seq_len}.aimodel"
    rc = 0
    if convert:
        rc = _convert(
            ep_path,
            aimodel,
            entry=f"text_s{seq_len}",
            inputs=["input_ids", "attention_mask"],
            outputs=["embedding"],
        )
    return {
        "tower": "text",
        "load": load_meta,
        "eager_shape": list(emb.shape),
        "seq_len": seq_len,
        "ep": str(ep_path),
        "aimodel": str(aimodel) if convert else None,
        "convert_rc": rc,
        "io": tower_io_spec("text"),
        "finite": bool(torch.isfinite(emb).all()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tower",
        action="append",
        choices=("vision", "audio", "text"),
        help="Repeatable. Default: vision, audio, text.",
    )
    parser.add_argument("--seq-len", type=int, default=128, help="Text package S (default 128).")
    parser.add_argument("--artifacts", type=Path, default=artifacts_root())
    parser.add_argument("--skip-convert", action="store_true")
    args = parser.parse_args()

    towers = args.tower or ["vision", "audio", "text"]
    out_dir = Path(args.artifacts) / "coreai"
    out_dir.mkdir(parents=True, exist_ok=True)
    convert = not args.skip_convert
    print(f"out_dir={out_dir} convert={convert} towers={towers}")
    print(f"ANEMLL_COREAI_PYTHON={_coreai_python()}")

    reports = []
    fail = False
    for name in towers:
        try:
            if name == "vision":
                reports.append(_export_vision(out_dir, convert=convert))
            elif name == "audio":
                reports.append(_export_audio(out_dir, convert=convert))
            else:
                reports.append(_export_text(out_dir, int(args.seq_len), convert=convert))
        except Exception as exc:
            fail = True
            reports.append({"tower": name, "error": f"{type(exc).__name__}: {exc}"})
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")

    meta = {
        "created_at_utc": utc_now(),
        "git_sha": git_sha(REPO_ROOT),
        "target": "coreai",
        "host": host_versions(),
        "towers": reports,
        "notes": [
            "Separate packages; host interleaves media placeholders.",
            "Not forge.py convert. FLOAT32 Core ML text tree untouched.",
            "PyTorch export is FP32; Core AI convert casts to 16-bit.",
        ],
    }
    write_json(out_dir / "towers.export.json", meta)
    print(f"wrote {out_dir / 'towers.export.json'}")
    if fail or any(int(r.get("convert_rc") or 0) != 0 for r in reports if "error" not in r):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
