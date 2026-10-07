#!/usr/bin/env python3
"""Export vision / audio / text towers to Core AI ``.aimodel`` packages.

Does **not** call ``forge.py convert``. Host processor + interleave stay
outside the graph (PLAN Phase 2 separate packages).

Phase A (embeddings venv): load the matching ST slice, wrap, torch.export,
save ``.pt2``. Phase B (``ANEMLL_COREAI_PYTHON``): convert ``.pt2`` → ``.aimodel``.
Audio ``.pt2`` save can hit TreeSpec/``flat_apply``; then convert the live
ExportedProgram with forge ``coreai`` site-packages on ``sys.path``.

FLOAT32 text Core ML artifacts are not touched.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.audio_export_patches import apply_audio_unfold_patch  # noqa: E402
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
from src.traceable_wrapper import (  # noqa: E402
    TraceableEmbeddingGemma2,
    TraceableEmbeddingGemma2Embeds,
)

DEFAULT_COREAI_PY = Path("/Users/anemll/anemll-forge/coreai/.venv/bin/python")
CONVERT_SCRIPT = REPO_ROOT / "scripts" / "_coreai_convert_ep.py"


def _coreai_python() -> Path:
    raw = os.environ.get("ANEMLL_COREAI_PYTHON")
    return Path(raw) if raw else DEFAULT_COREAI_PY


def _coreai_site_packages() -> Path | None:
    # ``.venv/bin/python`` often symlinks to Homebrew; do not resolve it
    # or site-packages becomes the framework install, not the venv.
    venv = _coreai_python().parent.parent
    sites = sorted(venv.glob("lib/python*/site-packages"))
    return sites[-1] if sites else None


def _export_program(module: torch.nn.Module, args: tuple, path: Path):
    """``torch.export`` and try to save ``.pt2``. Audio save can hit TreeSpec."""
    module.eval()
    with torch.no_grad():
        ep = torch.export.export(module, args, strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.export.save(ep, str(path))
        print(f"wrote {path}")
        return ep, None
    except Exception as exc:
        print(f"torch.export.save skipped ({type(exc).__name__}: {exc})")
        return ep, f"{type(exc).__name__}: {exc}"


def _unsupported_ops(exc: BaseException) -> list[str]:
    text = f"{exc}"
    found = re.findall(r"aten\.[A-Za-z0-9_]+\.[A-Za-z0-9_]+", text)
    return list(dict.fromkeys(found))


def _convert_inprocess(
    ep,
    out: Path,
    *,
    entry: str,
    inputs: list[str],
    outputs: list[str],
    no_cast16: bool = False,
) -> int:
    """Convert a live ExportedProgram. Used when ``.pt2`` save hits TreeSpec."""
    site = _coreai_site_packages()
    if site is None or not site.is_dir():
        print(f"ERROR: Core AI site-packages missing under {_coreai_python()}")
        return 2
    if str(site) not in sys.path:
        sys.path.insert(0, str(site))
    try:
        import coreai_torch
        from coreai_opt.casting import cast_to_16_bit_precision
    except ImportError as exc:
        print(f"ERROR: in-process coreai_torch import failed: {exc}")
        return 2
    print(f"in-process convert no_cast16={no_cast16} site={site}")
    ep = ep.run_decompositions(coreai_torch.get_decomp_table())
    if not no_cast16:
        cast_to_16_bit_precision(ep)
    conv = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    conv.add_exported_program(
        ep,
        input_names=inputs,
        output_names=outputs,
        entrypoint_name=entry,
    )
    prog = conv.to_coreai()
    prog.optimize()
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(out, ignore_errors=True)
    prog.save_asset(out)
    print(f"wrote {out}")
    return 0


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
        # cast16 zeros this graph (eager/.pt2 absmax ~5.1 → aimodel 0).
        rc = _convert(
            ep_path,
            aimodel,
            entry="vision_s280",
            inputs=["pixel_values", "pixel_position_ids"],
            outputs=["soft_tokens"],
            no_cast16=True,
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
    unfold_patch = apply_audio_unfold_patch()
    print(f"audio unfold patch={unfold_patch}")
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
    ep, save_err = _export_program(wrapped, (feat, mask), ep_path)
    if save_err and ep_path.is_file():
        ep_path.unlink()
        print(f"removed stale {ep_path}")
    aimodel = out_dir / "audio_s280.aimodel"
    rc = 0
    convert_error = None
    unsupported: list[str] = []
    if convert:
        try:
            if save_err is None:
                rc = _convert(
                    ep_path,
                    aimodel,
                    entry="audio_s280",
                    inputs=["input_features", "input_features_mask"],
                    outputs=["soft_tokens"],
                    no_cast16=True,
                )
            else:
                print("audio .pt2 save failed; converting live ExportedProgram")
                try:
                    rc = _convert_inprocess(
                        ep,
                        aimodel,
                        entry="audio_s280",
                        inputs=["input_features", "input_features_mask"],
                        outputs=["soft_tokens"],
                        no_cast16=True,
                    )
                except Exception as exc:
                    print(f"in-process convert failed with cast16: {type(exc).__name__}: {exc}")
                    rc = _convert_inprocess(
                        ep,
                        aimodel,
                        entry="audio_s280",
                        inputs=["input_features", "input_features_mask"],
                        outputs=["soft_tokens"],
                        no_cast16=True,
                    )
        except Exception as exc:
            convert_error = f"{type(exc).__name__}: {exc}"
            unsupported = _unsupported_ops(exc)
            print(f"FAIL audio convert: {convert_error}")
            if unsupported:
                print(f"unsupported ops: {unsupported}")
            rc = 1
    return {
        "tower": "audio",
        "load": load_meta,
        "eager_shape": list(soft.shape),
        "ep": str(ep_path),
        "aimodel": str(aimodel) if convert else None,
        "convert_rc": rc,
        "io": tower_io_spec("audio"),
        "finite": bool(torch.isfinite(soft).all()),
        "unfold_patch": unfold_patch,
        "save_error": save_err,
        "convert_error": convert_error,
        "unsupported_ops": unsupported,
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
    # Vocab is 262144 — si16 wraps SearchQuery tokens (236787 → -25357).
    # int32 I/O; wrapper still widens ids to long for F.embedding.
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


def _export_text_embeds(out_dir: Path, seq_len: int, *, convert: bool) -> dict:
    """embeds [1, S, 512] + mask → 768. S=320 fits caption+256+25."""
    print(f"Loading text-only ST on CPU FP32; Core AI text-embeds S={seq_len} …")
    apply_fixed_shape_patches(seq_len, batch=1)
    st, load_meta = load_sentence_transformer(dtype=torch.float32, device="cpu")
    wrapper = EmbeddingGemma2Wrapper.from_sentence_transformer(st, normalize=True)
    force_eager_attention(wrapper)
    wrapper = wrapper.float().eval()
    traced = TraceableEmbeddingGemma2Embeds(wrapper, seq_len=seq_len, batch=1).eval()
    hidden = int(wrapper.hidden_size)
    # ANE-legal I/O: f16 embeds + f16 mask. Graph widens embeds to f32.
    # Mask stays f16 at I/O (anec.not_equal_zero rejects si16; f32 widen
    # dropped the ANE region).
    embeds = torch.randn(1, seq_len, hidden, dtype=torch.float16)
    mask = torch.ones(1, seq_len, dtype=torch.float16)
    mask[:, -8:] = 0
    with torch.no_grad():
        emb = traced(embeds, mask)
    print(f"text-embeds eager embedding={tuple(emb.shape)} finite={bool(torch.isfinite(emb).all())}")
    ep_path = out_dir / f"text_embeds_s{seq_len}.pt2"
    _export_program(traced, (embeds, mask), ep_path)
    aimodel = out_dir / f"text_embeds_s{seq_len}.aimodel"
    rc = 0
    if convert:
        # Float I/O — skip cast16 (vision lesson: that pass can zero a graph).
        rc = _convert(
            ep_path,
            aimodel,
            entry=f"text_embeds_s{seq_len}",
            inputs=["inputs_embeds", "attention_mask"],
            outputs=["embedding"],
            no_cast16=True,
        )
    return {
        "tower": "text_embeds",
        "load": load_meta,
        "eager_shape": list(emb.shape),
        "seq_len": seq_len,
        "ep": str(ep_path),
        "aimodel": str(aimodel) if convert else None,
        "convert_rc": rc,
        "io": tower_io_spec("text_embeds"),
        "finite": bool(torch.isfinite(emb).all()),
        "cast16": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tower",
        action="append",
        choices=("vision", "audio", "text", "text_embeds"),
        help="Repeatable. Default: vision, audio, text.",
    )
    parser.add_argument("--seq-len", type=int, default=128, help="Ids text package S (default 128).")
    parser.add_argument(
        "--embeds-seq-len",
        type=int,
        default=320,
        help="inputs_embeds text package S (default 320; fits mix 288).",
    )
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
            elif name == "text_embeds":
                reports.append(
                    _export_text_embeds(out_dir, int(args.embeds_seq_len), convert=convert)
                )
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
            "PyTorch export is FP32; convert --no-cast16 (cast16 zeros vision).",
            "Text ids I/O is int32 (vocab 262144 overflows si16).",
            "ANE-legal float I/O: vision pixels f16, audio feat+mask f16, text_embeds f16 in/out.",
        ],
    }
    write_json(out_dir / "towers.export.json", meta)
    print(f"wrote {out_dir / 'towers.export.json'}")
    if fail or any(int(r.get("convert_rc") or 0) != 0 for r in reports if "error" not in r):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
