#!/usr/bin/env python3
"""Export short-text buckets and a packed text tower to Core AI.

Buckets are the existing ``text_embeds`` graph at a smaller fixed S
(``text_embeds_s32`` ... ``text_embeds_s256``). A packed tower
(``text_pack_256x16``) runs up to 16 texts in one 256-token forward, see
``model/text_pack.py``.

Phase A (embeddings venv) writes ``.pt2`` files and checks the packed math
in PyTorch FP32 against one-text-per-call. Phase B (Core AI venv) converts:

* ``--layout separate`` (default): one ``.aimodel`` per function.
* ``--layout multi``: one ``text_buckets.aimodel`` with every function.

    python model/export_text_buckets.py --artifacts ~/text-buckets \\
        --buckets 32,64,128,256 --pack 128x8,256x16 --layout multi
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from model.embed_wrapper import EmbeddingGemma2Wrapper  # noqa: E402
from model.export_coreai_towers import (  # noqa: E402
    _convert,
    _coreai_python,
    _export_program,
)
from model.export_utils import (  # noqa: E402
    force_eager_attention,
    git_sha,
    utc_now,
    write_json,
)
from model.load_text_model import load_sentence_transformer  # noqa: E402
from model.text_pack import (  # noqa: E402
    PACK_NEG,
    PackedTextEmbeds,
    example_pack_inputs,
)
from model.trace_patches import (  # noqa: E402
    apply_fixed_shape_patches,
    bind_used_weight_layout,
)
from model.traceable_wrapper import TraceableEmbeddingGemma2Embeds  # noqa: E402
from model.vision_export_patches import apply_fp16_safe_rms_norm_patch  # noqa: E402

MULTI_SCRIPT = REPO_ROOT / "model" / "_coreai_convert_multi.py"
BUCKET_IO = (["inputs_embeds", "attention_mask"], ["embedding"])
PACK_IO = (["inputs_embeds", "attention_bias", "positions", "pool"], ["embedding"])
CHECK_TEXTS = (
    "task: search result | query: What causes the northern lights?",
    "title: none | text: The northern lights are caused by charged particles from the sun.",
    "task: search result | query: grep -rn TODO src/",
    "title: none | text: def l2_normalize(x): return x / np.linalg.norm(x)",
    "task: sentence similarity | query: The cat sat on the mat.",
)


def bucket_name(seq_len: int) -> str:
    return f"text_embeds_s{int(seq_len)}"


def pack_name(n_tokens: int, max_texts: int) -> str:
    return f"text_pack_{int(n_tokens)}x{int(max_texts)}"


def _bucket_module(wrapper, seq_len: int):
    apply_fixed_shape_patches(seq_len, batch=1)
    module = TraceableEmbeddingGemma2Embeds(wrapper, seq_len=seq_len, batch=1).eval()
    bind_used_weight_layout(module)
    return module


def _pack_module(wrapper, n_tokens: int, max_texts: int):
    apply_fixed_shape_patches(n_tokens, batch=1)
    module = PackedTextEmbeds(wrapper, n_tokens=n_tokens, max_texts=max_texts).eval()
    bind_used_weight_layout(module)
    return module


def _token_rows(st, wrapper, text: str) -> torch.Tensor:
    ids = st.tokenizer(text, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        return wrapper.text_model.embed_tokens(ids)[0].to(torch.float32)  # [L, 512]


def check_pack_math(st, wrapper, n_tokens: int, max_texts: int) -> dict:
    """FP32 eager: packed rows vs each text alone at the same S (padded)."""
    rows = [_token_rows(st, wrapper, t) for t in CHECK_TEXTS[:max_texts]]
    embeds = torch.zeros(1, n_tokens, rows[0].shape[-1])
    bias = torch.full((1, 1, n_tokens, n_tokens), PACK_NEG)
    positions = torch.zeros(n_tokens, 1)
    pool = torch.zeros(max_texts, n_tokens)
    start = 0
    spans = []
    for slot, row in enumerate(rows):
        end = start + int(row.shape[0])
        embeds[0, start:end] = row
        bias[0, 0, start:end, start:end] = 0
        positions[start:end, 0] = torch.arange(end - start, dtype=torch.float32)
        pool[slot, start:end] = 1.0 / float(end - start)
        spans.append((start, end))
        start = end
    for idx in range(start, n_tokens):
        bias[0, 0, idx, idx] = 0
    pack = _pack_module(wrapper, n_tokens, max_texts)
    with torch.no_grad():
        packed = F.normalize(pack(embeds, bias, positions, pool).float(), dim=-1)
    single = _bucket_module(wrapper, n_tokens)
    cos = []
    for slot, row in enumerate(rows):
        e = torch.zeros(1, n_tokens, row.shape[-1])
        m = torch.zeros(1, n_tokens)
        e[0, : row.shape[0]] = row
        m[0, : row.shape[0]] = 1
        with torch.no_grad():
            ref = F.normalize(single(e, m).float(), dim=-1)[0]
        cos.append(float(torch.dot(ref, packed[slot])))
    return {"texts": len(rows), "tokens": start, "cosine": cos, "min_cosine": min(cos)}


def _convert_multi(programs: list[tuple[str, Path, tuple]], out: Path) -> int:
    py = _coreai_python()
    cmd = [str(py), str(MULTI_SCRIPT), "--out", str(out)]
    for entry, ep, (ins, outs) in programs:
        cmd += ["--program", f"{entry}={ep}:{','.join(ins)}:{','.join(outs)}"]
    print("convert:", " ".join(cmd))
    return int(subprocess.call(cmd))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--buckets", default="32,64,128,256")
    parser.add_argument(
        "--pack", default="256x16", help="comma list of NxT packed towers, empty for none"
    )
    parser.add_argument("--layout", choices=("separate", "multi", "both"), default="separate")
    parser.add_argument("--skip-convert", action="store_true")
    parser.add_argument("--skip-check", action="store_true")
    args = parser.parse_args()

    buckets = [int(x) for x in args.buckets.split(",") if x.strip()]
    packs = []
    for item in (x.strip() for x in args.pack.split(",")):
        if item:
            n, t = item.lower().split("x")
            packs.append((int(n), int(t)))
    out_dir = Path(args.artifacts) / "coreai"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"rms_norm={apply_fp16_safe_rms_norm_patch()}")
    st, load_meta = load_sentence_transformer(dtype=torch.float32, device="cpu")
    wrapper = EmbeddingGemma2Wrapper.from_sentence_transformer(st, normalize=True)
    force_eager_attention(wrapper)
    wrapper = wrapper.float().eval()
    hidden = int(wrapper.hidden_size)

    report: dict = {"created_at_utc": utc_now(), "git_sha": git_sha(REPO_ROOT), "load": load_meta}
    if not args.skip_check:
        for pack in packs:
            check = check_pack_math(st, wrapper, *pack)
            report[f"pack_check_fp32_{pack_name(*pack)}"] = check
            print(f"pack check fp32 {pack_name(*pack)}: {check}")

    programs: list[tuple[str, Path, tuple]] = []
    for seq_len in buckets:
        module = _bucket_module(wrapper, seq_len)
        embeds = torch.randn(1, seq_len, hidden, dtype=torch.float16)
        mask = torch.ones(1, seq_len, dtype=torch.float16)
        mask[:, seq_len // 2 :] = 0
        ep = out_dir / f"{bucket_name(seq_len)}.pt2"
        _export_program(module, (embeds, mask), ep)
        programs.append((bucket_name(seq_len), ep, BUCKET_IO))
    for pack in packs:
        module = _pack_module(wrapper, *pack)
        ep = out_dir / f"{pack_name(*pack)}.pt2"
        _export_program(module, example_pack_inputs(pack[0], pack[1], hidden), ep)
        programs.append((pack_name(*pack), ep, PACK_IO))

    rcs: dict[str, int] = {}
    if not args.skip_convert:
        if args.layout in ("separate", "both"):
            for entry, ep, (ins, outs) in programs:
                rcs[entry] = _convert(
                    ep, out_dir / f"{entry}.aimodel", entry=entry, inputs=ins, outputs=outs
                )
        if args.layout in ("multi", "both"):
            rcs["text_buckets"] = _convert_multi(programs, out_dir / "text_buckets.aimodel")
    report["programs"] = [{"entry": e, "ep": str(p)} for e, p, _ in programs]
    report["convert_rc"] = rcs
    write_json(out_dir / "text_buckets.export.json", report)
    return 1 if any(rcs.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
