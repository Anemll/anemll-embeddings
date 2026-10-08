"""Slim host embed table for Core AI inference (no full checkpoint).

``api.coreai_host.embeds_from_ids`` only calls ``text_model.get_input_embeddings()``.
EmbeddingGemma 2's table is ``language_model.embed_tokens`` (BF16
``[262144, 512]``) scaled by ``sqrt(hidden_size)`` — ``sqrt(512)``. When
``embed_tokens.safetensors`` is present we load that; otherwise Embedder
falls back to the full Sentence-Transformers checkpoint.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import torch
from torch import nn

SLIM_EMBED_NAME = "embed_tokens.safetensors"
EMBED_WEIGHT_KEY = "embed_tokens.weight"
EMBED_SCALE_KEY = "embed_scale"
# Gemma scales token rows by sqrt(hidden). Text hidden is 512.
EMBED_SCALE = 512**0.5
PROMPTS_FILE = "config_sentence_transformers.json"


class HostEmbedTable(nn.Module):
    """``nn.Embedding`` plus Gemma's ``sqrt(hidden)`` scale."""

    def __init__(self, weight: torch.Tensor, embed_scale: float, padding_idx: int = 0) -> None:
        super().__init__()
        self.embed = nn.Embedding.from_pretrained(
            weight.detach().to(torch.float32), padding_idx=padding_idx, freeze=True
        )
        self.embed_scale = float(embed_scale)

    def get_input_embeddings(self) -> HostEmbedTable:
        return self

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed(input_ids) * self.embed_scale


def parse_safetensors_header(blob: bytes) -> tuple[int, dict]:
    if len(blob) < 8:
        raise ValueError("safetensors header prefix missing")
    header_len = struct.unpack_from("<Q", blob, 0)[0]
    need = 8 + header_len
    if len(blob) < need:
        raise ValueError(f"safetensors header truncated ({len(blob)} < {need})")
    header = json.loads(blob[8:need].decode("utf-8"))
    return header_len, header


def _load_safetensors(path: Path) -> dict[str, tuple[str, list[int], bytes]]:
    raw = path.read_bytes()
    header_len, header = parse_safetensors_header(raw)
    out: dict[str, tuple[str, list[int], bytes]] = {}
    for name, info in header.items():
        if name == "__metadata__" or not isinstance(info, dict):
            continue
        start, end = info["data_offsets"]
        abs_start = 8 + header_len + int(start)
        out[name] = (str(info["dtype"]), list(info["shape"]), raw[abs_start : 8 + header_len + int(end)])
    return out


def _to_tensor(dtype: str, shape: list[int], data: bytes) -> torch.Tensor:
    if dtype == "BF16":
        vals = torch.frombuffer(bytearray(data), dtype=torch.bfloat16)
    elif dtype == "F32":
        vals = torch.frombuffer(bytearray(data), dtype=torch.float32)
    elif dtype == "F16":
        vals = torch.frombuffer(bytearray(data), dtype=torch.float16)
    else:
        raise ValueError(f"unsupported slim embed dtype {dtype}")
    return vals.reshape(shape).contiguous()


def load_prompts(model_dir: Path) -> dict[str, str]:
    path = model_dir / PROMPTS_FILE
    if not path.is_file():
        return {}
    pack = json.loads(path.read_text(encoding="utf-8"))
    raw = pack.get("prompts") or {}
    return {str(key): str(value) for key, value in raw.items()}


def load_slim_host(model_dir: Path | str) -> tuple[HostEmbedTable, dict[str, str]] | None:
    root = Path(model_dir)
    slim = root / SLIM_EMBED_NAME
    if not slim.is_file():
        return None
    tensors = _load_safetensors(slim)
    if EMBED_WEIGHT_KEY not in tensors:
        raise ValueError(f"{slim} has no {EMBED_WEIGHT_KEY}")
    dtype, shape, data = tensors[EMBED_WEIGHT_KEY]
    weight = _to_tensor(dtype, shape, data)
    scale = float(EMBED_SCALE)
    if EMBED_SCALE_KEY in tensors:
        s_dtype, s_shape, s_data = tensors[EMBED_SCALE_KEY]
        if s_dtype == "F32" and len(s_data) >= 4:
            scale = float(struct.unpack_from("<f", s_data, 0)[0])
        else:
            scale = float(_to_tensor(s_dtype, s_shape, s_data).reshape(-1)[0])
    pad = 0
    config = root / "config.json"
    if config.is_file():
        cfg = json.loads(config.read_text(encoding="utf-8"))
        text = cfg.get("text_config") or {}
        pad = int(text.get("pad_token_id") or cfg.get("pad_token_id") or 0)
    table = HostEmbedTable(weight, scale, padding_idx=pad)
    return table, load_prompts(root)
