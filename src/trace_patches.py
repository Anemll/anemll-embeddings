"""Export-time HF patches: replace shape-derived ``aten::Int`` with Python sizes.

coremltools 9.0 cannot const-fold ``int(non_scalar_tensor)`` from
``hidden_states.shape`` unpacks inside EmbeddingGemma 2 attention / PLE.

Preferred-ANE on ``text_embeds_s320`` names ``GPU_region_0`` outputs
``reshape->permute`` (head ``view`` + ``transpose(1, 2)``) and
``strided_slice->reshape`` (PLE ``[:, :, layer, :]``). Head layout uses
``swap_mid_dims``. Each PLE layer is a size-24 one-hot matmul. K's last-two
transpose is a factored one-hot (``320×256`` and ``320×512``). The two
``broadcasting_divide`` ops are the mask mean and the L2 norm, not a static
scale, so they stay.
"""

from __future__ import annotations

import transformers.models.embedding_gemma2.modeling_embedding_gemma2 as eg2
import torch
import torch.nn.functional as F

from src.audio_export_patches import swap_last_two, swap_mid_dims


def _rotate_half_chunk(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _factor_transpose_last(
    x: torch.Tensor, s_factors: tuple[int, int], d_factors: tuple[int, int]
) -> torch.Tensor:
    """``[..., S, D] → [..., D, S]`` by four small one-hot swaps.

    ``S = s1*s0`` and ``D = d1*d0``. Each swap's one-hot is ``(a*b)²`` for
    factors of 16–32, not ``(S*D)²``.
    """
    s1, s0 = s_factors
    d1, d0 = d_factors
    lead = x.shape[:-2]
    n = 1
    for size in lead:
        n *= int(size)
    t = x.reshape(n, s1, s0, d1, d0)
    t = swap_mid_dims(t.reshape(n * s1, s0, d1, d0)).reshape(n, s1, d1, s0, d0)
    t = swap_mid_dims(t.reshape(n, s1, d1, s0 * d0)).reshape(n, d1, s1, s0, d0)
    t = swap_mid_dims(t.reshape(n * d1 * s1, s0, d0, 1)).reshape(n, d1, s1, d0, s0)
    t = swap_mid_dims(t.reshape(n * d1, s1, d0, s0)).reshape(n, d1, d0, s1, s0)
    return t.reshape(*lead, d1 * d0, s1 * s0)


def bake_k_layout(key: torch.Tensor) -> torch.Tensor:
    """``[B, H, S, D] → [B, H, D, S]`` with no ``transpose``.

    Live activation transposes were 20× ``1x4x320x256`` and 4× ``1x4x320x512``.
    A single ``(S*D)²`` one-hot does not fit; factor those two shapes.
    """
    seq = int(key.shape[-2])
    dim = int(key.shape[-1])
    if seq == 320 and dim == 256:
        return _factor_transpose_last(key, (20, 16), (16, 16))
    if seq == 320 and dim == 512:
        return _factor_transpose_last(key, (20, 16), (32, 16))
    return swap_last_two(key)


def repeat_kv_index(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """GQA repeat along heads via ``index_select`` (not expand+reshape).

    Core AI fuses expand/reshape GQA into ``mps_spi.sdpa`` and then aborts:
    "grouping for the value tensor does not match the one available on the key
    tensor". Gather materializes matching Q/K/V head counts.
    """
    n_rep = int(n_rep)
    if n_rep <= 1:
        return x
    n_kv = int(x.shape[1])
    idx = torch.arange(n_kv, device=x.device, dtype=torch.long).repeat_interleave(n_rep)
    return x.index_select(1, idx).contiguous()


def _make_eager_attention(batch: int, seq_len: int):
    def eager_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout: float | int = 0.0,
        scaling: float | None = None,
        softcap: float | None = None,
        **kwargs,
    ):
        head_dim = int(module.head_dim)
        n_kv = int(module.k_proj.out_features) // head_dim
        n_heads = int(module.q_proj.out_features) // head_dim
        n_rep = n_heads // n_kv
        if scaling is None:
            scaling = head_dim**-0.5
        if n_rep == 1:
            key_states = key
            value_states = value
        else:
            key_states = repeat_kv_index(key, n_rep)
            value_states = repeat_kv_index(value, n_rep)
        attn_weights = torch.matmul(query, bake_k_layout(key_states)) * scaling
        if softcap is not None:
            attn_weights = attn_weights / softcap
            attn_weights = torch.tanh(attn_weights)
            attn_weights = attn_weights * softcap
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        if dropout and module.training:
            attn_weights = F.dropout(attn_weights, p=float(dropout), training=True)
        attn_output = torch.matmul(attn_weights, value_states)
        # [B, H, S, D] → [B, S, H, D]. One-hot, not transpose: that transpose
        # fused with the head reshape as GPU ``reshape->permute``.
        attn_output = swap_mid_dims(attn_output).contiguous()
        return attn_output, attn_weights

    return eager_attention_forward


def _make_attention_forward(batch: int, seq_len: int):
    eager = _make_eager_attention(batch, seq_len)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ):
        head_dim = int(self.head_dim)
        n_heads = int(self.q_proj.out_features) // head_dim
        n_kv = int(self.k_proj.out_features) // head_dim
        # Heads-last view, then one-hot swap to [B, H, S, D]. Do not
        # ``transpose(1, 2)`` — that pair is GPU ``reshape->permute``.
        query_states = swap_mid_dims(
            self.q_proj(hidden_states).reshape(batch, seq_len, n_heads, head_dim)
        )
        key_states = swap_mid_dims(
            self.k_proj(hidden_states).reshape(batch, seq_len, n_kv, head_dim)
        )
        value_states = swap_mid_dims(
            self.v_proj(hidden_states).reshape(batch, seq_len, n_kv, head_dim)
        )
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        value_states = self.v_norm(value_states)
        cos, sin = position_embeddings
        query_states, key_states = eg2.apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )
        attn_output, attn_weights = eager(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0,
            scaling=self.scaling,
        )
        attn_output = attn_output.reshape(batch, seq_len, n_heads * head_dim).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    return forward


def _make_ple_forward(batch: int, seq_len: int):
    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        per_layer_projection = (
            self.per_layer_model_projection(inputs_embeds)
            * self.per_layer_model_projection_scale
        )
        per_layer_projection = per_layer_projection.reshape(
            batch,
            seq_len,
            self.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        return self.per_layer_projection_norm(per_layer_projection)

    return forward


def _take_ple_layer(ple: torch.Tensor, layer: int) -> torch.Tensor:
    """``[B, S, L, W] → [B, S, W]`` for one layer via a length-L one-hot.

    HF indexes ``per_layer_inputs[:, :, i, :]``. That lowered as
    ``strided_slice->reshape`` inside ``GPU_region_0``.
    """
    batch, seq, n_layers, width = (int(s) for s in ple.shape)
    select = ple.new_zeros(1, n_layers)
    select[0, int(layer)] = 1
    flat = ple.reshape(batch * seq, n_layers, width)
    return (select @ flat).reshape(batch, seq, width)


def _text_model_forward(
    self,
    input_ids: torch.Tensor | None = None,
    attention_mask: torch.Tensor | dict | None = None,
    position_ids: torch.Tensor | None = None,
    inputs_embeds: torch.Tensor | None = None,
    **kwargs,
) -> eg2.BaseModelOutput:
    """Same as HF ``EmbeddingGemma2TextModel.forward``, PLE layers via matmul."""
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
    if input_ids is not None:
        inputs_embeds = self.embed_tokens(input_ids)
    per_layer_inputs = self.ple(inputs_embeds)
    if position_ids is None:
        position_ids = torch.arange(
            inputs_embeds.shape[1], device=inputs_embeds.device
        ).unsqueeze(0)
    if not isinstance(attention_mask, dict):
        mask_kwargs = {
            "config": self.config,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
        }
        attention_mask_mapping = {
            "full_attention": eg2.create_bidirectional_mask(**mask_kwargs),
            "sliding_attention": eg2.create_bidirectional_sliding_window_mask(**mask_kwargs),
        }
    else:
        attention_mask_mapping = attention_mask
    hidden_states = inputs_embeds
    position_embeddings = {}
    for layer_type in self.unique_layer_types:
        position_embeddings[layer_type] = self.rotary_emb(
            hidden_states, position_ids, layer_type
        )
    for i, encoder_layer in enumerate(self.layers):
        hidden_states = encoder_layer(
            hidden_states,
            _take_ple_layer(per_layer_inputs, i),
            attention_mask=attention_mask_mapping[self.config.layer_types[i]],
            position_embeddings=position_embeddings[self.config.layer_types[i]],
            **kwargs,
        )
    hidden_states = self.norm(hidden_states)
    hidden_states = self.embedding_projection(hidden_states)
    return eg2.BaseModelOutput(last_hidden_state=hidden_states)


def apply_fixed_shape_patches(seq_len: int, batch: int = 1) -> dict[str, object]:
    """Monkey-patch EmbeddingGemma 2 classes for a single fixed (B, S)."""
    eg2.rotate_half = _rotate_half_chunk
    eg2.eager_attention_forward = _make_eager_attention(batch, seq_len)
    eg2.EmbeddingGemma2Attention.forward = _make_attention_forward(batch, seq_len)
    eg2.EmbeddingGemma2TextPLE.forward = _make_ple_forward(batch, seq_len)
    eg2.EmbeddingGemma2TextModel.forward = _text_model_forward
    return {
        "seq_len": int(seq_len),
        "batch": int(batch),
        "patched": 1,
        "attn_layout": "swap_mid_dims",
        "ple_index": "onehot_layer",
    }
