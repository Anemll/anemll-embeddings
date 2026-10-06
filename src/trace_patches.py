"""Export-time HF patches: replace shape-derived ``aten::Int`` with Python sizes.

coremltools 9.0 cannot const-fold ``int(non_scalar_tensor)`` from
``hidden_states.shape`` unpacks inside EmbeddingGemma 2 attention / PLE.
"""

from __future__ import annotations

import transformers.models.embedding_gemma2.modeling_embedding_gemma2 as eg2
import torch
import torch.nn.functional as F


def _rotate_half_chunk(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


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
            key_states = (
                key.unsqueeze(2)
                .expand(batch, n_kv, n_rep, seq_len, head_dim)
                .reshape(batch, n_kv * n_rep, seq_len, head_dim)
            )
            value_states = (
                value.unsqueeze(2)
                .expand(batch, n_kv, n_rep, seq_len, head_dim)
                .reshape(batch, n_kv * n_rep, seq_len, head_dim)
            )
        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
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
        attn_output = attn_output.transpose(1, 2).contiguous()
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
        query_states = (
            self.q_proj(hidden_states)
            .view(batch, seq_len, n_heads, head_dim)
            .transpose(1, 2)
        )
        key_states = (
            self.k_proj(hidden_states).view(batch, seq_len, n_kv, head_dim).transpose(1, 2)
        )
        value_states = (
            self.v_proj(hidden_states).view(batch, seq_len, n_kv, head_dim).transpose(1, 2)
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


def apply_fixed_shape_patches(seq_len: int, batch: int = 1) -> dict[str, int]:
    """Monkey-patch EmbeddingGemma 2 classes for a single fixed (B, S)."""
    eg2.rotate_half = _rotate_half_chunk
    eg2.eager_attention_forward = _make_eager_attention(batch, seq_len)
    eg2.EmbeddingGemma2Attention.forward = _make_attention_forward(batch, seq_len)
    eg2.EmbeddingGemma2TextPLE.forward = _make_ple_forward(batch, seq_len)
    return {"seq_len": int(seq_len), "batch": int(batch), "patched": 1}
