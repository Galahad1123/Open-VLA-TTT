"""Qwen2 GQA with causal current-observation tokens and visible memory K/V prefixes.

Uses the existing Q/K/V/output projections. Math SDPA supports inner-loop double
backward. Original token positions are unchanged; prefix keys have no RoPE.
"""

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv


def attention_mask(valid, prefix_valid=None, *, dtype=torch.float32):
    batch, length = valid.shape
    causal = torch.ones(length, length, device=valid.device, dtype=torch.bool).tril()
    allowed = causal[None] & valid[:, None, :]
    if prefix_valid is not None:
        allowed = torch.cat((prefix_valid[:, None, :].expand(-1, length, -1), allowed), dim=-1)
    # Eager HF attention uses softmax: avoid NaNs for an invalid leading query
    # with no visible key. Give only those discarded rows a zero-input self key.
    empty = ~allowed.any(-1)
    offset = allowed.shape[-1] - length
    row = torch.arange(length, device=valid.device)
    allowed[:, row, row + offset] |= empty
    # Query outputs at invalid positions are discarded after every layer.
    return torch.zeros(
        (batch, 1, length, allowed.shape[-1]), device=valid.device, dtype=dtype
    ).masked_fill(~allowed[:, None], float("-inf"))


def attend(layer, x, valid, rotary, *, prefix=None, prefix_valid=None):
    attn = layer.self_attn
    batch, length, _ = x.shape
    d = attn.head_dim
    x = x.masked_fill(~valid[:, :, None], 0)
    q = attn.q_proj(x).view(batch, length, -1, d).transpose(1, 2)
    k = attn.k_proj(x).view(batch, length, -1, d).transpose(1, 2)
    v = attn.v_proj(x).view(batch, length, -1, d).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, *rotary)
    if prefix is not None:
        prefix = prefix.to(x).masked_fill(~prefix_valid[:, :, None], 0)
        pk = attn.k_proj(prefix).view(batch, prefix.shape[1], -1, d).transpose(1, 2)
        pv = attn.v_proj(prefix).view(batch, prefix.shape[1], -1, d).transpose(1, 2)
        k, v = torch.cat((pk, k), dim=2), torch.cat((pv, v), dim=2)
    k, v = repeat_kv(k, attn.num_key_value_groups), repeat_kv(v, attn.num_key_value_groups)
    mask = attention_mask(valid, prefix_valid, dtype=q.dtype)
    with sdpa_kernel(SDPBackend.MATH):
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=attn.attention_dropout if layer.training else 0.0,
            scale=attn.scaling,
        )
    y = y.transpose(1, 2).reshape(batch, length, -1)
    return attn.o_proj(y).masked_fill(~valid[:, :, None], 0)


def finish(layer, residual, mixed, valid):
    hidden = residual + mixed
    hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
    return hidden.masked_fill(~valid[:, :, None], 0)
