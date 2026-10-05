"""Pure batched fast-model operations; no module parameters are mutated."""

import torch
from torch.nn import functional as F


def mlp(weights, x):
    w1, b1, w2, b2 = weights
    return F.gelu(x @ w1 + b1[:, None]) @ w2 + b2[:, None]


def temporal_rope(x, position, theta=10000.0):
    """One temporal position per trajectory row, shared by chunk tokens."""
    half = x.shape[-1] // 2
    frequency = theta ** (-torch.arange(half, device=x.device, dtype=x.dtype) / half)
    angle = position[:, None, None] * frequency
    left, right = x.chunk(2, dim=-1)
    return torch.cat(
        (left * angle.cos() - right * angle.sin(), right * angle.cos() + left * angle.sin()), dim=-1
    )
