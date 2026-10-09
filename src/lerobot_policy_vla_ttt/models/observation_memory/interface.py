"""Observation-only aggregation; no action, state, flow-time or self-attention input."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ObservationCache:
    summaries: tuple
    contexts: tuple
    memory: tuple


class InterfaceLayer(nn.Module):
    def __init__(self, dim, heads, hidden):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )
        self.gate = nn.Parameter(torch.full((dim,), 0.01))

    def summarize(self, tokens, vl, valid):
        if valid.shape != vl.shape[:2] or not valid.any(1).all():
            raise ValueError("Each row requires valid VL tokens")
        vl = vl.masked_fill(~valid[:, :, None], 0)
        return self.attention(
            self.norm(tokens), vl, vl, key_padding_mask=~valid, need_weights=False
        )[0]

    def finish(self, tokens, summary, readout=None):
        mixed = summary if readout is None else summary + self.gate.tanh() * readout
        tokens = tokens + mixed
        return tokens + self.ff(tokens)
