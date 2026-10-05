"""Outer-learned initialization for a two-layer GeLU fast MLP."""

import torch
from torch import nn


class FastMLP(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(width, hidden))
        self.b1 = nn.Parameter(torch.zeros(hidden))
        self.w2 = nn.Parameter(torch.empty(hidden, width))
        self.b2 = nn.Parameter(torch.zeros(width))
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)

    def initial_weights(self, batch):
        return tuple(
            w.unsqueeze(0).expand(batch, *w.shape).clone()
            for w in (self.w1, self.b1, self.w2, self.b2)
        )
