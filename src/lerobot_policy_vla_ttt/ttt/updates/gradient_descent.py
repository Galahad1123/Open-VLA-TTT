"""Differentiable ordinary GD; replaceable independently of memory layout."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class GradientDescent(nn.Module):
    def __init__(self, learning_rate=0.1):
        super().__init__()
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        self.raw_lr = nn.Parameter(torch.tensor(math.log(math.expm1(learning_rate))))

    @property
    def learning_rate(self):
        return F.softplus(self.raw_lr)

    def forward(self, weights, gradients, write_mask):
        return tuple(
            torch.where(
                write_mask.reshape(-1, *([1] * (w.ndim - 1))), w - self.learning_rate * g, w
            )
            for w, g in zip(weights, gradients, strict=True)
        )
