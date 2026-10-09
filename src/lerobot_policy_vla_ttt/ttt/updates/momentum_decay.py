"""Titans-style first-order momentum and weight decay (paper equation 3).

Coefficients are per sample/update, predicted from masked mean input features.
This observation-level reduction is a VLA adaptation, not token-wise Titans.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .gradient_descent import GradientDescent


class MomentumDecay(GradientDescent):
    def __init__(
        self,
        learning_rate,
        input_dim,
        *,
        momentum=True,
        forgetting=True,
        momentum_init=0.9,
        decay_init=0.01,
        data_dependent=True,
    ):
        super().__init__(learning_rate)
        if any(not 0 < v < 1 for v in (momentum_init, decay_init)):
            raise ValueError("Momentum/decay initial values must be in (0, 1)")
        self.momentum = momentum
        self.forgetting = forgetting
        self.raw_momentum = nn.Parameter(
            torch.tensor(math.log(momentum_init / (1 - momentum_init)))
        )
        self.raw_decay = nn.Parameter(torch.tensor(math.log(decay_init / (1 - decay_init))))
        self.coefficients = nn.Linear(input_dim, 3, bias=False) if data_dependent else None
        if self.coefficients is not None:
            nn.init.zeros_(self.coefficients.weight)

    def step(self, weights, gradients, write_mask, auxiliary=None, features=None):
        if features is None:
            raise ValueError("MomentumDecay requires masked input features")
        delta = features.new_zeros((len(features), 3))
        if self.coefficients is not None:
            delta = self.coefficients(features)
        lr = F.softplus(self.raw_lr + delta[:, 0])
        eta = (self.raw_momentum + delta[:, 1]).sigmoid() if self.momentum else lr * 0
        alpha = (self.raw_decay + delta[:, 2]).sigmoid() if self.forgetting else lr * 0
        if auxiliary is None:
            auxiliary = tuple(torch.zeros_like(w) for w in weights)
        updated, momenta = [], []
        for w, g, previous in zip(weights, gradients, auxiliary, strict=True):
            shape = (-1,) + (1,) * (w.ndim - 1)
            active = write_mask.reshape(shape)
            velocity = eta.reshape(shape) * previous - lr.reshape(shape) * g
            candidate = (1 - alpha.reshape(shape)) * w + velocity
            updated.append(torch.where(active, candidate, w))
            momenta.append(torch.where(active, velocity, previous))
        return tuple(updated), tuple(momenta)
