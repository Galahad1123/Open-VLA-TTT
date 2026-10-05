"""KV associative memory, updated before Q readout with an outer-learned gate."""

import torch
from torch import nn

from .fast_models import FastMLP
from .functional import mlp, temporal_rope
from .state import FastWeightState
from .updates.gradient_descent import GradientDescent


class AssociativeMemory(nn.Module):
    def __init__(self, input_dim, width, hidden, learning_rate=0.1, gate_init=0.001):
        super().__init__()
        if width <= 0 or width % 2 or hidden <= 0:
            raise ValueError("memory width must be positive and even; hidden must be positive")
        self.norm = nn.RMSNorm(input_dim)
        self.qkv = nn.Linear(input_dim, 3 * width, bias=False)
        self.output = nn.Linear(width, input_dim, bias=False)
        self.fast = FastMLP(width, hidden)
        self.update = GradientDescent(learning_rate)
        self.gate = nn.Parameter(torch.full((input_dim,), gate_init))

    def initial_state(self, batch):
        return FastWeightState(
            self.fast.initial_weights(batch),
            torch.zeros(batch, device=self.gate.device, dtype=torch.long),
        )

    def reset_rows(self, state, reset_mask):
        fresh = self.initial_state(len(state.position))
        mask = reset_mask.to(device=state.position.device, dtype=torch.bool)
        if mask.shape != state.position.shape:
            raise ValueError("reset_mask must have shape (batch,)")
        return FastWeightState(
            tuple(
                torch.where(mask.reshape(-1, *([1] * (w.ndim - 1))), n, w)
                for w, n in zip(state.weights, fresh.weights, strict=True)
            ),
            torch.where(mask, 0, state.position),
        )

    def project(self, x, position):
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=-1)
        return temporal_rope(q, position), temporal_rope(k, position), v

    def forward(self, x, state=None, *, write_mask=None):
        if torch.is_inference_mode_enabled():
            raise RuntimeError(
                "TTT requires no_grad, not inference_mode, for fast-weight gradients"
            )
        outer_grad = torch.is_grad_enabled()
        dtype = self.fast.w1.dtype
        # Keep inner optimization out of mixed precision autocast.
        with torch.enable_grad(), torch.autocast(x.device.type, enabled=False):
            if state is None:
                state = self.initial_state(x.shape[0])
            if state.position.shape != (x.shape[0],):
                raise ValueError("Memory batch changed: reset state first")
            if write_mask is None:
                write_mask = torch.ones_like(state.position, dtype=torch.bool)
            write_mask = write_mask.to(device=x.device, dtype=torch.bool)
            if write_mask.shape != state.position.shape:
                raise ValueError("write_mask must have shape (batch,)")
            weights = tuple(
                w if w.requires_grad else w.detach().requires_grad_() for w in state.weights
            )
            q, k, v = self.project(x.to(dtype), state.position)
            loss = (mlp(weights, k) - v).square().mean((1, 2)).sum()
            gradients = torch.autograd.grad(loss, weights, create_graph=outer_grad)
            updated = self.update(weights, gradients, write_mask)
            out = x + (self.gate.tanh() * self.output(mlp(updated, q))).to(x.dtype)
            result = FastWeightState(updated, state.position + write_mask.long())
        return (out, result) if outer_grad else (out.detach(), result.detach())
