"""Independent-headed associative memory for observation interfaces.

No residual/gating in this module: connections own query sources and fusion.
Legacy single-head fast weights and checkpoints remain unchanged.
"""

from dataclasses import replace

import torch
from torch import nn
from torch.nn import functional as F

from .state import FastWeightState
from .updates.gradient_descent import GradientDescent
from .updates.momentum_decay import MomentumDecay
from .updates.t2mem_adaptive import T2MemAdaptive


def headed_mlp(weights, x):
    w1, b1, w2, b2 = weights
    return F.gelu(x @ w1 + b1) @ w2 + b2


def slot_rope(x):
    """Interleaved RoPE on interface-slot positions, never the environment clock."""
    d = x.shape[-1]
    freq = 10000.0 ** (-torch.arange(0, d, 2, device=x.device, dtype=x.dtype) / d)
    angle = torch.arange(x.shape[-2], device=x.device, dtype=x.dtype)[:, None] * freq
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack(
        (a * angle.cos() - b * angle.sin(), b * angle.cos() + a * angle.sin()), -1
    ).flatten(-2)


class HeadedMLP(nn.Module):
    def __init__(self, heads, dim, hidden):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(heads, dim, hidden) * 0.02)
        self.b1 = nn.Parameter(torch.zeros(heads, 1, hidden))
        self.w2 = nn.Parameter(torch.randn(heads, hidden, dim) * 0.02)
        self.b2 = nn.Parameter(torch.zeros(heads, 1, dim))

    def initial(self, batch):
        return tuple(
            p[None].expand(batch, *p.shape).clone() for p in (self.w1, self.b1, self.w2, self.b2)
        )


class HeadedMemory(nn.Module):
    def __init__(
        self,
        input_dim,
        heads=16,
        head_dim=64,
        hidden=256,
        *,
        update_rule="gradient_descent",
        learning_rate=0.1,
        rope=False,
        **update_kwargs,
    ):
        super().__init__()
        if min(heads, head_dim, hidden) < 1 or head_dim % 2:
            raise ValueError("Positive memory dimensions and even head_dim required")
        self.heads, self.head_dim, self.rope = heads, head_dim, rope
        self.q, self.k, self.v = (nn.Linear(input_dim, heads * head_dim) for _ in range(3))
        self.q_norm, self.k_norm, self.v_norm = (nn.RMSNorm(head_dim, eps=1e-6) for _ in range(3))
        self.output = nn.Linear(heads * head_dim, input_dim)
        self.fast = HeadedMLP(heads, head_dim, hidden)
        if update_rule == "t2mem_adaptive":
            self.update = T2MemAdaptive()
        elif update_rule == "momentum_decay":
            self.update = MomentumDecay(learning_rate, input_dim, **update_kwargs)
        elif update_rule == "gradient_descent":
            self.update = GradientDescent(learning_rate)
        else:
            raise ValueError("Unknown update rule")

    def initial_state(self, batch):
        w = self.fast.initial(batch)
        zero = torch.zeros(batch, device=w[0].device, dtype=torch.long)
        aux = (
            tuple(torch.zeros_like(x) for x in w)
            if isinstance(self.update, MomentumDecay)
            else None
        )
        return FastWeightState(w, zero, zero.clone(), zero.clone(), aux)

    def reset_rows(self, state, reset):
        reset = reset.to(state.position.device, dtype=torch.bool)
        if reset.shape != state.position.shape:
            raise ValueError("reset must match batch")
        fresh = self.initial_state(len(reset))

        def choose(a, b):
            return tuple(
                torch.where(reset.reshape(-1, *([1] * (x.ndim - 1))), y, x)
                for x, y in zip(a, b, strict=True)
            )

        return FastWeightState(
            choose(state.weights, fresh.weights),
            torch.where(reset, 0, state.position),
            torch.where(reset, 0, state.decision_index),
            torch.where(reset, 0, state.update_count),
            None if state.auxiliary is None else choose(state.auxiliary, fresh.auxiliary),
        )

    def project(self, x, projection, norm, rope=False):
        y = projection(x.to(self.fast.w1.dtype))
        y = norm(y.reshape(len(x), x.shape[1], self.heads, self.head_dim).transpose(1, 2))
        return slot_rope(y) if rope else y

    def read(self, x, state):
        if state.position.shape != (len(x),):
            raise ValueError("Memory batch changed; reset first")
        with torch.autocast(x.device.type, enabled=False):
            q = self.project(x, self.q, self.q_norm, self.rope)
            y = headed_mlp(state.weights, q).transpose(1, 2).flatten(2)
            return self.output(y).to(x.dtype)

    def write(self, x, state=None, *, write_mask=None):
        if torch.is_inference_mode_enabled():
            raise RuntimeError("Use no_grad, not inference_mode, for memory writes")
        outer = torch.is_grad_enabled()
        with torch.enable_grad(), torch.autocast(x.device.type, enabled=False):
            state = self.initial_state(len(x)) if state is None else state
            if state.position.shape != (len(x),):
                raise ValueError("Memory batch changed; reset first")
            active = (
                torch.ones_like(state.position, dtype=torch.bool)
                if write_mask is None
                else write_mask.to(x.device, dtype=torch.bool)
            )
            if active.shape != state.position.shape:
                raise ValueError("write_mask must match batch")
            if not active.any():
                return state if outer else state.detach()
            x = x.to(self.fast.w1.dtype)
            k, v = (
                self.project(x, self.k, self.k_norm, self.rope),
                self.project(x, self.v, self.v_norm),
            )
            weights = tuple(
                w if w.requires_grad else w.detach().requires_grad_() for w in state.weights
            )
            loss = (headed_mlp(weights, k) - v).square().mean((1, 2, 3)).sum()
            gradients = torch.autograd.grad(loss, weights, create_graph=outer)
            if isinstance(self.update, T2MemAdaptive):
                updated, aux = self.update.step(
                    weights,
                    gradients,
                    active,
                    initial=self.fast.initial(len(x)),
                    keys=k,
                    values=v,
                    forward=headed_mlp,
                )
            else:
                updated, aux = self.update.step(
                    weights, gradients, active, state.auxiliary, x.mean(1)
                )
            result = replace(
                state,
                weights=updated,
                auxiliary=aux,
                position=state.position + active.long(),
                update_count=state.update_count + active.long(),
            )
        return result if outer else result.detach()
