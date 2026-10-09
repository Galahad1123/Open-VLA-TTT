"""Shared associative memory with separate reads/writes and a legacy RoboTTT wrapper."""

import torch
from torch import nn

from .fast_models import FastMLP
from .functional import mlp, temporal_rope
from .state import FastWeightState
from .updates.gradient_descent import GradientDescent
from .updates.momentum_decay import MomentumDecay
from .updates.t2mem_adaptive import T2MemAdaptive


class AssociativeMemory(nn.Module):
    def __init__(
        self,
        input_dim,
        width,
        hidden,
        learning_rate=0.1,
        gate_init=0.001,
        *,
        update_rule="gradient_descent",
        **update_kwargs,
    ):
        super().__init__()
        if width <= 0 or width % 2 or hidden <= 0:
            raise ValueError("memory width must be positive and even; hidden must be positive")
        self.norm = nn.RMSNorm(input_dim)
        self.qkv = nn.Linear(input_dim, 3 * width, bias=False)
        self.output = nn.Linear(width, input_dim, bias=False)
        self.fast = FastMLP(width, hidden)
        if update_rule == "gradient_descent":
            self.update = GradientDescent(learning_rate)
        elif update_rule == "momentum_decay":
            self.update = MomentumDecay(learning_rate, input_dim, **update_kwargs)
        elif update_rule == "t2mem_adaptive":
            self.update = T2MemAdaptive()
        else:
            raise ValueError("Unknown memory update rule")
        # Preserve the original parameter names and forward behavior for old checkpoints.
        self.gate = nn.Parameter(torch.full((input_dim,), gate_init))

    def initial_state(self, batch):
        weights = self.fast.initial_weights(batch)
        zero = torch.zeros(batch, device=self.gate.device, dtype=torch.long)
        return FastWeightState(
            weights,
            zero,
            zero.clone(),
            zero.clone(),
            tuple(torch.zeros_like(w) for w in weights)
            if isinstance(self.update, MomentumDecay)
            else None,
        )

    def reset_rows(self, state, reset_mask):
        fresh = self.initial_state(len(state.position))
        mask = reset_mask.to(device=state.position.device, dtype=torch.bool)
        if mask.shape != state.position.shape:
            raise ValueError("reset_mask must have shape (batch,)")

        def select(old, new):
            return tuple(
                torch.where(mask.reshape(-1, *([1] * (w.ndim - 1))), n, w)
                for w, n in zip(old, new, strict=True)
            )

        return FastWeightState(
            select(state.weights, fresh.weights),
            torch.where(mask, 0, state.position),
            torch.where(mask, 0, state.decision_index),
            torch.where(mask, 0, state.update_count),
            None if state.auxiliary is None else select(state.auxiliary, fresh.auxiliary),
        )

    def project(self, x, position):
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=-1)
        return temporal_rope(q, position), temporal_rope(k, position), v

    def read(self, x, state, *, position=None):
        """Raw projected readout; no residual, output gate, mutation, or clock advance."""
        with torch.autocast(x.device.type, enabled=False):
            q = self.qkv(self.norm(x.to(self.fast.w1.dtype))).chunk(3, dim=-1)[0]
            q = temporal_rope(q, state.position if position is None else position)
            return self.output(mlp(state.weights, q))

    def write(
        self,
        x,
        state=None,
        *,
        write_mask=None,
        token_mask=None,
        advance_position=True,
        advance_decision=True,
    ):
        if torch.is_inference_mode_enabled():
            raise RuntimeError(
                "TTT requires no_grad, not inference_mode, for fast-weight gradients"
            )
        outer_grad = torch.is_grad_enabled()
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
            selected = torch.ones(x.shape[:2], device=x.device, dtype=torch.bool)
            if token_mask is not None:
                selected = token_mask.to(device=x.device, dtype=torch.bool)
                if selected.shape != x.shape[:2]:
                    raise ValueError("token_mask must have shape (batch, tokens)")
            active = write_mask & selected.any(dim=1)
            selected = selected & write_mask[:, None]
            values = x.to(self.fast.w1.dtype).masked_fill(~selected[:, :, None], 0)
            weights = tuple(
                w if w.requires_grad else w.detach().requires_grad_() for w in state.weights
            )
            _, k, v = self.project(values, state.position)
            error = (mlp(weights, k) - v).square()
            if token_mask is None:
                # Preserve the historical reduction order on the default path.
                loss = error.mean((1, 2)).sum()
            else:
                error = error.masked_fill(~selected[:, :, None], 0)
                count = selected.sum(1).clamp_min(1) * error.shape[-1]
                loss = (error.sum((1, 2)) / count).sum()
            gradients = torch.autograd.grad(loss, weights, create_graph=outer_grad)
            features = values.sum(1) / selected.sum(1).clamp_min(1)[:, None]
            if isinstance(self.update, T2MemAdaptive):
                # Treat the legacy MLP as one head; preserve its masked objective.
                scale = (x.shape[1] / selected.sum(1).clamp_min(1)).sqrt()[:, None, None]

                def evaluate(parameters, keys):
                    prediction = mlp(tuple(p[:, 0] for p in parameters), keys[:, 0])
                    return (prediction.masked_fill(~selected[:, :, None], 0) * scale)[:, None]

                updated, auxiliary = self.update.step(
                    tuple(w[:, None] for w in weights),
                    tuple(g[:, None] for g in gradients),
                    active,
                    initial=tuple(w[:, None] for w in self.fast.initial_weights(len(x))),
                    keys=k[:, None],
                    values=(v.masked_fill(~selected[:, :, None], 0) * scale)[:, None],
                    forward=evaluate,
                )
                updated = tuple(w[:, 0] for w in updated)
            else:
                updated, auxiliary = self.update.step(
                    weights, gradients, active, state.auxiliary, features
                )
            increment = active.long()
            result = FastWeightState(
                updated,
                state.position + increment * int(advance_position),
                state.decision_index + increment * int(advance_decision),
                state.update_count + increment,
                auxiliary,
            )
        return result if outer_grad else result.detach()

    def forward(
        self,
        x,
        state=None,
        *,
        write_mask=None,
        token_mask=None,
        advance_position=True,
        advance_decision=True,
    ):
        """RoboTTT: update then read at the same position, with its original residual."""
        if state is None:
            state = self.initial_state(x.shape[0])
        result = self.write(
            x,
            state,
            write_mask=write_mask,
            token_mask=token_mask,
            advance_position=advance_position,
            advance_decision=advance_decision,
        )
        out = x + (self.gate.tanh() * self.read(x, result, position=state.position)).to(x.dtype)
        return (out, result) if torch.is_grad_enabled() else (out.detach(), result.detach())
