"""Shared memory operations only; each architecture specifies its own topology."""

from dataclasses import replace

import torch
from torch import nn

from ...ttt.memory import AssociativeMemory


class MemoryConnection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.memory = AssociativeMemory(
            config.embed_dim,
            config.memory_dim,
            config.memory_hidden_dim,
            config.memory_learning_rate,
            **config.update_kwargs(),
        )
        # This parameter is exclusively the legacy RoboTTT output gate.
        self.memory.gate.requires_grad_(False)
        self.persistent = nn.Parameter(
            torch.randn(config.num_persistent_tokens, config.embed_dim) * 0.02
        )

    def prepare(self, x, valid, visual, state, reset_mask):
        selected = valid if self.config.memory_input == "all" else valid & visual
        if state is None:
            state = self.memory.initial_state(len(x))
        if reset_mask is not None:
            state = self.memory.reset_rows(state, reset_mask)
        return selected, state

    def prefix(self, x):
        p = self.persistent.to(x)[None].expand(len(x), -1, -1)
        mask = torch.ones(p.shape[:2], device=x.device, dtype=torch.bool)
        return p, mask

    def read(self, x, state, selected, position):
        safe = x.masked_fill(~selected[:, :, None], 0)
        return self.memory.read(safe, state, position=position).masked_fill(
            ~selected[:, :, None], 0
        )

    def write_read(self, x, state, selected, write_mask):
        position = state.position
        if self.config.memory_granularity == "observation":
            updated = self.memory.write(
                x,
                state,
                token_mask=selected,
                write_mask=write_mask,
                advance_position=False,
                advance_decision=False,
            )
            out = self.read(
                x,
                state if self.config.memory_update_protocol == "half_execution" else updated,
                selected,
                position,
            )
        else:
            updated, outputs = state, []
            for i in range(x.shape[1]):
                token = x[:, i : i + 1]
                mask = selected[:, i : i + 1]
                updated = self.memory.write(
                    token,
                    updated,
                    token_mask=mask,
                    write_mask=write_mask,
                    advance_position=False,
                    advance_decision=False,
                )
                outputs.append(self.read(token, updated, mask, position))
            out = torch.cat(outputs, dim=1)
        active = selected.any(1)
        if write_mask is not None:
            active = active & write_mask.to(device=x.device, dtype=torch.bool)
        updated = replace(
            updated,
            position=position + active.long(),
            decision_index=state.decision_index
            + (active.long() if self.config.memory_update_protocol == "observation" else 0),
        )
        return out, updated


class MemoryGate(nn.Module):
    """Nonlinear multiplicative fusion, near unit scale at initialization."""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.RMSNorm(dim)
        self.projection = nn.Linear(dim, dim)
        nn.init.normal_(self.projection.weight, std=0.001)
        nn.init.zeros_(self.projection.bias)

    def forward(self, attended, retrieved, selected):
        dtype = self.projection.weight.dtype
        gate = 2 * self.projection(self.norm(retrieved.to(dtype))).sigmoid()
        gate = torch.where(selected[:, :, None], gate, 1).to(attended.dtype)
        return attended * gate
