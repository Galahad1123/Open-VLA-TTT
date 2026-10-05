"""State/action token encoding, with no cross-call memory."""

import math

import torch
from torch import nn

from ...evo1.flow_matching import CategorySpecificMLP


class TimeEmbedding(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.register_buffer(
            "frequencies",
            torch.exp(
                -math.log(10000) * torch.arange(width // 2, dtype=torch.float32) / (width // 2)
            ),
        )
        self.mlp = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))

    def forward(self, time):
        phase = time.float()[:, None] * 1000 * self.frequencies.float()[None]
        return self.mlp(
            torch.cat((phase.sin(), phase.cos()), dim=-1).to(dtype=self.mlp[0].weight.dtype)
        )


class ActionTokens(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.embed_dim
        self.state_encoder = CategorySpecificMLP(
            config.max_state_dim, config.state_hidden_dim, d, config.num_categories
        )
        self.action_encoder = CategorySpecificMLP(
            config.max_action_dim, config.hidden_dim, d, config.num_categories
        )
        self.action_time = nn.Sequential(nn.Linear(2 * d, d), nn.SiLU(), nn.Linear(d, d))
        self.action_positions = nn.Parameter(torch.empty(1, config.chunk_size, d))
        nn.init.normal_(self.action_positions, std=0.02)
        self.registers = nn.Parameter(torch.empty(1, config.num_register_tokens, d))
        nn.init.normal_(self.registers, std=0.02)

    def forward(self, state, actions, time_embedding, ids):
        batch, horizon = actions.shape[:2]
        action_ids = ids[:, None].expand(batch, horizon).reshape(-1)
        encoded = self.action_encoder(actions, action_ids)
        time = time_embedding[:, None].expand(-1, horizon, -1)
        encoded = self.action_time(torch.cat((encoded, time), dim=-1)) + self.action_positions
        return torch.cat(
            (
                self.state_encoder(state, ids)[:, None],
                self.registers.expand(batch, -1, -1),
                encoded,
            ),
            dim=1,
        )
