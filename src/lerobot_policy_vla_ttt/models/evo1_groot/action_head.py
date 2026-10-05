"""Continuous flow matching with alternating cross/self attention.

Structure reference: NVIDIA Isaac-GR00T (Apache-2.0), commit
51d4c89f72fda44cbf77285c6a8114b52676b8a1, modules/dit.py.
This small PyTorch implementation is not weight-compatible with GR00T.
"""

import torch
from torch import nn

from ...evo1.flow_matching import CategorySpecificMLP
from .tokens import ActionTokens, TimeEmbedding


class DiTBlock(nn.Module):
    def __init__(self, config, *, cross_attention):
        super().__init__()
        self.cross_attention = cross_attention
        d = config.embed_dim
        self.norm = nn.LayerNorm(d, elementwise_affine=False)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        self.attention = nn.MultiheadAttention(
            d, config.num_heads, dropout=config.dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, d),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, tokens, context, time_embedding, context_mask):
        scale, shift = self.modulation(time_embedding).chunk(2, dim=-1)
        query = self.norm(tokens) * (1 + scale[:, None]) + shift[:, None]
        kv = context if self.cross_attention else query
        padding = ~context_mask if self.cross_attention else None
        attended = self.attention(query, kv, kv, key_padding_mask=padding, need_weights=False)[0]
        tokens = tokens + self.dropout(attended)
        return tokens + self.dropout(self.ff(tokens))


class GrootActionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokens = ActionTokens(config)
        self.time_embedding = TimeEmbedding(config.embed_dim)
        self.blocks = nn.ModuleList(
            DiTBlock(config, cross_attention=(i % 2 == 0)) for i in range(config.num_layers)
        )
        self.output_norm = nn.LayerNorm(config.embed_dim)
        self.decoder = CategorySpecificMLP(
            config.embed_dim, config.hidden_dim, config.max_action_dim, config.num_categories
        )

    def _conditioning(self, context, state, context_mask, embodiment_id):
        batch = context.shape[0]
        if context.ndim != 3 or context.shape[-1] != self.config.embed_dim:
            raise ValueError("context must have shape (batch, tokens, embed_dim)")
        if state is None or state.shape != (batch, self.config.max_state_dim):
            raise ValueError("state must have shape (batch, max_state_dim)")
        if context_mask is None:
            context_mask = torch.ones(context.shape[:2], device=context.device, dtype=torch.bool)
        else:
            context_mask = context_mask.to(device=context.device, dtype=torch.bool)
        if context_mask.shape != context.shape[:2] or not context_mask.any(dim=1).all():
            raise ValueError("context must have at least one valid token per sample")
        if embodiment_id is None:
            embodiment_id = torch.full(
                (batch,), self.config.default_embodiment_id, device=context.device, dtype=torch.long
            )
        else:
            embodiment_id = embodiment_id.to(device=context.device, dtype=torch.long).reshape(-1)
        if (
            embodiment_id.shape != (batch,)
            or (embodiment_id < 0).any()
            or (embodiment_id >= self.config.num_categories).any()
        ):
            raise ValueError("embodiment_id outside configured category range")
        dtype = next(self.parameters()).dtype
        # Mask before projecting keys/values, so padded token contents cannot leak.
        context = context.to(dtype=dtype).masked_fill(~context_mask[:, :, None], 0)
        return context, state.to(dtype=dtype), context_mask, embodiment_id

    def _mask(self, action_mask, batch, device):
        shape = (batch, self.config.chunk_size, self.config.max_action_dim)
        if action_mask is None:
            return torch.ones(shape, device=device, dtype=torch.bool)
        action_mask = action_mask.to(device=device, dtype=torch.bool)
        if action_mask.shape == (batch, self.config.max_action_dim):
            action_mask = action_mask[:, None].expand(shape)
        if action_mask.shape != shape:
            raise ValueError(f"action_mask must match {shape}")
        return action_mask

    def velocity(self, context, state, actions, time, *, context_mask, embodiment_id):
        time_embedding = self.time_embedding(time)
        tokens = self.tokens(state, actions, time_embedding, embodiment_id)
        for block in self.blocks:
            tokens = block(tokens, context, time_embedding, context_mask)
        actions = self.output_norm(tokens[:, -self.config.chunk_size :])
        ids = embodiment_id[:, None].expand(-1, self.config.chunk_size).reshape(-1)
        return self.decoder(actions, ids)

    def forward(
        self, context, state, actions_gt, action_mask=None, embodiment_id=None, context_mask=None
    ):
        context, state, context_mask, ids = self._conditioning(
            context, state, context_mask, embodiment_id
        )
        batch = context.shape[0]
        mask = self._mask(action_mask, batch, context.device)
        if actions_gt.shape != mask.shape:
            raise ValueError("actions_gt must have shape (batch, chunk_size, max_action_dim)")
        actions = actions_gt.to(device=context.device, dtype=context.dtype).masked_fill(~mask, 0)
        noise = torch.randn_like(actions).masked_fill(~mask, 0)
        alpha = torch.tensor(self.config.flow_beta_alpha, device=context.device)
        beta = torch.tensor(self.config.flow_beta_beta, device=context.device)
        time = self.config.flow_time_scale * (
            1 - torch.distributions.Beta(alpha, beta).sample((batch,))
        )
        weight = time[:, None, None].to(actions.dtype)
        noisy_actions = (1 - weight) * noise + weight * actions
        pred = self.velocity(
            context, state, noisy_actions, time, context_mask=context_mask, embodiment_id=ids
        )
        return pred.masked_fill(~mask, 0).reshape(batch, -1), noise

    @torch.no_grad()
    def get_action(
        self,
        context,
        state,
        action_mask=None,
        embodiment_id=None,
        context_mask=None,
        inference_delay=None,
        prev_chunk_left_over=None,
        execution_horizon=None,
        rtc_processor=None,
    ):
        if any(
            value is not None
            for value in (inference_delay, prev_chunk_left_over, execution_horizon, rtc_processor)
        ):
            raise ValueError("evo1_groot does not support RTC sampling")
        context, state, context_mask, ids = self._conditioning(
            context, state, context_mask, embodiment_id
        )
        batch = context.shape[0]
        mask = self._mask(action_mask, batch, context.device)
        # Keep the integration accumulator in fp32 even under autocast.
        actions = torch.randn(mask.shape, device=context.device).masked_fill(~mask, 0)
        steps = self.config.num_inference_timesteps
        for step in range(steps):
            time = torch.full((batch,), step / steps, device=context.device)
            pred = self.velocity(
                context,
                state,
                actions.to(context.dtype),
                time,
                context_mask=context_mask,
                embodiment_id=ids,
            )
            actions = (actions + pred.float() / steps).masked_fill(~mask, 0)
        return actions.reshape(batch, -1)
