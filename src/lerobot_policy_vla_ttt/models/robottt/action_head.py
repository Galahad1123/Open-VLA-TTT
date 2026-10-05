"""RoboTTT connection: attention residual -> memory residual -> FFN residual.

Reuses GR00T-style components, not community hooks (which duplicate residuals).
Fast state is explicit in training and episode-local in sampling.
"""

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from ...training.flow_matching import sample_flow
from ...ttt.memory import AssociativeMemory
from ..evo1_groot.action_head import GrootActionHead


class RoboTTTActionHead(GrootActionHead):
    def __init__(self, config):
        super().__init__(config)
        self.memories = nn.ModuleList(
            AssociativeMemory(
                config.embed_dim,
                config.memory_dim,
                config.memory_hidden_dim,
                config.memory_learning_rate,
                config.memory_gate_init,
            )
            for _ in self.blocks
        )
        self.reset()

    def reset(self):
        self.inference_state = None

    def initialize_from_baseline(self, baseline):
        """Copy every backbone tensor, allowing only newly introduced registers."""
        source, target = baseline.state_dict(), self.state_dict()
        copied, new = [], []
        for key, value in target.items():
            if key.startswith("memories."):
                new.append(key)
            elif key == "tokens.registers" and source[key].shape[1] == 0:
                new.append(key)
            elif key not in source or source[key].shape != value.shape:
                raise ValueError(f"Baseline tensor mismatch: {key}")
            else:
                copied.append(key)
        if set(source) - set(target):
            raise ValueError("Unexpected baseline tensors")
        with torch.no_grad():
            for key in copied:
                target[key].copy_(source[key])
        return {"copied": copied, "new": new}

    def velocity_step(
        self,
        context,
        state,
        actions,
        time,
        *,
        context_mask,
        embodiment_id,
        memory=None,
        write_mask=None,
        reset_mask=None,
        memory_enabled=None,
    ):
        enabled = self.config.memory_enabled if memory_enabled is None else memory_enabled
        if memory is not None and len(memory) != len(self.memories):
            raise ValueError("Memory layer count mismatch")
        embedding = self.time_embedding(time)
        tokens = self.tokens(state, actions, embedding, embodiment_id)
        if not enabled:
            tokens = torch.cat((tokens[:, :1], tokens[:, -self.config.chunk_size :]), dim=1)
        updated = []
        # Math attention supports the higher-order derivatives needed by inner GD.
        with sdpa_kernel(SDPBackend.MATH):
            for i, block in enumerate(self.blocks):
                scale, shift = block.modulation(embedding).chunk(2, dim=-1)
                query = block.norm(tokens) * (1 + scale[:, None]) + shift[:, None]
                kv = context if block.cross_attention else query
                padding = ~context_mask if block.cross_attention else None
                attended = block.attention(
                    query, kv, kv, key_padding_mask=padding, need_weights=False
                )[0]
                tokens = tokens + block.dropout(attended)
                if enabled:
                    layer_state = None if memory is None else memory[i]
                    if reset_mask is not None and layer_state is not None:
                        layer_state = self.memories[i].reset_rows(layer_state, reset_mask)
                    tokens, layer_state = self.memories[i](
                        tokens, layer_state, write_mask=write_mask
                    )
                    updated.append(layer_state)
                tokens = tokens + block.dropout(block.ff(tokens))
        actions = self.output_norm(tokens[:, -self.config.chunk_size :])
        ids = embodiment_id[:, None].expand(-1, self.config.chunk_size).reshape(-1)
        return self.decoder(actions, ids), tuple(updated) if enabled else None

    def training_step(
        self,
        context,
        state,
        actions_gt,
        *,
        action_mask=None,
        embodiment_id=None,
        context_mask=None,
        memory=None,
        write_mask=None,
        reset_mask=None,
    ):
        context, state, context_mask, ids = self._conditioning(
            context, state, context_mask, embodiment_id
        )
        mask = self._mask(action_mask, len(context), context.device)
        if actions_gt.shape != mask.shape:
            raise ValueError("actions_gt must match the configured horizon and action width")
        actions = actions_gt.to(context)
        noisy, noise, time = sample_flow(actions, mask, self.config)
        pred, updated = self.velocity_step(
            context,
            state,
            noisy,
            time,
            context_mask=context_mask,
            embodiment_id=ids,
            memory=memory,
            write_mask=write_mask,
            reset_mask=reset_mask,
        )
        return pred.masked_fill(~mask, 0), noise, updated

    def forward(
        self, context, state, actions_gt, action_mask=None, embodiment_id=None, context_mask=None
    ):
        pred, noise, _ = self.training_step(
            context,
            state,
            actions_gt,
            action_mask=action_mask,
            embodiment_id=embodiment_id,
            context_mask=context_mask,
        )
        return pred.flatten(1), noise

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
            v is not None
            for v in (inference_delay, prev_chunk_left_over, execution_horizon, rtc_processor)
        ):
            raise ValueError("RoboTTT does not support RTC")
        context, state, context_mask, ids = self._conditioning(
            context, state, context_mask, embodiment_id
        )
        mask = self._mask(action_mask, len(context), context.device)
        actions = torch.randn(mask.shape, device=context.device).masked_fill(~mask, 0)
        memory = self.inference_state
        steps = self.config.num_inference_timesteps
        for step in range(steps):
            time = torch.full((len(context),), step / steps, device=context.device)
            pred, memory = self.velocity_step(
                context,
                state,
                actions.to(context.dtype),
                time,
                context_mask=context_mask,
                embodiment_id=ids,
                memory=memory,
            )
            actions = (actions + pred.float() / steps).masked_fill(~mask, 0)
        # Commit only a complete chunk; failed sampling leaves previous state intact.
        self.inference_state = memory
        return actions.flatten(1)
