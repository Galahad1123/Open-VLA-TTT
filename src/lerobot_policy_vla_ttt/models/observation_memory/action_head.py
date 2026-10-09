"""Shared tokenization/sampling; subclasses define memory-to-action connections."""

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from ...training.flow_matching import sample_flow
from ...ttt.headed_memory import HeadedMemory
from ..evo1_groot.action_head import GrootActionHead
from .interface import InterfaceLayer, ObservationCache


class ObservationActionHead(GrootActionHead):
    interface_reads = False

    def __init__(self, config):
        super().__init__(config)
        self.interface_tokens = nn.Parameter(
            torch.randn(1, config.num_interface_tokens, config.embed_dim) * 0.02
        )
        self.interfaces = nn.ModuleList(
            InterfaceLayer(config.embed_dim, config.num_heads, config.hidden_dim)
            for _ in range(config.interface_depth)
        )
        for layer in self.interfaces:
            layer.gate.data.fill_(config.memory_gate_init)
        self.memories = nn.ModuleList(
            HeadedMemory(
                config.embed_dim,
                config.memory_heads,
                config.memory_head_dim,
                config.memory_hidden_dim,
                learning_rate=config.memory_learning_rate,
                rope=self.interface_reads,
                **config.update_kwargs(),
            )
            for _ in self.interfaces
        )
        self.action_gates = (
            nn.ParameterList(
                nn.Parameter(torch.full((config.embed_dim,), config.memory_gate_init))
                for _ in self.interfaces
            )
            if not self.interface_reads
            else nn.ParameterList()
        )

    def initialize_from_baseline(self, baseline):
        source, target = baseline.state_dict(), self.state_dict()
        if not set(source) <= set(target) or any(
            target[k].shape != v.shape for k, v in source.items()
        ):
            raise ValueError("Backbone tensor mismatch")
        with torch.no_grad():
            for k, v in source.items():
                target[k].copy_(v)
        return {"copied": list(source), "new": sorted(set(target) - set(source))}

    def memory_parameters(self):
        yield self.interface_tokens
        for modules in (self.interfaces, self.memories, self.action_gates):
            yield from modules.parameters()

    def prepare_observation(self, features, valid, memory=None, reset_mask=None):
        if len(features) != len(self.interfaces):
            raise ValueError("Wrong number of VL feature layers")
        if memory is None:
            memory = tuple(m.initial_state(len(features[0])) for m in self.memories)
        if len(memory) != len(self.memories):
            raise ValueError("Memory layer count mismatch")
        if reset_mask is not None:
            memory = tuple(
                m.reset_rows(s, reset_mask) for m, s in zip(self.memories, memory, strict=True)
            )
        tokens = self.interface_tokens.expand(len(features[0]), -1, -1)
        summaries, contexts = [], []
        with sdpa_kernel(SDPBackend.MATH):
            for layer, m, s, vl in zip(
                self.interfaces, self.memories, memory, features, strict=True
            ):
                summary = layer.summarize(tokens, vl.to(tokens), valid)
                readout = (
                    m.read(summary, s)
                    if self.interface_reads and self.config.memory_enabled
                    else None
                )
                tokens = layer.finish(tokens, summary, readout)
                summaries.append(summary)
                contexts.append(tokens)
        return ObservationCache(tuple(summaries), tuple(contexts), tuple(memory))

    def commit(self, cache, write_mask=None):
        if not self.config.memory_enabled:
            return cache.memory
        return tuple(
            m.write(x, s, write_mask=write_mask)
            for m, x, s in zip(self.memories, cache.summaries, cache.memory, strict=True)
        )

    def connect_attention(self, i, block, tokens, query, context, context_mask, cache):
        raise NotImplementedError

    def velocity_with_memory(
        self, context, state, actions, time, cache, *, context_mask, embodiment_id
    ):
        if not self.config.memory_enabled:
            with sdpa_kernel(SDPBackend.MATH):
                return super().velocity(
                    context,
                    state,
                    actions,
                    time,
                    context_mask=context_mask,
                    embodiment_id=embodiment_id,
                )
        embedding = self.time_embedding(time)
        tokens = self.tokens(state, actions, embedding, embodiment_id)
        with sdpa_kernel(SDPBackend.MATH):
            for i, block in enumerate(self.blocks):
                scale, shift = block.modulation(embedding).chunk(2, -1)
                query = block.norm(tokens) * (1 + scale[:, None]) + shift[:, None]
                tokens = self.connect_attention(
                    i, block, tokens, query, context, context_mask, cache
                )
                tokens = tokens + block.dropout(block.ff(tokens))
        output = self.output_norm(tokens[:, -self.config.chunk_size :])
        ids = embodiment_id[:, None].expand(-1, self.config.chunk_size).reshape(-1)
        return self.decoder(output, ids)

    def training_prediction(
        self,
        context,
        state,
        actions,
        cache,
        *,
        action_mask=None,
        context_mask=None,
        embodiment_id=None,
    ):
        context, state, context_mask, ids = self._conditioning(
            context, state, context_mask, embodiment_id
        )
        mask = self._mask(action_mask, len(context), context.device)
        if actions.shape != mask.shape:
            raise ValueError("Invalid action shape")
        noisy, noise, time = sample_flow(actions.to(context), mask, self.config)
        pred = self.velocity_with_memory(
            context, state, noisy, time, cache, context_mask=context_mask, embodiment_id=ids
        )
        return pred.masked_fill(~mask, 0), noise

    @torch.no_grad()
    def sample(
        self, context, state, cache, *, action_mask=None, context_mask=None, embodiment_id=None
    ):
        context, state, context_mask, ids = self._conditioning(
            context, state, context_mask, embodiment_id
        )
        mask = self._mask(action_mask, len(context), context.device)
        actions = torch.randn(mask.shape, device=context.device).masked_fill(~mask, 0)
        steps = self.config.num_inference_timesteps
        for step in range(steps):
            time = torch.full((len(context),), step / steps, device=context.device)
            pred = self.velocity_with_memory(
                context,
                state,
                actions.to(context.dtype),
                time,
                cache,
                context_mask=context_mask,
                embodiment_id=ids,
            )
            actions = (actions + pred.float() / steps).masked_fill(~mask, 0)
        return actions
