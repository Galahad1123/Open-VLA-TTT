"""Action queries old memory, with observation-only K/V writes."""

from ...training.episode_parallel.diagnostics import record_read
from ..observation_memory.action_head import ObservationActionHead


class AQMemActionHead(ObservationActionHead):
    def connect_attention(self, i, block, tokens, query, context, context_mask, cache):
        kv = context if block.cross_attention else query
        padding = ~context_mask if block.cross_attention else None
        attended = block.attention(query, kv, kv, key_padding_mask=padding, need_weights=False)[0]
        tokens = tokens + block.dropout(attended)
        # Evaluation-only: writes and fast-state updates remain unchanged.
        if getattr(self, "disable_memory_output", False):
            return tokens
        residual = self.action_gates[i].tanh() * self.memories[i].read(attended, cache.memory[i])
        record_read(i, tokens, residual)
        return tokens + residual
