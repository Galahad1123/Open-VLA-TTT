"""MAC: old-state retrieval as context -> attention -> write -> read/gate.

Topology: Titans NeurIPS 2025 section 3.1. Uses prefix K/V rather than
materializing prefix query outputs; only current observation tokens are written.
"""

import torch

from ..titans.attention import attend, finish
from ..titans.connection import MemoryConnection, MemoryGate


class MACConnection(MemoryConnection):
    def __init__(self, config):
        super().__init__(config)
        self.gate = MemoryGate(config.embed_dim)

    def forward(self, layer, x, valid, visual, rotary, state, write_mask, reset_mask):
        selected, state = self.prepare(x, valid, visual, state, reset_mask)
        normalized = layer.input_layernorm(x)
        history = self.read(normalized, state, selected, state.position)
        prefix, prefix_valid = self.prefix(x)
        prefix = torch.cat((prefix, history), dim=1)
        prefix_valid = torch.cat((prefix_valid, selected), dim=1)
        y = attend(layer, normalized, valid, rotary, prefix=prefix, prefix_valid=prefix_valid)
        retrieved, updated = self.write_read(y, state, selected, write_mask)
        return finish(layer, x, self.gate(y, retrieved, selected), valid), updated
