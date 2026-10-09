"""MAG: attention and neural memory process the same input, then gated fusion."""

from ..titans.attention import attend, finish
from ..titans.connection import MemoryConnection, MemoryGate


class MAGConnection(MemoryConnection):
    def __init__(self, config):
        super().__init__(config)
        self.gate = MemoryGate(config.embed_dim)

    def forward(self, layer, x, valid, visual, rotary, state, write_mask, reset_mask):
        selected, state = self.prepare(x, valid, visual, state, reset_mask)
        normalized = layer.input_layernorm(x)
        prefix, prefix_valid = self.prefix(x)
        y = attend(layer, normalized, valid, rotary, prefix=prefix, prefix_valid=prefix_valid)
        retrieved, updated = self.write_read(normalized, state, selected, write_mask)
        return finish(layer, x, self.gate(y, retrieved, selected), valid), updated
