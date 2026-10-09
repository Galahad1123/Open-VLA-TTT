"""MAL: memory output feeds attention. No extra pre-attention residual adapter."""

import torch

from ..titans.attention import attend, finish
from ..titans.connection import MemoryConnection


class MALConnection(MemoryConnection):
    def forward(self, layer, x, valid, visual, rotary, state, write_mask, reset_mask):
        selected, state = self.prepare(x, valid, visual, state, reset_mask)
        normalized = layer.input_layernorm(x)
        retrieved, updated = self.write_read(normalized, state, selected, write_mask)
        # Visual-only ablation applies the memory transformation only at visual positions.
        mixed = torch.where(selected[:, :, None], retrieved.to(normalized), normalized)
        prefix, prefix_valid = self.prefix(x)
        y = attend(layer, mixed, valid, rotary, prefix=prefix, prefix_valid=prefix_valid)
        return finish(layer, x, y, valid), updated
