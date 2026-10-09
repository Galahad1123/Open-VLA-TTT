"""VL interface queries old memory; subsequent attention consumes fused interface."""

import torch

from ..observation_memory.action_head import ObservationActionHead


class T2MemActionHead(ObservationActionHead):
    interface_reads = True

    def connect_attention(self, i, block, tokens, query, context, context_mask, cache):
        kv = context if block.cross_attention else query
        valid = (
            context_mask
            if block.cross_attention
            else torch.ones(query.shape[:2], device=query.device, dtype=torch.bool)
        )
        if i > 0:
            prefix = cache.contexts[i - 1].to(kv)
            kv = torch.cat((kv, prefix), dim=1)
            valid = torch.cat(
                (valid, torch.ones(prefix.shape[:2], device=valid.device, dtype=torch.bool)), dim=1
            )
        attended = block.attention(query, kv, kv, key_padding_mask=~valid, need_weights=False)[0]
        return tokens + block.dropout(attended)
