from types import SimpleNamespace
from unittest.mock import Mock

import torch
from lerobot_policy_vla_ttt.models.aqmem.action_head import AQMemActionHead


def test_output_ablation_preserves_attention_and_skips_memory_read():
    tokens = torch.ones(1, 2, 3)
    attended = torch.full_like(tokens, 2)
    block = SimpleNamespace(
        cross_attention=False, attention=Mock(return_value=(attended, None)), dropout=lambda x: x
    )
    memory = SimpleNamespace(read=Mock(return_value=torch.full_like(tokens, 4)))
    head = SimpleNamespace(
        action_gates=[torch.tensor(0.5)], memories=[memory], disable_memory_output=True
    )
    cache = SimpleNamespace(memory=[object()])
    result = AQMemActionHead.connect_attention(head, 0, block, tokens, tokens, tokens, None, cache)
    assert torch.equal(result, tokens + attended)
    memory.read.assert_not_called()
    head.disable_memory_output = False
    result = AQMemActionHead.connect_attention(head, 0, block, tokens, tokens, tokens, None, cache)
    assert torch.equal(result, tokens + attended + torch.tensor(0.5).tanh() * 4)
    memory.read.assert_called_once()
