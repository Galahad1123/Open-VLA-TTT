import torch

from lerobot_policy_vla_ttt.models.aqmem.configuration_aqmem import AQMemConfig
from lerobot_policy_vla_ttt.models.aqmem.modeling_aqmem import AQMemPolicy
from lerobot_policy_vla_ttt.training.sequence import HalfExecutionWindows, collate_windows
from lerobot_policy_vla_ttt.training.tbptt import train_sequence

from .assets import patch_assets
from .test_policy import batch, policy_config


def test_windows_remain_decision_aligned_with_odd_length():
    raw = [{"action": torch.zeros(3, 4), "observation.state": torch.ones(4) * i} for i in range(19)]
    windows = HalfExecutionWindows(raw, [(0, 9), (9, 19)], length=3, execution=4)
    assert [int(windows[i]["frames"][0]["observation.state"][0]) for i in range(len(windows))] == [
        0,
        4,
        8,
        9,
        13,
        17,
    ]
    result = collate_windows([windows[0], windows[2]])
    assert result["supervision_mask"].tolist() == [[True, False, True], [True, False, False]]
    assert windows[0]["frames"][1]["observation.state"][0] == 2


def test_context_only_segment_backwards_and_row_reset(monkeypatch):
    patch_assets(monkeypatch)
    p = AQMemPolicy(policy_config(AQMemConfig))
    frames = [batch() for _ in range(4)]
    supervised = torch.tensor([[1, 0, 1, 0], [1, 0, 1, 0]]).bool()
    resets = torch.zeros_like(supervised)
    resets[1, 2] = True
    optimizer = torch.optim.Adam(p.parameters(), lr=1e-3)
    result, memory = train_sequence(
        p, frames, optimizer, segment_length=1, supervision_mask=supervised, reset_mask=resets
    )
    assert result["segments"] == 4
    assert memory[0].update_count.tolist() == [4, 2]
    assert all(w.grad_fn is None for s in memory for w in s.weights)
