from dataclasses import replace

import pytest
import torch

from lerobot_policy_vla_ttt.training.episode_parallel.config import Recipe
from lerobot_policy_vla_ttt.training.episode_parallel.data import Event
from lerobot_policy_vla_ttt.training.episode_parallel.microbatch import backward_microbatch
from lerobot_policy_vla_ttt.training.train_episode_parallel import resolve_config

from .conftest import batch


def mag(setup):
    if setup.config.type != "mag":
        pytest.skip("MAG batching only")
    return replace(
        setup.config,
        memory_training="joint",
        memory_update_protocol="half_execution",
        apply_training_stage_defaults=False,
        finetune_vlm=False,
        finetune_language_model=False,
        finetune_vision_model=False,
        finetune_action_head=True,
        dropout=0.0,
        memory_update_rule="t2mem_adaptive",
    )


def test_mag_config_and_row_losses(setup):
    cfg = mag(setup)
    recipe = Recipe(
        horizon=3, execution=2, freeze_vlm=True, episode_microbatch_size=2, training_mode="joint"
    )
    assert resolve_config(cfg, "mag", recipe, device="cpu").type == "mag"
    p = setup.policy_class(cfg)
    b = batch()
    mask = torch.tensor([True, False])
    write = torch.tensor([True, False])
    torch.manual_seed(3)
    num, count, state = p.forward_step(
        b, write_mask=write, supervision_mask=mask, loss_reduction="none"
    )
    assert num.shape == count.shape == (2,)
    assert count.tolist() == [21, 0] and num[1] == 0
    torch.manual_seed(3)
    total, n, _ = p.forward_step(b, write_mask=write, supervision_mask=mask)
    torch.testing.assert_close(total, num.sum())
    assert n == count.sum()
    zero, c, _ = p.forward_step(
        b, supervision_mask=torch.zeros(2, dtype=torch.bool), loss_reduction="none"
    )
    assert zero.shape == c.shape == (2,) and not c.any()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("checkpointing,offload", [(False, False), (True, False), (True, True)])
def test_mag_microbatch_matches_serial_with_padding_and_tbptt(
    setup, checkpointing, offload, device
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    cfg = replace(mag(setup), enable_gradient_checkpointing=checkpointing, device=device)
    torch.manual_seed(8)
    p = setup.policy_class(cfg).to(device)
    q = setup.policy_class(cfg).to(device)
    q.load_state_dict(p.state_dict())
    streams = []
    for row, length in enumerate([5, 3]):
        events = []
        for i in range(length):
            b = batch()
            frame = {
                k: (v[row : row + 1] if isinstance(v, torch.Tensor) else [v[row]])
                for k, v in b.items()
            }
            supervised = i % 2 == 0
            if not supervised:
                frame.pop("action")
            events.append(Event(frame, supervised))
        streams.append(events)
    recipe = Recipe(
        global_batch_size=2,
        tbptt_events=2,
        activation_offload=offload,
        episode_microbatch_size=2,
        freeze_vlm=True,
    )

    def preprocess(frame):
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in frame.items()}

    counts = [3, 2]
    serial = []
    metrics = []
    for row in range(2):
        m, s = backward_microbatch(
            p,
            [iter(streams[row])],
            decisions=[counts[row]],
            seeds=[100 + row],
            recipe=replace(recipe, activation_offload=False),
            preprocess=preprocess,
            backward=lambda x: x.backward(),
        )
        serial.append(s)
        metrics.append(m)
    m, state = backward_microbatch(
        q,
        [iter(s) for s in streams],
        preprocess=preprocess,
        decisions=counts,
        seeds=[100, 101],
        recipe=recipe,
        backward=lambda x: x.backward(),
    )
    assert m["loss_sum"] == pytest.approx(sum(v["loss_sum"] for v in metrics), rel=1e-4)
    assert m["events"] == 8 and m["decisions"] == 5
    for (name, a), (_, b) in zip(p.named_parameters(), q.named_parameters(), strict=True):
        assert (a.grad is None) == (b.grad is None), name
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, rtol=5e-3, atol=2e-5, msg=name)
    assert all(v.grad is None for v in q.model.embedder.model.parameters())
    assert any(
        v.grad is not None and v.grad.abs().sum() > 0
        for v in q.model.embedder.connections.parameters()
    )
    for layer, s in enumerate(state):
        for row in range(2):
            for field in ["position", "decision_index", "update_count"]:
                assert getattr(s, field)[row] == getattr(serial[row][layer], field)[0]
            for a, b in zip(s.weights, serial[row][layer].weights, strict=True):
                torch.testing.assert_close(a[row], b[0], rtol=5e-3, atol=2e-5)


def test_mag_optimizer_partition_and_unfrozen_backbone_rejected(setup):
    from lerobot_policy_vla_ttt.training.episode_parallel.optim import parameter_groups

    cfg = mag(setup)
    p = setup.policy_class(cfg)
    recipe = Recipe(training_mode="joint", freeze_vlm=True, policy_lr=1e-5, memory_lr=1e-4)
    groups = parameter_groups(p, "joint", recipe)
    names = [n for g in groups for n in g["names"]]
    assert len(names) == len(set(names))
    assert all(n.startswith(("model.embedder.connections.", "model.action_head.")) for n in names)
    for g in groups:
        assert g["lr"] == (1e-5 if g["category"] == "action_expert" else 1e-4)
    p.model.embedder.model.vision_tower.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen VLM"):
        backward_microbatch(
            p, [iter([])], decisions=[1], seeds=[1], recipe=recipe, backward=lambda x: x.backward()
        )
