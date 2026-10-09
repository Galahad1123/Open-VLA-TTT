from dataclasses import replace

import numpy as np
import pytest
import torch
from lerobot.policies.factory import get_policy_class, make_policy_config

from lerobot_policy_vla_ttt.cli import make_pre_post_processors
from lerobot_policy_vla_ttt.evaluation.policy import Evo1Inference
from lerobot_policy_vla_ttt.training.tbptt import train_sequence

from .conftest import batch, stats


@pytest.mark.parametrize("rule", ["gradient_descent", "momentum_decay"])
def test_policy_processors_and_frozen_dit_outer_gradient(setup, rule):
    cfg = replace(setup.config, memory_update_rule=rule)
    assert get_policy_class(cfg.type) is setup.policy_class
    assert make_policy_config(cfg.type, device="cpu").type == cfg.type
    p = setup.policy_class(cfg)
    pre, _ = make_pre_post_processors(cfg, dataset_stats=stats())
    numerator, count, state = p.forward_step(pre(batch()))
    (numerator / count).backward()
    assert count > 0 and numerator.isfinite()
    assert all(x.grad is None for x in p.model.action_head.parameters())
    assert all(x.grad is None for x in p.model.embedder.model.parameters())
    for connection in p.model.embedder.connections.values():
        assert connection.memory.fast.w1.grad.abs().sum() > 0
    assert state[0].update_count.tolist() == [1, 1]
    assert p.inference_state is None


def test_checkpoint_identity_strict_semantics_and_episode_state(setup, tmp_path):
    cfg = replace(setup.config, memory_update_rule="momentum_decay")
    p = setup.policy_class(cfg)
    pre, post = make_pre_post_processors(cfg, dataset_stats=stats())
    p.predict_action_chunk(pre(batch()))
    p.save_pretrained(tmp_path)
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)
    loaded = setup.policy_class.from_pretrained(tmp_path, local_files_only=True)
    assert loaded.inference_state is None
    assert loaded.config.memory_layers == cfg.memory_layers
    for name in ("memory_input", "memory_update_rule", "memory_granularity"):
        value = {
            "memory_input": "visual",
            "memory_update_rule": "gradient_descent",
            "memory_granularity": "token",
        }[name]
        with pytest.raises(ValueError, match=name):
            setup.policy_class.from_pretrained(tmp_path, config=replace(cfg, **{name: value}))
    with pytest.raises(ValueError, match="strict"):
        setup.policy_class.from_pretrained(tmp_path, strict=False)
    assert not any("auxiliary" in k or "inference_state" in k for k in loaded.state_dict())
    service = Evo1Inference(tmp_path, cfg.type, "libero", device="cpu", n_action_steps=2)
    obs = dict(
        state=np.zeros(8, np.float32),
        task="move",
        images={k: np.zeros((256, 256, 3), np.uint8) for k in ["image", "image2"]},
    )
    before = {k: v.clone() for k, v in service.policy.state_dict().items()}
    service.reset(5)
    first = service.infer(obs)
    service.infer(obs)
    assert service.policy.inference_state[0].update_count.item() == 1
    service.infer(obs)
    assert service.policy.inference_state[0].update_count.item() == 2
    service.reset(5)
    np.testing.assert_array_equal(first, service.infer(obs))
    for k, v in service.policy.state_dict().items():
        torch.testing.assert_close(v, before[k])
    assert all(p.grad is None for p in service.policy.parameters())


def test_inference_updates_independent_of_denoising_and_failure_atomic(setup, monkeypatch):
    p = setup.policy_class(setup.config)
    b = batch()
    p.predict_action_chunk(b)
    before = p.inference_state
    p.config.num_inference_timesteps = 5
    p.predict_action_chunk(b)
    assert p.inference_state[0].update_count.tolist() == [2, 2]
    before = p.inference_state

    def fail(*args, **kwargs):
        raise RuntimeError("sampling failed")

    monkeypatch.setattr(p.model.action_head, "get_action", fail)
    with pytest.raises(RuntimeError, match="sampling failed"):
        p.predict_action_chunk(b)
    assert p.inference_state is before
    with torch.inference_mode(), pytest.raises(RuntimeError, match="inference_mode"):
        p.predict_action_chunk(b)


def test_action_labels_do_not_enter_memory_and_context_only_supported(setup):
    p = setup.policy_class(setup.config)
    b = batch()
    _, _, a = p.forward_step(b)
    changed = dict(b, action=b["action"] + 100)
    _, _, c = p.forward_step(changed)
    for sa, sc in zip(a, c, strict=True):
        torch.testing.assert_close(sa.weights, sc.weights)
    context = {k: v for k, v in b.items() if k != "action"}
    loss, count, state = p.forward_step(context, supervision_mask=torch.tensor([False, False]))
    assert loss == 0 and count == 0
    assert state[0].update_count.tolist() == [1, 1]


def test_tbptt_real_backward_padding_reset_and_joint_stage(setup):
    cfg = replace(
        setup.config,
        memory_update_rule="momentum_decay",
        memory_training="joint",
        training_stage="stage2",
    )
    p = setup.policy_class(cfg)
    frames = [batch() for _ in range(3)]
    optimizer = torch.optim.Adam(p.get_optim_params(), lr=1e-4)
    metrics, state = train_sequence(
        p,
        frames,
        optimizer,
        segment_length=2,
        valid_mask=torch.tensor([[True, True, True], [True, True, False]]),
        reset_mask=torch.tensor([[False, False, True], [False, False, False]]),
    )
    assert metrics["segments"] == 2 and np.isfinite(metrics["loss"])
    assert state[0].update_count.tolist() == [1, 2]
    assert all(w.grad_fn is None for w in (*state[0].weights, *state[0].auxiliary))
    assert p.model.embedder.model.vision_tower.weight.grad.abs().sum() > 0
    assert any(x.grad is not None for x in p.model.action_head.parameters())


def test_half_execution_clock_reads_before_write_and_updates_midpoint(setup):
    from dataclasses import replace

    from .conftest import batch

    cfg = replace(
        setup.config,
        memory_update_protocol="half_execution",
        chunk_size=6,
        n_action_steps=4,
        memory_update_rule="t2mem_adaptive",
    )
    policy = setup.policy_class(cfg).eval()
    b = batch()
    initial = {k: v.clone() for k, v in policy.state_dict().items()}
    first = policy.select_action(b, env_step=0)
    torch.testing.assert_close(first, policy.select_action(b, env_step=0))
    policy.select_action(b, env_step=1)
    policy.select_action(b, env_step=2)
    assert policy.inference_state[0].update_count.tolist() == [2, 2]
    assert policy.inference_state[0].decision_index.tolist() == [1, 1]
    for k, v in policy.state_dict().items():
        torch.testing.assert_close(v, initial[k])


def test_half_execution_disabled_memory_is_valid_baseline(setup):
    cfg = replace(
        setup.config,
        memory_update_protocol="half_execution",
        memory_enabled=False,
        memory_training="joint",
        chunk_size=6,
        n_action_steps=4,
    )
    policy = setup.policy_class(cfg).eval()
    for step in range(5):
        assert policy.select_action(batch(), env_step=step).shape[0] == 2
    assert policy.inference_state is None
