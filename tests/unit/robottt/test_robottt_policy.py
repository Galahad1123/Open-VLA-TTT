import dataclasses

import numpy as np
import pytest
import torch
from lerobot.policies.factory import get_policy_class, make_policy_config

from lerobot_policy_vla_ttt.cli import make_pre_post_processors
from lerobot_policy_vla_ttt.models.robottt.modeling_robottt import RoboTTTPolicy


def batch():
    return {
        "observation.state": torch.rand(2, 8),
        "observation.images.image": torch.rand(2, 3, 16, 16),
        "observation.images.image2": torch.rand(2, 3, 16, 16),
        "action": torch.rand(2, 3, 7),
        "task": ["move", "pick"],
    }


def stats():
    return {
        key: {"min": -torch.ones(width), "max": torch.ones(width)}
        for key, width in [("action", 7), ("observation.state", 8)]
    }


def test_policy_factory_training_and_phase(config):
    assert get_policy_class("robottt") is RoboTTTPolicy
    assert make_policy_config("robottt", device="cpu").type == "robottt"
    policy = RoboTTTPolicy(config)
    pre, _ = make_pre_post_processors(config, dataset_stats=stats())
    loss_sum, count, memory = policy.forward_step(pre(batch()))
    (loss_sum / count).backward()
    assert memory[0].position.tolist() == [1, 1]
    assert all(p.grad is None for p in policy.model.embedder.parameters())
    assert policy.model.action_head.memories[0].fast.w1.grad.abs().sum() > 0
    assert all(not p.requires_grad for p in policy.model.action_head.blocks.parameters())
    joint = RoboTTTPolicy(
        dataclasses.replace(config, memory_training="joint", training_stage="stage2")
    )
    assert all(p.requires_grad for p in joint.model.parameters())


def test_checkpoint_and_evaluation_reset(config, tmp_path):
    from lerobot_policy_vla_ttt.evaluation.policy import Evo1Inference

    policy = RoboTTTPolicy(config).eval()
    pre, post = make_pre_post_processors(config, dataset_stats=stats())
    policy.predict_action_chunk(pre(batch()))
    policy.save_pretrained(tmp_path)
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)
    loaded = RoboTTTPolicy.from_pretrained(tmp_path, local_files_only=True)
    assert loaded.model.action_head.inference_state is None
    with pytest.raises(ValueError, match="memory_dim"):
        RoboTTTPolicy.from_pretrained(tmp_path, config=dataclasses.replace(config, memory_dim=8))
    service = Evo1Inference(tmp_path, "robottt", "libero", device="cpu", n_action_steps=2)
    obs = dict(
        state=np.zeros(8, np.float32),
        task="move",
        images={k: np.zeros((256, 256, 3), np.uint8) for k in ["image", "image2"]},
    )
    service.reset(5)
    first = service.infer(obs)
    assert first.shape == (7,) and np.isfinite(first).all()
    service.infer(obs)  # Queued execution does not update memory.
    assert service.policy.model.action_head.inference_state[0].position.item() == 1
    assert service.policy.model.action_head.inference_state[0].update_count.item() == 2
    service.infer(obs)
    assert service.policy.model.action_head.inference_state[0].position.item() == 2
    assert service.policy.model.action_head.inference_state[0].update_count.item() == 4
    service.reset(5)
    np.testing.assert_array_equal(first, service.infer(obs))


def test_disabled_memory_rejects_memory_only_training_but_allows_inference(config):
    policy = RoboTTTPolicy(dataclasses.replace(config, memory_enabled=False))
    with pytest.raises(ValueError, match="memory_only.*memory_enabled"):
        policy.forward_step(batch())
    actions = policy.predict_action_chunk(batch())
    assert actions.shape == (2, 3, 10)
    assert policy.model.action_head.inference_state is None
