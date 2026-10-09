"""The shared updater also works through the existing DiT memory lifecycle."""

from dataclasses import replace

import torch

from lerobot_policy_vla_ttt.models.robottt.modeling_robottt import RoboTTTPolicy


def test_momentum_training_reload_and_denoising_state(config, tmp_path):
    cfg = replace(config, memory_update_rule="momentum_decay")
    policy = RoboTTTPolicy(cfg)
    batch = {
        "observation.state": torch.rand(2, 8),
        "observation.images.image": torch.rand(2, 3, 16, 16),
        "observation.images.image2": torch.rand(2, 3, 16, 16),
        "action": torch.rand(2, 3, 7),
        "task": ["move", "pick"],
    }
    total, count, state = policy.forward_step(batch)
    (total / count).backward()
    updater = policy.model.action_head.memories[0].update
    assert updater.raw_decay.grad.isfinite()
    assert updater.coefficients.weight.grad.abs().sum() > 0
    assert any(v.abs().sum() > 0 for v in state[0].auxiliary)
    policy.save_pretrained(tmp_path)
    restored = RoboTTTPolicy.from_pretrained(tmp_path, local_files_only=True)
    assert restored.config.memory_update_rule == "momentum_decay"
    assert restored.model.action_head.inference_state is None
    actions = restored.predict_action_chunk(batch)
    assert actions.isfinite().all()
    state = restored.model.action_head.inference_state[0]
    assert state.decision_index.tolist() == [1, 1]
    assert state.update_count.tolist() == [cfg.num_inference_timesteps] * 2
    assert all(not v.requires_grad for v in state.auxiliary)
    restored.reset()
    assert restored.model.action_head.inference_state is None
