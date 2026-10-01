import importlib

import numpy as np
import pytest


def test_robomme_frame_excludes_demo_and_privileged_labels():
    prepare = importlib.import_module("lerobot_policy_vla_ttt.baselines.prepare")
    frame = {
        "image": np.zeros((3, 256, 256), np.uint8),
        "wrist_image": np.zeros((3, 256, 256), np.uint8),
        "state": np.arange(8, dtype=np.float32),
        "actions": np.arange(8, dtype=np.float32),
        "is_demo": np.array([False]),
        "task": "watch the video",
        "grounded_subgoal": "the hidden answer",
    }
    result = prepare.control_frame(frame)
    assert set(result) == {
        "observation.images.image",
        "observation.images.image2",
        "observation.state",
        "action",
        "task",
    }
    assert result["observation.images.image"].shape == (256, 256, 3)
    np.testing.assert_array_equal(result["action"], np.arange(8))
    assert prepare.control_frame({**frame, "is_demo": np.array([True])}) is None
    with pytest.raises(ValueError):
        prepare.control_frame({**frame, "actions": np.full(8, np.nan)})


def test_training_config_preserves_isolation_and_stage_defaults(tmp_path):
    from lerobot.configs.default import DatasetConfig
    from lerobot.policies.evo1.configuration_evo1 import Evo1Config

    training = importlib.import_module("lerobot_policy_vla_ttt.baselines.train")
    cfg = training.make_config(
        Evo1Config(device="cpu"),
        DatasetConfig("local/example", root=str(tmp_path)),
        tmp_path / "output",
        steps=2,
        batch_size=1,
        seed=123,
        stage="stage1",
    )
    assert cfg.env is None and cfg.env_eval_freq == 0
    assert cfg.policy.finetune_vlm is False
    assert cfg.policy.finetune_action_head is True
    assert cfg.policy.task_field == "task"
    assert cfg.wandb.enable is False and cfg.policy.push_to_hub is False
    assert cfg.policy.scheduler_warmup_steps < cfg.steps
    with pytest.raises(ValueError):
        training.make_config(
            Evo1Config(device="cpu"),
            cfg.dataset,
            tmp_path,
            steps=0,
            batch_size=1,
            seed=0,
            stage="stage1",
        )
