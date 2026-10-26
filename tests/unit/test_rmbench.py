import numpy as np
import pytest
import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from torch import nn
from vla_ttt_eval.rmbench import canonical_observation

from lerobot_policy_vla_ttt.baselines.rmbench import TASKS, resolve_tasks
from lerobot_policy_vla_ttt.evaluation.policy import observation_batch, validate_config
from lerobot_policy_vla_ttt.models.evo1_groot.configuration_evo1_groot import Evo1GrootConfig
from lerobot_policy_vla_ttt.models.evo1_groot.modeling_evo1_groot import Evo1GrootPolicy


def test_fixed_tasks():
    assert len(TASKS) == 9
    assert resolve_tasks(["all"]) == list(TASKS)
    assert resolve_tasks(["swap_T"]) == ["swap_T"]
    for names in ([], ["unknown"], ["swap_T", "swap_T"], ["all", "swap_T"]):
        with pytest.raises(ValueError):
            resolve_tasks(names)


def test_observation_preserves_joint_order_and_rgb():
    images = {
        name: {"rgb": np.full((240, 320, 3), i, np.uint8)}
        for i, name in enumerate(("head_camera", "left_camera", "right_camera"))
    }
    raw = {"observation": images, "joint_action": {"vector": np.arange(14)}}
    obs = canonical_observation(raw, "put the block back")
    np.testing.assert_array_equal(obs["state"], np.arange(14))
    assert list(obs["images"]) == ["image", "image2", "image3"]
    assert obs["images"]["image3"][0, 0, 0] == 2
    raw["joint_action"]["vector"] = np.zeros(8)
    with pytest.raises(ValueError):
        canonical_observation(raw, "task")


def test_grouped_lr_covers_parameters_and_preserves_ratio():
    policy = Evo1GrootPolicy.__new__(Evo1GrootPolicy)
    nn.Module.__init__(policy)
    policy.config = Evo1GrootConfig(optimizer_lr=1e-4, optimizer_vlm_lr=1e-5)
    policy.model = nn.Module()
    policy.model.embedder = nn.Linear(2, 2)
    policy.model.action_head = nn.Linear(2, 2)
    groups = policy.get_optim_params()
    params = [p for g in groups for p in g["params"]]
    assert len(params) == len(set(map(id, params))) == 4
    for g in groups:
        expected = (
            1e-5
            if any(
                p is policy.model.embedder.weight or p is policy.model.embedder.bias
                for p in g["params"]
            )
            else 1e-4
        )
        assert g["lr"] == expected
    opt = torch.optim.AdamW(groups, lr=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 0.5)
    assert sorted(set(sched.get_last_lr())) == [5e-6, 5e-5]
    policy.model.embedder.requires_grad_(False)
    assert all(g["lr"] == 1e-4 for g in policy.get_optim_params())


def test_default_optimizer_unchanged():
    policy = Evo1GrootPolicy.__new__(Evo1GrootPolicy)
    nn.Module.__init__(policy)
    policy.config = Evo1GrootConfig()
    policy.model = nn.Linear(2, 2)
    assert all("lr" not in g for g in policy.get_optim_params())


def test_server_rmbench_contract():
    config = Evo1GrootConfig(
        input_features={
            **{
                f"observation.images.image{s}": PolicyFeature(FeatureType.VISUAL, (3, 240, 320))
                for s in ("", "2", "3")
            },
            "observation.state": PolicyFeature(FeatureType.STATE, (14,)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (14,))},
    )
    assert validate_config(config, "rmbench", 10) == 14
    obs = {
        "state": np.zeros(14),
        "task": "task",
        "images": {f"image{s}": np.full((240, 320, 3), 255, np.uint8) for s in ("", "2", "3")},
    }
    batch = observation_batch(obs, benchmark="rmbench")
    assert batch["observation.images.image3"].shape == (1, 3, 240, 320)
    assert batch["observation.images.image3"].min() == 1
    with pytest.raises(ValueError):
        validate_config(config, "libero", 10)
