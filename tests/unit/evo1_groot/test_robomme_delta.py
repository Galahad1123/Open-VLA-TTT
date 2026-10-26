from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature

from lerobot_policy_vla_ttt.models.evo1_groot.configuration_evo1_groot import Evo1GrootConfig
from lerobot_policy_vla_ttt.models.evo1_groot.processor_evo1_groot import (
    make_evo1_groot_pre_post_processors,
)


def delta_config(**kwargs):
    values = dict(
        device="cpu",
        robomme_action_representation="delta_joint",
        input_features={"observation.state": PolicyFeature(FeatureType.STATE, (8,))},
        output_features={"action": PolicyFeature(FeatureType.ACTION, (8,))},
        normalization_mapping={
            "STATE": NormalizationMode.QUANTILES,
            "ACTION": NormalizationMode.QUANTILES,
        },
        postprocess_action_dim=8,
        binarize_gripper=False,
    )
    values.update(kwargs)
    return Evo1GrootConfig(**values)


def stats():
    return {
        key: {"q01": torch.full((8,), -2.0), "q99": torch.full((8,), 4.0)}
        for key in ("observation.state", "action")
    }


def test_delta_roundtrip_preserves_gripper_and_padding():
    pre, post = make_evo1_groot_pre_post_processors(delta_config(), stats())
    state = torch.arange(8.0).unsqueeze(0)
    actions = torch.stack((state + 1, state + 2), dim=1)
    output = pre({"observation.state": state, "action": actions})
    delta = post(output["action"])
    expected = actions.clone()
    expected[..., :7] -= state[:, None, :7]
    torch.testing.assert_close(delta, expected)
    torch.testing.assert_close(output["action"][..., 8:], torch.zeros(1, 2, 16))
    torch.testing.assert_close(actions[0, 0], state[0] + 1)


def test_legacy_config_defaults():
    config = Evo1GrootConfig(device="cpu")
    assert config.robomme_action_representation == "absolute"
    assert config.robomme_augmentation == "none"
    assert config.scheduler_min_lr_ratio == 0


@pytest.mark.parametrize(
    "override",
    [
        dict(postprocess_action_dim=7),
        dict(binarize_gripper=True),
        dict(
            normalization_mapping={
                "STATE": NormalizationMode.MIN_MAX,
                "ACTION": NormalizationMode.QUANTILES,
            }
        ),
        dict(output_features={"action": PolicyFeature(FeatureType.ACTION, (7,))}),
    ],
)
def test_delta_rejects_incompatible_semantics(override):
    with pytest.raises(ValueError, match="delta_joint"):
        make_evo1_groot_pre_post_processors(delta_config(**override), stats())


def test_inference_queue_anchors_entire_chunk_once_and_reset():
    from lerobot_policy_vla_ttt.evaluation.policy import Evo1Inference

    engine = Evo1Inference.__new__(Evo1Inference)
    calls = []

    def predict(batch):
        calls.append(batch)
        return torch.ones(1, 3, 8)

    engine.policy = SimpleNamespace(
        config=SimpleNamespace(robomme_action_representation="delta_joint", n_action_steps=2),
        predict_action_chunk=predict,
        reset=lambda: None,
    )

    class Identity:
        def __call__(self, value):
            return value

        def reset(self):
            pass

    engine.pre = engine.post = Identity()
    engine._absolute_action_queue = deque()

    def obs(value):
        return {
            "state": np.full(8, value),
            "task": "move",
            "images": {key: np.zeros((256, 256, 3), dtype=np.uint8) for key in ("image", "image2")},
        }

    np.testing.assert_array_equal(engine.infer(obs(10)), [11] * 7 + [1])
    np.testing.assert_array_equal(engine.infer(obs(20)), [11] * 7 + [1])
    assert len(calls) == 1
    np.testing.assert_array_equal(engine.infer(obs(25)), [26] * 7 + [1])
    engine.reset(1)
    np.testing.assert_array_equal(engine.infer(obs(30)), [31] * 7 + [1])
    assert len(calls) == 3


def test_serialized_delta_pair_reloads_and_rejects_absolute_pair(tmp_path):
    from lerobot_policy_vla_ttt.cli import make_pre_post_processors

    config = delta_config()
    pre, post = make_evo1_groot_pre_post_processors(config, stats())
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)
    loaded_pre, loaded_post = make_pre_post_processors(config, pretrained_path=tmp_path)
    batch = {"observation.state": torch.ones(1, 8), "action": torch.full((1, 2, 8), 3.0)}
    torch.testing.assert_close(loaded_post(loaded_pre(batch)["action"]), post(pre(batch)["action"]))
    absolute = delta_config(robomme_action_representation="absolute")
    with pytest.raises(ValueError, match="representation"):
        make_pre_post_processors(absolute, pretrained_path=tmp_path)
    pre, post = make_evo1_groot_pre_post_processors(absolute, stats())
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)
    with pytest.raises(ValueError, match="delta_joint"):
        make_pre_post_processors(config, pretrained_path=tmp_path)


def test_delta_generic_select_action_fails_clearly():
    from lerobot_policy_vla_ttt.models.evo1_groot.modeling_evo1_groot import Evo1GrootPolicy

    policy = Evo1GrootPolicy.__new__(Evo1GrootPolicy)
    policy.config = delta_config()
    with pytest.raises(ValueError, match="Evo1Inference"):
        policy.select_action({})


def test_delta_pair_rejects_mismatched_quantiles(tmp_path):
    from lerobot.processor import UnnormalizerProcessorStep

    from lerobot_policy_vla_ttt.cli import make_pre_post_processors

    config = delta_config()
    pre, post = make_evo1_groot_pre_post_processors(config, stats())
    pre.save_pretrained(tmp_path)
    step = next(s for s in post.steps if isinstance(s, UnnormalizerProcessorStep))
    step.stats["action"]["q99"] = step.stats["action"]["q99"].clone() + 1
    step.to(device=step.device, dtype=step.dtype)
    post.save_pretrained(tmp_path)
    with pytest.raises(ValueError, match="quantiles do not match"):
        make_pre_post_processors(config, pretrained_path=tmp_path)
