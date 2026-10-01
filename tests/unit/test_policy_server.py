import importlib

import numpy as np
import pytest


def test_server_rejects_wrong_camera_and_action_semantics_before_model_load():
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.evo1.configuration_evo1 import Evo1Config

    bridge = importlib.import_module("lerobot_policy_vla_ttt.evaluation.policy")
    config = Evo1Config(
        device="cpu",
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (8,)),
            "observation.images.image": PolicyFeature(FeatureType.VISUAL, (3, 256, 256)),
            "observation.images.image2": PolicyFeature(FeatureType.VISUAL, (3, 256, 256)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (7,))},
    )
    bridge.validate_config(config, "libero", 10)
    with pytest.raises(ValueError, match="action"):
        bridge.validate_config(config, "robomme", 10)
    config.input_features["observation.images.other"] = config.input_features.pop(
        "observation.images.image2"
    )
    with pytest.raises(ValueError, match="camera"):
        bridge.validate_config(config, "libero", 10)


def test_checkpoint_identity_changes_if_weights_change(tmp_path):
    module = importlib.import_module("lerobot_policy_vla_ttt.evaluation.policy")
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"first")
    original = module.checkpoint_identity(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"second")
    assert module.checkpoint_identity(tmp_path) != original


def test_inference_boundary_builds_batch_and_rejects_wrong_state():
    bridge = importlib.import_module("lerobot_policy_vla_ttt.evaluation.policy")
    obs = {
        "images": {
            "image": np.full((256, 256, 3), 255, np.uint8),
            "image2": np.zeros((256, 256, 3), np.uint8),
        },
        "state": np.arange(8, dtype=np.float32),
        "task": "pick",
    }
    batch = bridge.observation_batch(obs)
    assert batch["observation.images.image"].shape == (1, 3, 256, 256)
    assert batch["observation.images.image"].max().item() == 1
    assert batch["observation.state"].shape == (1, 8)
    assert batch["task"] == ["pick"]
    with pytest.raises(ValueError):
        bridge.observation_batch({**obs, "state": np.zeros(7)})
    with pytest.raises(ValueError):
        bridge.observation_batch({**obs, "state": np.full(8, np.nan)})
