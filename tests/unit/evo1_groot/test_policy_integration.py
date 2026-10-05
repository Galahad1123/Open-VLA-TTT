import json

import numpy as np
import pytest
import torch
from lerobot.configs.policies import PreTrainedConfig

from lerobot_policy_vla_ttt.cli import make_pre_post_processors
from lerobot_policy_vla_ttt.models.evo1_groot.configuration_evo1_groot import Evo1GrootConfig
from lerobot_policy_vla_ttt.models.evo1_groot.modeling_evo1_groot import Evo1GrootPolicy


def batch():
    return {
        "observation.state": torch.rand(2, 8),
        "observation.images.image": torch.rand(2, 3, 16, 16),
        "observation.images.image2": torch.rand(2, 3, 16, 16),
        "action": torch.rand(2, 3, 7),
        "action_is_pad": torch.tensor([[0, 0, 1]] * 2).bool(),
        "task": ["move", "pick"],
        "embodiment_id": torch.zeros(2, dtype=torch.long),
    }


def stats():
    return {
        key: {"min": -torch.ones(width), "max": torch.ones(width)}
        for key, width in (("action", 7), ("observation.state", 8))
    }


@pytest.mark.parametrize("stage", ["stage1", "stage2"])
def test_policy_training_obeys_stage_freezing_and_padding(config, stage):
    import dataclasses

    config = dataclasses.replace(
        config,
        training_stage=stage,
        finetune_vlm=None,
        finetune_language_model=None,
        finetune_vision_model=None,
        finetune_action_head=None,
    )
    policy = Evo1GrootPolicy(config).train()
    pre, _ = make_pre_post_processors(config, dataset_stats=stats())
    loss, info = policy(pre(batch()))
    assert info["active_action_dims"] == 14
    loss.backward()
    assert torch.isfinite(loss)
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in policy.model.action_head.parameters()
    )
    if stage == "stage1":
        assert not policy.model.embedder.training
        assert all(p.grad is None for p in policy.model.embedder.parameters())
    else:
        assert all(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in policy.model.embedder.parameters()
        )


def test_checkpoint_processors_queue_and_inference_server_round_trip(config, tmp_path):
    from lerobot_policy_vla_ttt.evaluation.policy import Evo1Inference

    policy = Evo1GrootPolicy(config).eval()
    pre, post = make_pre_post_processors(config, dataset_stats=stats())
    policy.save_pretrained(tmp_path)
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)
    loaded_config = PreTrainedConfig.from_pretrained(tmp_path, local_files_only=True)
    assert type(loaded_config) is Evo1GrootConfig
    restored = Evo1GrootPolicy.from_pretrained(tmp_path, local_files_only=True)
    loaded_pre, loaded_post = make_pre_post_processors(loaded_config, pretrained_path=tmp_path)
    raw = batch()
    assert "embodiment_id" in loaded_pre(raw)
    torch.manual_seed(2)
    expected = policy.predict_action_chunk(pre(raw))
    torch.manual_seed(2)
    actual = restored.predict_action_chunk(loaded_pre(raw))
    torch.testing.assert_close(actual, expected)
    assert loaded_post(actual[:, 0]).shape == (2, 7)
    torch.manual_seed(2)
    torch.testing.assert_close(restored.select_action(loaded_pre(raw)), expected[:, 0])
    torch.testing.assert_close(restored.select_action(loaded_pre(raw)), expected[:, 1])
    restored.reset()
    assert len(restored._action_queue) == 0
    service = Evo1Inference(tmp_path, "evo1_groot", "libero", device="cpu", n_action_steps=2)
    observation = {
        "state": np.zeros(8, dtype=np.float32),
        "task": "move",
        "images": {k: np.zeros((256, 256, 3), dtype=np.uint8) for k in ("image", "image2")},
    }
    service.reset(22)
    action = service.infer(observation)
    assert action.shape == (7,) and np.isfinite(action).all()
    assert service.metadata["implementation"] == "evo1_groot"
    service.reset(22)
    np.testing.assert_array_equal(action, service.infer(observation))


def test_strict_loading_rejects_other_baselines_and_missing_weights(config, tmp_path):
    from safetensors.torch import load_file, save_file

    (tmp_path / "config.json").write_text(json.dumps({"type": "vla_ttt"}))
    with pytest.raises(ValueError, match="expected 'evo1_groot'"):
        Evo1GrootPolicy.from_pretrained(tmp_path)
    with pytest.raises(ValueError, match="strict=True"):
        Evo1GrootPolicy.from_pretrained(tmp_path, strict=False)
    Evo1GrootPolicy(config).save_pretrained(tmp_path)
    tensors = load_file(tmp_path / "model.safetensors")
    tensors.pop(next(k for k in tensors if "action_head" in k))
    save_file(tensors, tmp_path / "model.safetensors")
    with pytest.raises(RuntimeError):
        Evo1GrootPolicy.from_pretrained(tmp_path, local_files_only=True)


def test_loading_rejects_semantic_architecture_changes_but_allows_runtime_overrides(
    config, tmp_path
):
    import dataclasses

    Evo1GrootPolicy(config).save_pretrained(tmp_path)
    with pytest.raises(ValueError, match="num_heads"):
        Evo1GrootPolicy.from_pretrained(
            tmp_path, config=dataclasses.replace(config, num_heads=2), local_files_only=True
        )
    override = dataclasses.replace(config, n_action_steps=1, training_stage="stage2")
    restored = Evo1GrootPolicy.from_pretrained(tmp_path, config=override, local_files_only=True)
    assert restored.config.n_action_steps == 1
    assert all(p.requires_grad for p in restored.model.embedder.parameters())


def test_loading_preserves_camera_order(config, tmp_path):
    import dataclasses

    Evo1GrootPolicy(config).save_pretrained(tmp_path)
    reordered = dataclasses.replace(
        config, input_features=dict(reversed(config.input_features.items()))
    )
    with pytest.raises(ValueError, match="camera order"):
        Evo1GrootPolicy.from_pretrained(tmp_path, config=reordered, local_files_only=True)
