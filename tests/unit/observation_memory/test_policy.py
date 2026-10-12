from dataclasses import fields

import pytest
import torch
from lerobot.configs.types import FeatureType, PolicyFeature

from lerobot_policy_vla_ttt.models.aqmem.modeling_aqmem import AQMemPolicy
from lerobot_policy_vla_ttt.models.evo1_groot.configuration_evo1_groot import Evo1GrootConfig
from lerobot_policy_vla_ttt.models.evo1_groot.modeling_evo1_groot import Evo1GrootPolicy
from lerobot_policy_vla_ttt.models.t2mem.modeling_t2mem import T2MemPolicy

from .assets import patch_assets
from .test_connections import KINDS, config


def batch():
    return {
        "observation.state": torch.rand(2, 4),
        "observation.images.image": torch.rand(2, 3, 16, 16),
        "action": torch.rand(2, 3, 4),
        "task": ["move", "pick"],
    }


def policy_config(cls, **kwargs):
    return config(
        cls,
        **(
            dict(
                vlm_num_layers=2,
                vlm_dtype="float32",
                image_resolution=(16, 16),
                input_features={
                    "observation.state": PolicyFeature(FeatureType.STATE, (4,)),
                    "observation.images.image": PolicyFeature(FeatureType.VISUAL, (3, 16, 16)),
                },
                output_features={"action": PolicyFeature(FeatureType.ACTION, (4,))},
            )
            | kwargs
        ),
    )


@pytest.mark.parametrize("cls,policy_cls", [(KINDS[0][0], AQMemPolicy), (KINDS[1][0], T2MemPolicy)])
def test_real_backbone_export_bypass_and_joint_gradient(cls, policy_cls, monkeypatch):
    patch_assets(monkeypatch)
    cfg = policy_config(
        cls, memory_training="joint", training_stage="stage2", enable_gradient_checkpointing=True
    )
    base = Evo1GrootPolicy(
        Evo1GrootConfig(**{f.name: getattr(cfg, f.name) for f in fields(Evo1GrootConfig)})
    ).eval()
    p = policy_cls(cfg)
    p.initialize_from_baseline(base)
    p.config.memory_enabled = False
    b = batch()
    torch.manual_seed(4)
    expected = base.predict_action_chunk(b)
    torch.manual_seed(4)
    actual = p.predict_action_chunk(b)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    p.config.memory_enabled = True
    p.train()
    _, _, memory = p.forward_step(b)
    loss, count, _ = p.forward_step(b, memory)
    (loss / count).backward()
    assert p.model.action_head.memories[0].k.weight.grad.abs().sum() > 0
    assert any(
        v.grad is not None and v.grad.abs().sum() > 0
        for v in p.model.embedder.model.vision_tower.parameters()
    )
    assert p.model.embedder._capture is None


@pytest.mark.parametrize("cls,policy_cls", [(KINDS[0][0], AQMemPolicy), (KINDS[1][0], T2MemPolicy)])
def test_clock_midpoint_duplicates_and_checkpoint(cls, policy_cls, monkeypatch, tmp_path):
    patch_assets(monkeypatch)
    p = policy_cls(policy_config(cls, chunk_size=6, n_action_steps=4)).eval()
    before = {k: v.clone() for k, v in p.state_dict().items()}
    b = batch()
    b.pop("action")
    first = p.select_action(b, env_step=0)
    assert p.inference_state[0].update_count.tolist() == [1, 1]
    torch.testing.assert_close(first, p.select_action(b, env_step=0))
    p.select_action(b, env_step=1)
    p.select_action(b, env_step=2)
    assert p.inference_state[0].update_count.tolist() == [2, 2]
    p.select_action(b, env_step=3)
    p.select_action(b, env_step=4)
    assert p.inference_state[0].update_count.tolist() == [3, 3]
    with pytest.raises(ValueError, match="skip"):
        p.select_action(b, env_step=7)
    for k, v in p.state_dict().items():
        torch.testing.assert_close(v, before[k])
    assert all(v.grad is None for v in p.parameters())
    assert all(w.grad_fn is None for s in p.inference_state for w in s.weights)
    p.save_pretrained(tmp_path)
    restored = policy_cls.from_pretrained(tmp_path, local_files_only=True)
    assert restored.inference_state is None
    p.reset()
    assert p.inference_state is None


@pytest.mark.parametrize("cls,policy_cls", [(KINDS[0][0], AQMemPolicy), (KINDS[1][0], T2MemPolicy)])
def test_write_is_label_state_independent_and_clock_failure_atomic(cls, policy_cls, monkeypatch):
    patch_assets(monkeypatch)
    p = policy_cls(policy_config(cls, chunk_size=6, n_action_steps=4)).eval()
    b = batch()
    # Context-only events must not touch actions or proprioception values.
    modified = dict(b, action=torch.full_like(b["action"], float("nan")))
    modified["observation.state"] = b["observation.state"] + 100
    mask = torch.zeros(2, dtype=torch.bool)
    _, _, a = p.forward_step(b, supervision_mask=mask)
    _, _, c = p.forward_step(modified, supervision_mask=mask)
    for x, y in zip(a, c, strict=True):
        for u, v in zip(x.weights, y.weights, strict=True):
            torch.testing.assert_close(u, v)
    with pytest.raises(ValueError):
        p.select_action(b, env_step=-1)
    p.select_action(b, env_step=0)
    with pytest.raises(ValueError, match="batch|Batch"):
        p.select_action({k: v[:1] for k, v in b.items()}, env_step=1)
    assert p.clock.last == 0
    with pytest.raises(ValueError, match="skip"):
        p.predict_action_chunk(b, env_step=4)
    assert p.clock.last == 0
    p.observe(b, env_step=2)
    memory = p.inference_state

    def fail(*args, **kwargs):
        raise RuntimeError("injected sample failure")

    monkeypatch.setattr(p.model.action_head, "sample", fail)
    with pytest.raises(RuntimeError, match="injected"):
        p.predict_action_chunk(b, env_step=4)
    assert p.inference_state is memory and p.clock.last == 2


@pytest.mark.parametrize("cls,policy_cls", [(KINDS[0][0], AQMemPolicy), (KINDS[1][0], T2MemPolicy)])
def test_actual_service_uses_midpoint_and_preserves_all_parameters(
    cls, policy_cls, monkeypatch, tmp_path
):
    import numpy as np

    from lerobot_policy_vla_ttt.cli import make_pre_post_processors
    from lerobot_policy_vla_ttt.evaluation.policy import Evo1Inference

    patch_assets(monkeypatch)
    cfg = policy_config(
        cls,
        chunk_size=6,
        n_action_steps=4,
        max_state_dim=8,
        max_action_dim=8,
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (8,)),
            **{
                f"observation.images.{name}": PolicyFeature(FeatureType.VISUAL, (3, 16, 16))
                for name in ("image", "image2")
            },
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (7,))},
    )
    p = policy_cls(cfg)
    p.save_pretrained(tmp_path)
    stats = {
        k: {"min": -torch.ones(n), "max": torch.ones(n)}
        for k, n in [("action", 7), ("observation.state", 8)]
    }
    pre, post = make_pre_post_processors(cfg, dataset_stats=stats)
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)
    vlm = tmp_path / "relocated-vlm"
    vlm.mkdir()
    service = Evo1Inference(
        tmp_path, cfg.type, "libero", device="cpu", n_action_steps=4, vlm_path=vlm
    )
    assert service.policy.config.vlm_model_name == str(vlm)
    before = {k: v.clone() for k, v in service.policy.state_dict().items()}
    obs = {
        "state": np.zeros(8, np.float32),
        "task": "move",
        "images": {k: np.zeros((256, 256, 3), np.uint8) for k in ("image", "image2")},
    }
    service.reset(17)
    for _ in range(5):
        assert service.infer(obs).shape == (7,)
    assert service.policy.inference_state[0].update_count.tolist() == [3]
    assert service.policy.inference_state[0].decision_index.tolist() == [2]
    for k, v in service.policy.state_dict().items():
        torch.testing.assert_close(v, before[k])
    assert all(p.grad is None for p in service.policy.parameters())
    service.reset(17)
    assert service.policy.inference_state is None
