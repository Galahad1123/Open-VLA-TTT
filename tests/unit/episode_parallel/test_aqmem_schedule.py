from dataclasses import fields

import pytest
from observation_memory.test_policy import policy_config

from lerobot_policy_vla_ttt.models.evo1_groot.configuration_evo1_groot import Evo1GrootConfig
from lerobot_policy_vla_ttt.models.t2mem.configuration_t2mem import T2MemConfig
from lerobot_policy_vla_ttt.training.episode_parallel.config import Recipe
from lerobot_policy_vla_ttt.training.train_episode_parallel import resolve_config


def test_four_asymmetric_rounds():
    recipe = Recipe(steps=32, memory_phase_length=6, policy_phase_length=2)
    assert [recipe.phase(i) for i in range(32)] == (["memory"] * 6 + ["policy"] * 2) * 4
    assert recipe.phase_budget("memory") == 24
    assert recipe.phase_budget("policy") == 8
    assert [i for i in range(1, 33) if recipe.phase_boundary(i)] == [6, 8, 14, 16, 22, 24, 30, 32]


def test_legacy_equal_phases():
    recipe = Recipe(steps=8, phase_length=2)
    assert [recipe.phase(i) for i in range(8)] == ["memory", "memory", "policy", "policy"] * 2


def test_freeze_survives_stage_defaults():
    tiny = policy_config(T2MemConfig)
    base = Evo1GrootConfig(**{f.name: getattr(tiny, f.name) for f in fields(Evo1GrootConfig)})
    cfg = resolve_config(
        base, "aqmem", Recipe(freeze_vlm=True, horizon=3, execution=2), device="cpu"
    )
    assert not cfg.finetune_language_model
    assert not cfg.finetune_vision_model
    assert cfg.finetune_action_head


@pytest.mark.parametrize(
    "kwargs", [{"memory_phase_length": 3}, {"memory_phase_length": 0, "policy_phase_length": 1}]
)
def test_invalid_phases(kwargs):
    with pytest.raises(ValueError):
        Recipe(**kwargs)


def test_independent_cosine_and_frozen_vlm(monkeypatch):
    import torch

    from lerobot_policy_vla_ttt.training.episode_parallel.engine import backward_episode
    from lerobot_policy_vla_ttt.training.episode_parallel.optim import PhaseOptimizers

    from .test_engine import episode_events, make_policy

    policy = make_policy("aqmem", monkeypatch)
    policy.config.finetune_language_model = False
    policy.config.finetune_vision_model = False
    recipe = Recipe(
        steps=16,
        memory_phase_length=3,
        policy_phase_length=1,
        warmup_ratio=0.05,
        global_batch_size=1,
        horizon=3,
        execution=2,
        tbptt_events=4,
        freeze_vlm=True,
        memory_lr=5e-4,
        policy_lr=2e-6,
    )
    control = PhaseOptimizers(policy, recipe)
    frozen = {n: p.clone() for n, p in policy.model.embedder.named_parameters()}
    for step in range(recipe.steps):
        control.begin(step)
        assert not any(p.requires_grad for p in policy.model.embedder.parameters())
        backward_episode(
            policy, episode_events(), decisions=3, recipe=recipe, backward=lambda x: x.backward()
        )
        control.step()
    assert control.counts == {"memory": 12, "policy": 4}
    for optimizer in control.optimizers.values():
        assert all(g["lr"] == pytest.approx(g["peak_lr"] * 0.1) for g in optimizer.param_groups)
    assert all(torch.equal(frozen[n], p) for n, p in policy.model.embedder.named_parameters())
