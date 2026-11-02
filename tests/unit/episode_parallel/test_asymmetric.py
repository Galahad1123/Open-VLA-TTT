import copy

import pytest
import torch

from lerobot_policy_vla_ttt.training.episode_parallel.config import Recipe
from lerobot_policy_vla_ttt.training.episode_parallel.microbatch import backward_microbatch
from lerobot_policy_vla_ttt.training.episode_parallel.optim import PhaseOptimizers

from .test_engine import episode_events, make_policy
from .test_microbatch import frozen


def test_240_60_schedule_and_checkpoints():
    r = Recipe(
        steps=1200,
        memory_phase_length=240,
        policy_phase_length=60,
        save_phase_boundaries=True,
        save_every=1200,
    )
    assert [r.phase(i) for i in range(1200)] == (["memory"] * 240 + ["policy"] * 60) * 4
    assert r.phase_budget("memory") == 960
    assert r.phase_budget("policy") == 240
    assert [i for i in range(1, 1201) if r.should_save(i, 1200)] == [
        240,
        300,
        540,
        600,
        840,
        900,
        1140,
        1200,
    ]
    assert r.should_save(2, 2)  # stop-after still saves
    assert Recipe(phase_length=2).phase(2) == "policy"


@pytest.mark.parametrize(
    "field,bad",
    [
        ("memory_phase_length", 0),
        ("policy_phase_length", True),
        ("memory_warmup_steps", -1),
        ("policy_warmup_steps", 1.5),
        ("save_phase_boundaries", 1),
    ],
)
def test_invalid_options(field, bad):
    with pytest.raises(ValueError):
        Recipe(**{field: bad})


def test_microbatch_alternation_freezes_weights_and_resume_keeps_schedule(monkeypatch):
    torch.manual_seed(12)
    p = frozen(make_policy("aqmem", monkeypatch))
    r = Recipe(
        steps=6,
        memory_phase_length=2,
        policy_phase_length=1,
        memory_warmup_steps=2,
        policy_warmup_steps=1,
        global_batch_size=2,
        episode_microbatch_size=2,
        freeze_vlm=True,
        horizon=3,
        execution=2,
        tbptt_events=2,
    )
    control = PhaseOptimizers(p, r)
    for step in range(6):
        phase = control.begin(step)
        if step in (0, 1, 2, 5):
            factor = {0: 0.5, 1: 1.0, 2: 1.0, 5: 0.1}[step]
            for group in control.optimizers[phase].param_groups:
                assert group["lr"] == pytest.approx(group["peak_lr"] * factor)
        before = {n: x.detach().clone() for n, x in p.named_parameters()}
        active = {n for n, x in p.named_parameters() if x.requires_grad}
        _, memory = backward_microbatch(
            p,
            [episode_events(3), episode_events(3)],
            decisions=[2, 2],
            seeds=[10, 11],
            recipe=r,
            backward=lambda x: x.backward(),
        )
        assert any(
            x.grad is not None and x.grad.abs().sum() > 0
            for n, x in p.named_parameters()
            if n in active
        )
        assert memory is not None
        control.step()
        assert all(torch.equal(before[n], x) for n, x in p.named_parameters() if n not in active)
        assert any(not torch.equal(before[n], x) for n, x in p.named_parameters() if n in active)
        if step == 2:
            state = copy.deepcopy(control.state_dict())
            control = PhaseOptimizers(p, r)
            control.load_state_dict(state)
            assert control.counts == {"memory": 2, "policy": 1}
    assert control.counts == {"memory": 4, "policy": 2}


def test_old_checkpoint_recipe_retains_equal_phases():
    from dataclasses import asdict

    from lerobot_policy_vla_ttt.training.episode_parallel.checkpoint import normalize_saved_recipe

    current = asdict(Recipe())
    old = {
        k: v
        for k, v in current.items()
        if k
        not in {
            "memory_phase_length",
            "policy_phase_length",
            "memory_warmup_steps",
            "policy_warmup_steps",
            "save_phase_boundaries",
        }
    }
    assert normalize_saved_recipe(old) == current


def test_experiment_recipe_preserves_data_and_budget():
    import json
    from pathlib import Path

    root = Path("configs/training/robomme-screen")
    original = json.loads((root / "repeat-two-frozen-aqmem.json").read_text())
    values = json.loads((root / "repeat-two-alternating-aqmem-240-60.json").read_text())
    r = Recipe(**values)
    assert r.task_groups == original["task_groups"]
    assert r.steps == 1200 and r.global_batch_size == r.episode_microbatch_size == 6
    assert r.freeze_vlm and r.training_mode == "alternating"
    assert r.policy_lr == 1e-6 and r.memory_lr == 1e-4
    assert r.phase_warmup("memory") == 50 and r.phase_warmup("policy") == 20
