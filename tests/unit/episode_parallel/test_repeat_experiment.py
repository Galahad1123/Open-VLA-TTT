import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from lerobot_policy_vla_ttt.training.episode_parallel.config import Recipe


def test_diagnostics_default_and_validation():
    assert Recipe().diagnostics_every == 0
    assert Recipe(diagnostics_every=50).diagnostics_every == 50
    for bad in [-1, True, 1.5]:
        with pytest.raises(ValueError):
            Recipe(diagnostics_every=bad)


def test_repeat_classification_checks_goals():
    p = Path("scripts/prepare_robomme_repeats.py")
    spec = importlib.util.spec_from_file_location("prepare_repeats", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.repeat_count("pick it twice, repeating this action two times") == 2
    assert mod.repeat_count("pick up the cube then stop") == 1
    assert (
        mod.goal_count(
            "PickXtimes",
            ["pick up the red cube for the first time", "pick up the red cube for the second time"],
        )
        == 2
    )
    assert (
        mod.goal_count(
            "SwingXtimes",
            [
                "move to the top of the left-side target for the first time",
                "move to the top of the left-side target for the third time",
            ],
        )
        == 3
    )


def test_repeat_eval_preserves_legacy_selection():
    from vla_ttt_eval.subset import eligible, summarize

    env = SimpleNamespace(num_repeats=2)
    assert not eligible("PickXtimes", env)
    assert eligible("PickXtimes", env, [1, 2, 3])
    rows = [
        {"type": "episode", "task": "PickXtimes", "num_repeats": n, "success": s}
        for n, s in [(1, True), (2, False), (2, True)]
    ]
    groups = summarize(rows)["repeat_groups"]
    assert groups["PickXtimes/2"]["episodes"] == 2
    assert groups["PickXtimes/2"]["success_rate"] == 0.5


def test_diagnostics_preserve_rng_gradients_and_memory(monkeypatch):
    import torch

    from lerobot_policy_vla_ttt.training.episode_parallel.diagnostics import Diagnostics
    from lerobot_policy_vla_ttt.training.episode_parallel.microbatch import backward_microbatch
    from lerobot_policy_vla_ttt.training.episode_parallel.optim import PhaseOptimizers

    from .test_engine import episode_events, make_policy
    from .test_microbatch import frozen

    results = []
    for interval in [0, 1]:
        torch.manual_seed(91)
        p = frozen(make_policy("aqmem", monkeypatch))
        recipe = Recipe(
            steps=2,
            training_mode="joint",
            global_batch_size=2,
            horizon=3,
            execution=2,
            freeze_vlm=True,
            episode_microbatch_size=2,
            diagnostics_every=interval,
        )
        optimizer = PhaseOptimizers(p, recipe)
        phase = optimizer.begin(0)
        with Diagnostics(interval) as diag:
            diag.begin(1)
            events = [episode_events(3), episode_events(3)]
            metrics, memory = backward_microbatch(
                p,
                events,
                decisions=[2, 2],
                seeds=[4, 5],
                recipe=recipe,
                backward=lambda x: x.backward(),
            )
            rng = torch.get_rng_state().clone()
            diag.before_step(optimizer.optimizers[phase], p)
            grads = {k: v.grad.clone() for k, v in p.named_parameters() if v.grad is not None}
            optimizer.step()
            info = diag.after_step()
            assert torch.equal(rng, torch.get_rng_state())
        if interval:
            assert info["optimizer_groups"] and info["memory_reads"] and info["fast_weights"]
        results.append((metrics, memory, grads, p.state_dict()))
    assert results[0][0] == results[1][0]
    for a, b in zip(results[0][1], results[1][1], strict=True):
        torch.testing.assert_close(a.weights, b.weights, rtol=0, atol=0)
    torch.testing.assert_close(results[0][2], results[1][2], rtol=0, atol=0)
    torch.testing.assert_close(results[0][3], results[1][3], rtol=0, atol=0)


def test_committed_recipes_match_and_sample_all_six_strata():
    import json
    from dataclasses import asdict

    from lerobot_policy_vla_ttt.training.episode_parallel.data import (
        Episode,
        MixedTaskEpisodeSampler,
    )

    root = Path("configs/training/robomme-screen")
    recipes = [
        Recipe(**json.loads((root / f"repeat-two-frozen-{kind}.json").read_text()))
        for kind in ["baseline", "aqmem"]
    ]
    a, b = map(asdict, recipes)
    assert {k for k in a if a[k] != b[k]} == {"training_mode"}
    groups = recipes[0].task_groups
    assert sum(map(len, groups.values())) == 175
    episodes = [Episode(i, g, i * 1000, (i + 1) * 1000) for g, ids in groups.items() for i in ids]
    sampler = MixedTaskEpisodeSampler(episodes, seed=1000, global_batch_size=6)
    task = {e.index: e.task for e in episodes}
    for step in range(20):
        assert {task[s.episode_id] for s in sampler.batch(step)} == set(groups)
