import json
from dataclasses import replace

import pytest
import torch
from accelerate import Accelerator

from lerobot_policy_vla_ttt.training.episode_parallel.checkpoint import load_checkpoint
from lerobot_policy_vla_ttt.training.episode_parallel.config import Recipe
from lerobot_policy_vla_ttt.training.episode_parallel.data import Episode, EpisodeSource
from lerobot_policy_vla_ttt.training.episode_parallel.optim import PhaseOptimizers
from lerobot_policy_vla_ttt.training.episode_parallel.runner import run

from .test_engine import make_policy


def assert_control_equal(actual, expected):
    assert actual["counts"] == expected["counts"]
    assert actual["optimizers"].keys() == expected["optimizers"].keys()
    for phase in actual["optimizers"]:
        a, b = actual["optimizers"][phase], expected["optimizers"][phase]
        assert a["param_groups"] == b["param_groups"]
        torch.testing.assert_close(a["state"], b["state"], atol=0, rtol=0)


def source():
    g = torch.Generator().manual_seed(14)
    frames = [
        {
            "observation.state": torch.rand(4, generator=g),
            "observation.images.image": torch.rand(3, 16, 16, generator=g),
            "action": torch.rand(3, 4, generator=g),
            "task": "move",
        }
        for _ in range(10)
    ]
    return EpisodeSource(
        frames, [Episode(0, "move", 0, 3), Episode(1, "move", 3, 10)], execution=2, horizon=3
    )


@pytest.mark.parametrize("joint", [False, True])
@pytest.mark.parametrize("split_step", [1, 2, 3])
def test_resume_matches_uninterrupted_across_phase_boundary(
    monkeypatch, tmp_path, split_step, joint
):
    accelerator = Accelerator(cpu=True, mixed_precision="no")
    r = Recipe(
        steps=4,
        phase_length=1,
        warmup_steps=1,
        horizon=3,
        execution=2,
        global_batch_size=2,
        tbptt_events=3,
        save_every=2,
        training_mode="joint" if joint else "alternating",
        sampling_mode="mixed_tasks" if joint else "same_task",
    )
    torch.manual_seed(12)
    whole = make_policy("aqmem" if joint else "t2mem", monkeypatch)
    expected = run(
        whole,
        mixed_source() if joint else source(),
        None,
        None,
        recipe=r,
        accelerator=accelerator,
        output=tmp_path / "whole",
        identity={"dataset": "fixture"},
    )
    torch.manual_seed(12)
    split = make_policy("aqmem" if joint else "t2mem", monkeypatch)
    run(
        split,
        mixed_source() if joint else source(),
        None,
        None,
        recipe=r,
        accelerator=accelerator,
        output=tmp_path / "split",
        identity={"dataset": "fixture"},
        stop_after=split_step,
    )
    checkpoint = tmp_path / f"split/step-{split_step:08d}"
    torch.manual_seed(999)
    restored = make_policy("aqmem" if joint else "t2mem", monkeypatch)
    actual = run(
        restored,
        mixed_source() if joint else source(),
        None,
        None,
        recipe=r,
        accelerator=accelerator,
        output=tmp_path / "resumed",
        identity={"dataset": "fixture"},
        resume=checkpoint,
    )
    torch.testing.assert_close(whole.state_dict(), restored.state_dict(), atol=0, rtol=0)
    assert (
        actual["counts"]
        == expected["counts"]
        == ({"joint": 4} if joint else {"memory": 2, "policy": 2})
    )
    assert actual["totals"] == expected["totals"]
    manifest = json.loads((tmp_path / "resumed/run.json").read_text())
    assert manifest["world_size"] == 1 and manifest["recipe"]["global_batch_size"] == 2
    assert len((tmp_path / "resumed/metrics.jsonl").read_text().splitlines()) == 4 - split_step
    final_whole = torch.load(tmp_path / "whole/step-00000004/training.pt", weights_only=False)
    final_restored = torch.load(tmp_path / "resumed/step-00000004/training.pt", weights_only=False)
    assert_control_equal(final_whole["control"], final_restored["control"])
    saved = torch.load(checkpoint / "training.pt", weights_only=False)
    if split_step >= 2:
        assert all(
            saved["control"]["optimizers"][phase]["state"]
            for phase in saved["control"]["optimizers"]
        )
    whole_metrics = [
        json.loads(line) for line in (tmp_path / "whole/metrics.jsonl").read_text().splitlines()
    ]
    resumed_metrics = [
        json.loads(line) for line in (tmp_path / "resumed/metrics.jsonl").read_text().splitlines()
    ]
    for expected_row, actual_row in zip(whole_metrics[split_step:], resumed_metrics, strict=True):
        for key in (
            "loss",
            "global_task_count",
            "global_task_counts",
            "optimizer_groups",
            "totals",
        ):
            assert expected_row[key] == actual_row[key]
        assert actual_row["global_task_count"] == (2 if joint else 1)
    control = PhaseOptimizers(restored, r)
    for field, value in (
        ("training_mode", "alternating" if joint else "joint"),
        ("sampling_mode", "same_task" if joint else "mixed_tasks"),
    ):
        with pytest.raises(RuntimeError, match="recipe"):
            load_checkpoint(
                checkpoint,
                restored,
                control,
                accelerator,
                recipe=replace(r, **{field: value}),
                identity={"dataset": "fixture"},
            )
    with pytest.raises(RuntimeError, match="identity"):
        load_checkpoint(
            checkpoint, restored, control, accelerator, recipe=r, identity={"dataset": "other"}
        )
    with pytest.raises(RuntimeError, match="recipe"):
        load_checkpoint(
            checkpoint,
            restored,
            control,
            accelerator,
            recipe=replace(r, global_batch_size=4),
            identity={"dataset": "fixture"},
        )
    for key, invalid in [("world_size", 2), ("model_contract", {})]:
        torch.save(saved | {key: invalid}, checkpoint / "training.pt")
        with pytest.raises(RuntimeError, match=key):
            load_checkpoint(
                checkpoint,
                restored,
                control,
                accelerator,
                recipe=r,
                identity={"dataset": "fixture"},
            )
    torch.save(saved, checkpoint / "training.pt")
    (checkpoint / "COMPLETE").unlink()
    with pytest.raises(RuntimeError, match="Incomplete"):
        load_checkpoint(
            checkpoint, restored, control, accelerator, recipe=r, identity={"dataset": "fixture"}
        )


def mixed_source():
    data = source()
    frames = [dict(frame, task="move" if i < 3 else "lift") for i, frame in enumerate(data.dataset)]
    return EpisodeSource(
        frames, [Episode(0, "move", 0, 3), Episode(1, "lift", 3, 10)], execution=2, horizon=3
    )


def test_legacy_recipe_missing_only_new_keys_restores(monkeypatch, tmp_path):
    accelerator = Accelerator(cpu=True, mixed_precision="no")
    recipe = Recipe(steps=2, phase_length=1, global_batch_size=1, horizon=3, execution=2)
    torch.manual_seed(12)
    policy = make_policy("aqmem", monkeypatch)
    run(
        policy,
        source(),
        None,
        None,
        recipe=recipe,
        accelerator=accelerator,
        output=tmp_path / "first",
        identity={},
        stop_after=1,
    )
    path = tmp_path / "first/step-00000001"
    saved = torch.load(path / "training.pt", weights_only=False)
    for key in ("training_mode", "sampling_mode"):
        del saved["recipe"][key]
    torch.save(saved, path / "training.pt")
    control = PhaseOptimizers(policy, recipe)
    step, _ = load_checkpoint(path, policy, control, accelerator, recipe=recipe, identity={})
    assert step == 1
    assert_control_equal(control.state_dict(), saved["control"])
    del saved["recipe"]["seed"]
    torch.save(saved, path / "training.pt")
    with pytest.raises(RuntimeError, match="recipe"):
        load_checkpoint(path, policy, control, accelerator, recipe=recipe, identity={})
