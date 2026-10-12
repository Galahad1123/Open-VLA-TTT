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
    for phase in ("memory", "policy"):
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


@pytest.mark.parametrize("split_step", [1, 2, 3])
def test_resume_matches_uninterrupted_across_phase_boundary(monkeypatch, tmp_path, split_step):
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
    )
    torch.manual_seed(12)
    whole = make_policy("t2mem", monkeypatch)
    expected = run(
        whole,
        source(),
        None,
        None,
        recipe=r,
        accelerator=accelerator,
        output=tmp_path / "whole",
        identity={"dataset": "fixture"},
    )
    torch.manual_seed(12)
    split = make_policy("t2mem", monkeypatch)
    run(
        split,
        source(),
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
    restored = make_policy("t2mem", monkeypatch)
    actual = run(
        restored,
        source(),
        None,
        None,
        recipe=r,
        accelerator=accelerator,
        output=tmp_path / "resumed",
        identity={"dataset": "fixture"},
        resume=checkpoint,
    )
    torch.testing.assert_close(whole.state_dict(), restored.state_dict(), atol=0, rtol=0)
    assert actual["counts"] == expected["counts"] == {"memory": 2, "policy": 2}
    assert actual["totals"] == expected["totals"]
    manifest = json.loads((tmp_path / "resumed/run.json").read_text())
    assert manifest["world_size"] == 1 and manifest["recipe"]["global_batch_size"] == 2
    assert len((tmp_path / "resumed/metrics.jsonl").read_text().splitlines()) == 4 - split_step
    final_whole = torch.load(tmp_path / "whole/step-00000004/training.pt", weights_only=False)
    final_restored = torch.load(tmp_path / "resumed/step-00000004/training.pt", weights_only=False)
    assert_control_equal(final_whole["control"], final_restored["control"])
    saved = torch.load(checkpoint / "training.pt", weights_only=False)
    if split_step >= 2:
        assert all(saved["control"]["optimizers"][phase]["state"] for phase in ("memory", "policy"))
    control = PhaseOptimizers(restored, r)
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
