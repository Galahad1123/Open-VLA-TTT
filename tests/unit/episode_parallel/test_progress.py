import io
from types import SimpleNamespace

import pytest

from lerobot_policy_vla_ttt.training.episode_parallel.progress import Output, Progress


class Clock:
    now = 0

    def __call__(self):
        return self.now


def make(mode="auto", tty=False, start=0, end=10, main=True, phase=None):
    terminal, log, clock = io.StringIO(), io.StringIO(), Clock()
    recipe = SimpleNamespace(steps=20, save_every=5, phase=phase or (lambda s: "policy"))
    output = Output(terminal.write, log.write, tty)
    progress = Progress(recipe, start, end, mode=mode, enabled=main, output=output, clock=clock)
    return progress, terminal, log, clock


def update(p, clock, step, phase="policy", seconds=2):
    clock.now += seconds
    p.update({"step": step, "phase": phase, "loss": 0.1, "update_seconds": seconds})


def test_plain_rate_limit_and_stop_after():
    p, terminal, log, clock = make(start=5, end=10)
    for step in range(6, 10):
        update(p, clock, step)
    assert terminal.getvalue().count("\n") == 1
    assert "6/10" in terminal.getvalue() and "recipe=20" in terminal.getvalue()
    assert p.eta() == (2, False)
    clock.now += 60
    update(p, clock, 10)
    p.finish()
    assert "complete" in terminal.getvalue() and "ETA=" in terminal.getvalue()
    assert "\r" not in log.getvalue()


def test_bar_updates_without_polluting_log():
    p, terminal, log, clock = make(tty=True)
    for step in range(1, 5):
        update(p, clock, step)
    assert terminal.getvalue().count("\r") == 4
    assert log.getvalue().count("\n") == 1
    assert "\r" not in log.getvalue()
    p.saving()
    clock.now += 3
    p.saved()
    assert "saved" in log.getvalue()
    assert p.eta()[0] == 18  # six updates at 2s + two remaining saves at 3s


def test_phase_eta_uses_future_schedule_and_unseen_is_provisional():
    p, _, _, clock = make(phase=lambda s: "memory" if s < 5 else "policy")
    for step in range(1, 4):
        update(p, clock, step, "memory")
    assert p.eta() == (14, True)
    for step in range(4, 6):
        update(p, clock, step, "memory")
    update(p, clock, 6, "policy", 4)
    assert p.eta() == (16, False)


@pytest.mark.parametrize("mode,main", [("off", True), ("bar", False)])
def test_off_and_non_main_are_silent(mode, main):
    p, terminal, log, clock = make(mode=mode, main=main)
    update(p, clock, 1)
    p.saving()
    p.saved()
    p.finish()
    assert not terminal.getvalue() and not log.getvalue()


def test_failure_never_claims_complete():
    p, terminal, log, clock = make()
    with pytest.raises(ValueError):
        with p:
            update(p, clock, 1)
            raise ValueError("failed save")
    assert "failed" in terminal.getvalue()
    assert "complete" not in log.getvalue()


def test_rank_log_preserves_tty_and_keeps_refreshes_out_of_file(tmp_path, monkeypatch, capfd):
    import os

    from lerobot_policy_vla_ttt.training.episode_parallel.run_files import rank_log

    monkeypatch.setattr(os, "isatty", lambda fd: True)
    clock = Clock()
    recipe = SimpleNamespace(steps=4, save_every=4, phase=lambda s: "policy")
    with rank_log(tmp_path / "rank.log"):
        with Progress(recipe, 0, 4, clock=clock) as p:
            for step in range(1, 5):
                update(p, clock, step)
            os.write(2, b"important warning\n")
    terminal = capfd.readouterr().err
    log = (tmp_path / "rank.log").read_bytes()
    assert "\r" in terminal
    assert b"\r" not in log
    assert b"important warning" in log and b"complete" in log
    assert b"training 2/4" not in log


def test_display_mode_does_not_change_training(monkeypatch, tmp_path, capsys):
    import json

    import torch
    from accelerate import Accelerator

    from lerobot_policy_vla_ttt.training.episode_parallel.config import Recipe
    from lerobot_policy_vla_ttt.training.episode_parallel.runner import run

    from .test_checkpoint import source
    from .test_engine import make_policy

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
    results = []
    for mode in ["off", "plain", "bar"]:
        torch.manual_seed(123)
        policy = make_policy("t2mem", monkeypatch)
        run(
            policy,
            source(),
            None,
            None,
            recipe=r,
            accelerator=Accelerator(cpu=True, mixed_precision="no"),
            output=tmp_path / mode,
            identity={"dataset": "fixture"},
            progress_mode=mode,
        )
        metrics = [
            json.loads(line)
            for line in (tmp_path / mode / "metrics.jsonl").read_text().splitlines()
        ]
        assert len(metrics) == 4
        assert (tmp_path / mode / "step-00000004/COMPLETE").exists()
        results.append((policy.state_dict(), [m["loss"] for m in metrics]))
    for weights, losses in results[1:]:
        torch.testing.assert_close(weights, results[0][0], rtol=0, atol=0)
        assert losses == results[0][1]
    assert "sampled_episode_ids" not in capsys.readouterr().out


def test_narrow_bar_stays_on_one_physical_line():
    p, terminal, log, clock = make(tty=True)
    p.output.columns = lambda: 80
    update(p, clock, 1)
    for frame in terminal.getvalue().split("\r"):
        assert len(frame) <= 79
    assert "recipe=20" in log.getvalue()


def test_diagnostic_log_failure_does_not_mask_training_error():
    p, terminal, log, clock = make()

    def broken(text):
        raise OSError("disk full")

    p.output.log = broken
    with pytest.raises(ValueError, match="training error"):
        with p:
            update(p, clock, 1)
            p.saving()
            p.saved()
            raise ValueError("training error")
