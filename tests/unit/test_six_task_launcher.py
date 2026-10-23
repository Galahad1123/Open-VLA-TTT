import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/train_robomme_six_tasks.py"


def module():
    spec = importlib.util.spec_from_file_location("six_task_launcher", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def machine(tmp_path):
    p = tmp_path / "machine.toml"
    p.write_text('[storage]\nruns_root="runs"\n')
    return p


def test_plan_resolves_machine_relative_paths_and_stage_handoff(tmp_path):
    m = module()
    plan = m.build_plan(machine(tmp_path), "trial")
    assert plan[0]["output"] == tmp_path / "runs/trial-stage1"
    assert "--fresh" in plan[0]["command"] and "--fresh" not in plan[1]["command"]
    cmd = plan[1]["command"]
    assert cmd[cmd.index("--checkpoint") + 1] == str(
        tmp_path / "runs/trial-stage1/checkpoints/000750/pretrained_model"
    )


def test_failed_stage1_never_starts_stage2(tmp_path):
    m = module()
    plan = m.build_plan(machine(tmp_path), "trial")
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        raise subprocess.CalledProcessError(7, cmd)

    with pytest.raises(subprocess.CalledProcessError):
        m.execute(plan, runner=run)
    assert len(calls) == 1


def test_missing_checkpoint_stops_handoff(tmp_path):
    m = module()
    plan = m.build_plan(machine(tmp_path), "trial")
    calls = []
    with pytest.raises(RuntimeError):
        m.execute(plan, runner=lambda cmd, **kwargs: calls.append(cmd))
    assert len(calls) == 1


def test_successful_handoff_and_no_overwrite(tmp_path):
    m = module()
    plan = m.build_plan(machine(tmp_path), "trial")
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        if len(calls) == 1:
            out = plan[0]["output"]
            cp = out / "checkpoints/000750/pretrained_model"
            cp.mkdir(parents=True)
            for name in m.REQUIRED_CHECKPOINT_FILES:
                (cp / name).write_text("{}")
            (out / "summary.json").write_text(json.dumps({"status": "complete", "updates": 750}))

    m.execute(plan, runner=run)
    assert calls == [x["command"] for x in plan]
    with pytest.raises(FileExistsError):
        m.build_plan(tmp_path / "machine.toml", "trial")


def test_dry_run_never_launches_or_creates_outputs(tmp_path, capsys):
    m = module()
    m.main(["--machine", str(machine(tmp_path)), "--run-prefix", "trial", "--dry-run"])
    assert not (tmp_path / "runs").exists()
    assert "--fresh" in capsys.readouterr().out
