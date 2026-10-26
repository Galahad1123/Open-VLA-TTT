import pytest

from lerobot_policy_vla_ttt.training.train_rmbench import BalancedTaskSampler, training_budget


def test_per_task_budget():
    assert training_budget(40000, 8, 1) == 5000
    assert training_budget(40000, 8, 9) == 45000
    with pytest.raises(ValueError):
        training_budget(0, 8, 1)


def test_balanced_tasks_and_resume():
    sampler = BalancedTaskSampler([0, 10], [10, 1010], task_by_episode=["a", "b"], seed=8)
    values = list(sampler)
    assert len(values) == 1010
    assert 450 < sum(x < 10 for x in values) < 560
    resumed = BalancedTaskSampler([0, 10], [10, 1010], task_by_episode=["a", "b"], seed=8)
    resumed.load_state_dict({"epoch": 0, "start_index": 64})
    assert list(resumed) == values[64:]
    assert min(values) >= 0 and max(values) < 1010
    assert len(set(values)) > 100


def test_sampler_refuses_unknown_subset():
    with pytest.raises(ValueError):
        BalancedTaskSampler([0], [10], task_by_episode=["a"], episode_indices_to_use=[1])


def test_checkpoint_lineage(tmp_path):
    import json

    from lerobot_policy_vla_ttt.training.train_rmbench import validate_lineage

    identity = {
        "stage": "stage2",
        "tasks": ["a"],
        "recipe": {"lr": 1},
        "seed": 1,
        "preparation_sha256": "abc",
        "steps": 8,
        "batch_size": 4,
    }
    with pytest.raises((ValueError, FileNotFoundError)):
        validate_lineage(tmp_path, identity, resume=False)
    saved = dict(identity, stage="stage1", steps=2, batch_size=8)
    (tmp_path / "rmbench_run.json").write_text(json.dumps(saved))
    validate_lineage(tmp_path, identity, resume=False)
    with pytest.raises(ValueError):
        validate_lineage(tmp_path, dict(identity, preparation_sha256="other"), resume=False)
    with pytest.raises(ValueError):
        validate_lineage(tmp_path, identity, resume=True)


def test_recipe_checkpoint_interval_reaches_both_stages(tmp_path, monkeypatch, capsys):
    import json
    import sys

    from lerobot_policy_vla_ttt.training import train_rmbench as training

    vlm = tmp_path / "vlm"
    vlm.mkdir()
    (vlm / "config.json").write_text("{}")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "preparation.json").write_text(
        json.dumps(
            {
                "tasks": ["rearrange_blocks"],
                "upstream_revision": training.UPSTREAM_REVISION,
                "partial": False,
                "episodes": [],
            }
        )
    )
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "chunk_size": 50,
                "n_action_steps": 10,
                "action_lr": 1e-4,
                "vlm_lr": 1e-5,
                "save_freq": 5000,
                "stage1": {"batch_size": 8, "chunks_per_task": 40000},
                "stage2": {"batch_size": 4, "chunks_per_task": 320000},
            }
        )
    )
    monkeypatch.setattr(
        training,
        "read_machine",
        lambda _: {
            "assets": {"vlm": str(vlm), "rmbench_dataset": str(dataset)},
            "storage": {"runs_root": str(tmp_path / "runs")},
        },
    )
    monkeypatch.setattr(training, "configure_cache", lambda *a: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--machine",
            "unused",
            "--recipe",
            str(recipe),
            "--tasks",
            "rearrange_blocks",
            "--mode",
            "single",
            "--run-name",
            "test",
            "--dry-run",
        ],
    )
    training.main()
    output = capsys.readouterr().out
    assert output.count("--save_freq=5000") == 2
    assert "--steps=5000" in output
    assert "--steps=80000" in output
