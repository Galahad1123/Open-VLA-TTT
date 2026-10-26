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
