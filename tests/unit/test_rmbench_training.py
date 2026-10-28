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


@pytest.mark.parametrize("world_size", [1, 2])
@pytest.mark.parametrize("full_dataset", [False, True])
def test_recipe_checkpoint_interval_reaches_both_stages(
    tmp_path, monkeypatch, capsys, full_dataset, world_size
):
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
                "tasks": ["observe_and_pickup", "rearrange_blocks"]
                if full_dataset
                else ["rearrange_blocks"],
                "upstream_revision": training.UPSTREAM_REVISION,
                "partial": False,
                "episodes": [
                    dict(episode=0, task="observe_and_pickup"),
                    dict(episode=1, task="rearrange_blocks"),
                ],
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
            "--num-processes",
            str(world_size),
            "--dry-run",
        ],
    )
    training.main()
    output = capsys.readouterr().out
    assert f"--steps={5000 // world_size}" in output
    assert f"--steps={80000 // world_size}" in output
    assert f'"global_batch_size": {8 * world_size}' in output
    assert not (tmp_path / "runs").exists()
    if full_dataset:
        assert output.count("--dataset.episodes=[1]") == 2
    else:
        assert "--dataset.episodes=" not in output


def test_select_episodes_from_full_dataset():
    from lerobot_policy_vla_ttt.training.train_rmbench import select_episodes

    preparation = {
        "tasks": ["a", "b", "c"],
        "episodes": [dict(episode=i, task=t) for i, t in enumerate(["a", "b", "a", "c"])],
    }
    assert select_episodes(preparation, ["b", "c"]) == [1, 3]
    assert select_episodes(preparation, ["a", "b", "c"]) is None
    with pytest.raises(ValueError, match="missing"):
        select_episodes(preparation, ["d"])
    with pytest.raises(ValueError, match="episodes"):
        select_episodes(dict(preparation, episodes=[]), ["a"])


def test_subset_sampler_maps_noncontiguous_episodes_and_resumes():
    # Original frames 10..19 and 30..129 become compact indices 0..109.
    kwargs = dict(
        task_by_episode=["a", "b", "a", "c"],
        episode_indices_to_use=[1, 3],
        absolute_to_relative_idx={
            **{i: i - 10 for i in range(10, 20)},
            **{i: i - 20 for i in range(30, 130)},
        },
        seed=8,
    )
    sampler = BalancedTaskSampler([0, 10, 20, 30], [10, 20, 30, 130], **kwargs)
    values = list(sampler)
    assert len(values) == 110
    assert all(0 <= i < 110 for i in values)
    assert 35 < sum(i < 10 for i in values) < 75
    resumed = BalancedTaskSampler([0, 10, 20, 30], [10, 20, 30, 130], **kwargs)
    resumed.load_state_dict({"epoch": 0, "start_index": 21})
    assert list(resumed) == values[21:]


def test_lerobot_subset_keeps_source_stats_and_action_boundaries(tmp_path):
    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = tmp_path / "dataset"
    features = {
        name: {"dtype": "float32", "shape": (1,), "names": ["value"]}
        for name in ["observation.state", "action"]
    }
    writer = LeRobotDataset.create(
        "local/test", root=root, fps=10, features=features, use_videos=False
    )
    for episode in range(4):
        for frame in range(3):
            value = np.array([episode * 100 + frame], dtype=np.float32)
            writer.add_frame({"observation.state": value, "action": value, "task": str(episode)})
        writer.save_episode()
    writer.finalize()
    before = (root / "meta/stats.json").read_bytes()
    selected = LeRobotDataset(
        "local/test", root=root, episodes=[1, 3], delta_timestamps={"action": [0, 0.1, 0.2]}
    )
    sampler = BalancedTaskSampler(
        selected.meta.episodes["dataset_from_index"],
        selected.meta.episodes["dataset_to_index"],
        task_by_episode=["a", "b", "a", "c"],
        episode_indices_to_use=selected.episodes,
        absolute_to_relative_idx=selected.absolute_to_relative_idx,
    )
    assert len(selected) == 6
    assert {selected[i]["episode_index"].item() for i in sampler} <= {1, 3}
    # The final action chunk pads within the selected episode, never entering the next.
    assert selected[2]["action"].flatten().tolist() == [102, 102, 102]
    assert selected[5]["action"].flatten().tolist() == [302, 302, 302]
    assert (root / "meta/stats.json").read_bytes() == before
    full = LeRobotDataset("local/test", root=root)
    np.testing.assert_array_equal(
        selected.meta.stats["action"]["mean"], full.meta.stats["action"]["mean"]
    )


def test_budget_uses_global_batch_and_fixed_steps():
    from lerobot_policy_vla_ttt.training.train_rmbench import stage_budget

    assert training_budget(40000, 8, 9, world_size=2) == 22500
    assert stage_budget({"batch_size": 4, "steps": 75000}, 9, 2) == (4, 75000)
    assert stage_budget({"batch_size": 4, "chunks_per_task": 320000}, 9, 2) == (4, 360000)
    assert stage_budget({"batch_size": 4, "steps": 75000}, 9, 2, smoke_steps=20) == (1, 20)
    assert stage_budget(
        {"batch_size": 4, "steps": 75000}, 9, 2, smoke_steps=20, smoke_use_recipe_batch=True
    ) == (4, 20)
    for settings in (
        {"batch_size": 4, "steps": 0},
        {"batch_size": 4, "steps": 5, "chunks_per_task": 20},
    ):
        with pytest.raises(ValueError):
            stage_budget(settings, 9, 2)
    with pytest.raises(ValueError):
        training_budget(40000, 8, 9, world_size=0)


def test_balanced_sampler_accelerate_shards_and_resumes_global_stream():
    from accelerate.data_loader import BatchSamplerShard
    from torch.utils.data import BatchSampler

    def stream(rank, start=0):
        sampler = BalancedTaskSampler([0, 40], [40, 400], task_by_episode=["a", "b"], seed=8)
        sampler.load_state_dict({"epoch": 0, "start_index": start})
        return list(
            BatchSamplerShard(
                BatchSampler(sampler, batch_size=4, drop_last=False),
                num_processes=2,
                process_index=rank,
            )
        )

    full = list(BalancedTaskSampler([0, 40], [40, 400], task_by_episode=["a", "b"], seed=8))
    ranks = [stream(0), stream(1)]
    combined = [x for pair in zip(*ranks, strict=True) for batch in pair for x in batch]
    assert combined == full  # Shard sample positions once; replacement can repeat frame IDs.
    for rank in range(2):
        assert stream(rank, start=80) == ranks[rank][10:]
        assert 60 < sum(x < 40 for batch in ranks[rank] for x in batch) < 140


@pytest.mark.parametrize("frames", [400, 399])
def test_prepared_loader_resumes_after_epoch_zero_including_padding(frames):
    import torch
    from accelerate.data_loader import prepare_data_loader
    from lerobot.datasets.sampler import compute_sampler_state
    from torch.utils.data import DataLoader

    def loader(rank, step=None):
        sampler = BalancedTaskSampler(
            [0, 40], [40, frames], task_by_episode=["a", "b"], seed=8, global_batch_size=8
        )
        if step is not None:
            sampler.load_state_dict(compute_sampler_state(step, frames, 4, 2))
        return prepare_data_loader(
            DataLoader(list(range(frames)), batch_size=4, sampler=sampler),
            device=torch.device("cpu"),
            num_processes=2,
            process_index=rank,
            rng_types=[],
        )

    for rank in range(2):
        full = loader(rank)
        list(full)  # Epoch 0.
        epoch1 = [batch.tolist() for batch in full]
        epoch2 = [batch.tolist() for batch in full]
        resumed = loader(rank, step=60)  # 50 updates/epoch, resume ten updates into epoch 1.
        assert [batch.tolist() for batch in resumed] == epoch1[10:]
        assert [batch.tolist() for batch in resumed] == epoch2
