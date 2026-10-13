import json

import pytest
import torch

from lerobot_policy_vla_ttt.training.episode_parallel.data import Episode, EpisodeSource


def test_demo_prefix_has_no_labels_and_execution_has_its_own_clock():
    frames = [
        {"observation.state": torch.tensor([i]), "action": torch.ones(5, 8)} for i in range(20)
    ]
    source = EpisodeSource(
        frames,
        [Episode(0, "move", 2, 20, execution_start=9)],
        execution=4,
        horizon=5,
        demo_stride=3,
    )
    events = list(source.events(0, seed=0))
    assert [e.frame["observation.state"].item() for e in events] == [2, 5, 8, 9, 11, 13, 15, 17, 19]
    assert [e.supervised for e in events] == [
        False,
        False,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
    ]
    assert source.decision_count(0) == 3
    assert all("action" not in e.frame for e in events if not e.supervised)
    assert events[-2].frame["action_mask"].sum() == 24


@pytest.mark.parametrize("boundary", [1, 20, 21])
def test_invalid_execution_boundary_rejected(boundary):
    with pytest.raises(ValueError, match="execution"):
        EpisodeSource(
            [{}] * 20, [Episode(0, "move", 2, 20, execution_start=boundary)], execution=4, horizon=5
        )


def test_robomme_cli_dry_run_uses_raw_actions_and_checkpoint_horizon(tmp_path, capsys):
    from lerobot.configs.types import FeatureType, PolicyFeature

    from lerobot_policy_vla_ttt.models.evo1_groot.configuration_evo1_groot import Evo1GrootConfig
    from lerobot_policy_vla_ttt.training.train_episode_parallel import main

    checkpoint = tmp_path / "base"
    Evo1GrootConfig(
        device="cpu",
        chunk_size=20,
        n_action_steps=10,
        output_features={"action": PolicyFeature(FeatureType.ACTION, (8,))},
    ).save_pretrained(checkpoint)
    (checkpoint / "train_config.json").write_text(
        json.dumps({"dataset": {"image_transforms": {"enable": False}}})
    )
    data = tmp_path / "data"
    (data / "meta").mkdir(parents=True)
    (data / "meta/info.json").write_text(
        json.dumps({"fps": 10, "features": {"actions": {"shape": [8]}}})
    )
    main(
        [
            "--benchmark",
            "robomme",
            "--checkpoint",
            str(checkpoint),
            "--dataset-root",
            str(data),
            "--output",
            str(tmp_path / "out"),
            "--dry-run",
        ]
    )
    result = json.loads(capsys.readouterr().out)
    assert result["recipe"]["horizon"] == 20
    assert result["identity"]["dataset"]["repo"] == "lerobot/robomme"
    assert result["identity"]["sequence_protocol"]["demo_stride"] == 5
    assert not (tmp_path / "out").exists()


def test_raw_view_keeps_demo_and_global_indices(monkeypatch):
    from types import SimpleNamespace

    import numpy as np
    from lerobot.datasets import lerobot_dataset

    from lerobot_policy_vla_ttt.training.episode_parallel import robomme

    monkeypatch.setattr(robomme, "inspect_source", lambda _: (None, np.array([3]), {}, {}))
    monkeypatch.setattr(robomme, "load_episode_tasks", lambda _: {0: "move"})

    class Raw:
        def __init__(self, repo_id, **kwargs):
            assert kwargs["delta_timestamps"] == {"actions": [0, 0.1, 0.2, 0.3, 0.4]}
            self.meta = SimpleNamespace(
                episodes=[
                    dict(episode_index=0, tasks=["move"], dataset_from_index=0, dataset_to_index=5)
                ]
            )

        def __len__(self):
            return 5

        def __getitem__(self, i):
            return dict(
                image=torch.zeros(3, 8, 8, dtype=torch.uint8),
                wrist_image=torch.zeros(3, 8, 8, dtype=torch.uint8),
                state=torch.full((8,), i),
                actions=torch.ones(5, 8),
                actions_is_pad=torch.tensor([False] + [True] * 4),
                task="move",
                is_demo=i < 2,
                exec_start_idx=2,
            )

    monkeypatch.setattr(lerobot_dataset, "LeRobotDataset", Raw)
    view = robomme.RoboMMEMemoryDataset("unused", horizon=5)
    assert len(view) == 5
    assert view.episodes[0].execution_start == 2
    source = EpisodeSource(view, view.episodes, execution=2, horizon=5)
    events = list(source.events(0, seed=1))
    assert [e.supervised for e in events] == [False, False, True, False, True]
    assert events[0].frame["observation.state"].tolist() == [[0] * 8]
    assert set(events[0].frame) == {
        "task",
        "observation.state",
        "observation.images.image",
        "observation.images.image2",
    }
    assert events[-1].frame["action_mask"].sum() == 8


def test_tasks_loaded_from_numeric_frames_without_episode_tasks(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from lerobot_policy_vla_ttt.training.episode_parallel.robomme import load_episode_tasks

    (tmp_path / "meta").mkdir()
    (tmp_path / "data/chunk-000").mkdir(parents=True)
    pq.write_table(
        pa.table({"task_index": [0, 1], "task": ["a", "b"]}), tmp_path / "meta/tasks.parquet"
    )
    path = tmp_path / "data/chunk-000/file-000.parquet"
    pq.write_table(pa.table({"episode_index": [1, 0, 1], "task_index": [1, 0, 1]}), path)
    assert load_episode_tasks(tmp_path) == {0: "a", 1: "b"}
    pq.write_table(pa.table({"episode_index": [1, 0, 1], "task_index": [1, 0, 0]}), path)
    with pytest.raises(ValueError, match="task"):
        load_episode_tasks(tmp_path)
