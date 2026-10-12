import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot_policy_vla_ttt.baselines.robomme_data import (
    RoboMMEControlDataset,
    inspect_source,
    make_control_datasets,
)


def test_execution_view_keeps_action_mask_and_drops_privileged_columns():
    view = object.__new__(RoboMMEControlDataset)
    view.indices = np.array([1])
    view.source = [
        None,
        {
            "image": torch.zeros(3, 256, 256, dtype=torch.uint8),
            "wrist_image": torch.zeros(3, 256, 256, dtype=torch.uint8),
            "state": torch.zeros(8),
            "actions": torch.zeros(20, 8),
            "actions_is_pad": torch.arange(20) > 0,
            "is_demo": torch.tensor(False),
            "task": "pick",
            "grounded_subgoal": "privileged secret",
        },
    ]
    row = view[0]
    assert set(row) == {
        "observation.images.image",
        "observation.images.image2",
        "observation.state",
        "action",
        "action_is_pad",
        "task",
    }
    assert row["action_is_pad"].sum() == 19
    assert row["action"].shape == (20, 8)
    view.source[1]["is_demo"] = torch.tensor(True)
    with pytest.raises(ValueError, match="Demo"):
        view[0]


def test_control_factory_refuses_resplitting_or_streaming():
    for streaming, episodes, split in [(True, None, 0), (False, [0], 0), (False, None, 0.1)]:
        cfg = SimpleNamespace(
            dataset=SimpleNamespace(streaming=streaming, episodes=episodes, eval_split=split)
        )
        with pytest.raises(ValueError, match="all training episodes"):
            make_control_datasets(cfg)


def test_control_view_sampler_does_not_sample_demo_holes():
    from lerobot.datasets.sampler import EpisodeAwareSampler

    # Source episodes [demo, exec, exec] and [demo, demo, exec].
    source_indices = np.array([1, 2, 5])
    sampler = EpisodeAwareSampler([0, 2], [2, 3], shuffle=True, seed=1000)
    assert sorted(source_indices[list(sampler)].tolist()) == [1, 2, 5]


@pytest.fixture
def raw_source(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    for directory in ("data/chunk-000", "meta/episodes/chunk-000", "videos"):
        (tmp_path / directory).mkdir(parents=True)
    info = {
        "fps": 10,
        "splits": {"train": "0:1600"},
        "total_episodes": 1600,
        "total_frames": 3200,
        "total_tasks": 16,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}.mp4",
    }
    (tmp_path / "meta/info.json").write_text(json.dumps(info))
    table = pa.table(
        {
            "index": np.arange(3200),
            "episode_index": np.repeat(np.arange(1600), 2),
            "frame_index": [0, 1] * 1600,
            "is_demo": [True, False] * 1600,
            "exec_start_idx": [1] * 3200,
            "state": [[99.0] * 8, [1.0] * 8] * 1600,
            "actions": [[99.0] * 8, [1.0] * 8] * 1600,
        }
    )
    pq.write_table(table, tmp_path / "data/chunk-000/file-000.parquet")
    episodes = [
        {
            "episode_index": ep,
            "length": 2,
            "dataset_from_index": 2 * ep,
            "dataset_to_index": 2 * ep + 2,
            "data/chunk_index": 0,
            "data/file_index": 0,
            **{
                f"videos/{key}/{field}": 0
                for key in ("image", "wrist_image")
                for field in ("chunk_index", "file_index")
            },
        }
        for ep in range(1600)
    ]
    pq.write_table(
        pa.Table.from_pylist(episodes), tmp_path / "meta/episodes/chunk-000/file-000.parquet"
    )
    for key in ("image", "wrist_image"):
        (tmp_path / f"videos/{key}.mp4").touch()
    return tmp_path


def test_full_audit_excludes_demo_from_statistics(raw_source):
    indices, lengths, stats, report = inspect_source(raw_source)
    np.testing.assert_array_equal(indices, np.arange(1, 3200, 2))
    assert np.all(lengths == 1)
    np.testing.assert_array_equal(stats["action"]["max"], np.ones(8))
    assert report["execution_frames"] == report["demo_frames"] == 1600
    assert report["video_files"] == 2


def test_full_audit_rejects_missing_video(raw_source):
    (raw_source / "videos/image.mp4").unlink()
    with pytest.raises(FileNotFoundError):
        inspect_source(raw_source)


def test_full_audit_rejects_duplicate_frame_indices(raw_source):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = raw_source / "data/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    table = table.set_column(table.schema.get_field_index("index"), "index", pa.array([0] * 3200))
    pq.write_table(table, path)
    with pytest.raises(ValueError, match="frame indices"):
        inspect_source(raw_source)
