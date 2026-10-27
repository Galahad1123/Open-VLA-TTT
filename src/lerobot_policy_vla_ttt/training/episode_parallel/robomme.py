"""Chronological RoboMME view retaining the unsupervised demonstration prefix.

Unlike the control baseline, indices are never compressed. Numeric metadata is
validated before decoding; images use the baseline's uint8 camera convention.
"""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ...baselines.prepare import ROBOMME_REVISION
from ...baselines.robomme_data import KEYS, inspect_source
from .data import Episode


def load_episode_tasks(root):
    """Recover one instruction per episode; RoboMME episode metadata omits tasks."""
    root = Path(root)
    tasks = {
        int(row["task_index"]): str(row["task"])
        for row in pq.read_table(root / "meta/tasks.parquet").to_pylist()
    }
    table = pa.concat_tables(
        [
            pq.read_table(path, columns=["episode_index", "task_index"])
            for path in sorted(root.glob("data/*/*.parquet"))
        ]
    )
    pairs = table.group_by(["episode_index", "task_index"]).aggregate([]).to_pylist()
    result = {}
    for row in pairs:
        episode, task = int(row["episode_index"]), int(row["task_index"])
        if episode in result or task not in tasks:
            raise ValueError("Expected one known task per episode")
        result[episode] = tasks[task]
    return result


class RoboMMEMemoryDataset:
    def __init__(self, root, *, horizon, revision=ROBOMME_REVISION, repo_id="lerobot/robomme"):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        if horizon < 1:
            raise ValueError("horizon must be positive")
        _, lengths, _, self.report = inspect_source(root)
        self.source = LeRobotDataset(
            repo_id,
            root=root,
            revision=revision,
            video_backend="pyav",
            return_uint8=True,
            delta_timestamps={"actions": [i / 10 for i in range(horizon)]},
        )
        tasks = load_episode_tasks(root)
        self.episodes = []
        for row in self.source.meta.episodes:
            index = int(row["episode_index"])
            end = int(row["dataset_to_index"])
            self.episodes.append(
                Episode(
                    index,
                    tasks[index],
                    int(row["dataset_from_index"]),
                    end,
                    execution_start=end - int(lengths[index]),
                )
            )

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        frame = self.source[index]
        return {
            **{target: frame[key] for key, target in KEYS.items()},
            "action_is_pad": frame["actions_is_pad"],
            "task": frame["task"],
        }


def select_task_subset(episodes, task_groups):
    """Change sampler labels only; dataset language prompts remain unchanged."""
    from dataclasses import replace

    if task_groups is None:
        return episodes
    lookup = {e.index: e for e in episodes}
    result, seen = [], set()
    for task, ids in task_groups.items():
        for index in ids:
            if index not in lookup or index in seen:
                raise ValueError(f"Unknown or duplicate subset episode: {index}")
            seen.add(index)
            result.append(replace(lookup[index], task=task))
    return result
