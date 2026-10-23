"""Offline, execution-only view of the original RoboMME LeRobot videos.

No video rewrite or mutation of source metadata. The source dataset retains its
original timeline for action queries; only the training sampler sees compressed
execution-frame indices. Demo frames remain available for future memory models.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from torch.utils.data import Dataset

from .prepare import ROBOMME_REVISION

KEYS = {
    "image": "observation.images.image",
    "wrist_image": "observation.images.image2",
    "state": "observation.state",
    "actions": "action",
}


def inspect_source(root):
    """Validate all numeric rows, demo boundaries and referenced local files."""
    root = Path(root)
    info = json.loads((root / "meta/info.json").read_text())
    if info["fps"] != 10 or info["splits"] != {"train": "0:1600"} or info["total_episodes"] != 1600:
        raise ValueError("Expected the pinned 1600-episode, 10 Hz RoboMME training release")
    columns = [
        "index",
        "episode_index",
        "frame_index",
        "is_demo",
        "exec_start_idx",
        "state",
        "actions",
    ]
    paths = sorted(root.glob("data/*/*.parquet"))
    table = pa.concat_tables([pq.read_table(path, columns=columns) for path in paths])
    table = table.take(pc.sort_indices(table, sort_keys=[("index", "ascending")]))
    indices = table["index"].to_numpy()
    if not np.array_equal(indices, np.arange(info["total_frames"])):
        raise ValueError("Missing, duplicated or non-contiguous frame indices")
    episodes = pa.concat_tables(
        [pq.read_table(p) for p in sorted(root.glob("meta/episodes/*/*.parquet"))]
    )
    episodes = episodes.take(pc.sort_indices(episodes, sort_keys=[("episode_index", "ascending")]))
    rows = episodes.to_pylist()
    if [row["episode_index"] for row in rows] != list(range(info["total_episodes"])):
        raise ValueError("Invalid episode metadata")
    demo = table["is_demo"].to_numpy()
    episode_ids = table["episode_index"].to_numpy()
    frame_ids = table["frame_index"].to_numpy()
    exec_starts = table["exec_start_idx"].to_numpy()
    videos = set()
    lengths = []
    previous_end = 0
    for row in rows:
        start, end = row["dataset_from_index"], row["dataset_to_index"]
        if start != previous_end or end - start != row["length"]:
            raise ValueError("Invalid episode boundaries")
        previous_end = end
        ep = row["episode_index"]
        local_demo = demo[start:end]
        n_demo = int(local_demo.sum())
        if not np.array_equal(local_demo, np.arange(end - start) < n_demo) or n_demo == end - start:
            raise ValueError(f"Episode {ep}: demo must be a prefix followed by execution")
        if not (
            np.all(episode_ids[start:end] == ep)
            and np.array_equal(frame_ids[start:end], np.arange(end - start))
            and np.all(exec_starts[start:end] == n_demo)
        ):
            raise ValueError(f"Episode {ep}: inconsistent frame/demo metadata")
        lengths.append(end - start - n_demo)
        data_path = root / info["data_path"].format(
            chunk_index=row["data/chunk_index"], file_index=row["data/file_index"]
        )
        if not data_path.is_file():
            raise FileNotFoundError(data_path)
        for key in ("image", "wrist_image"):
            path = root / info["video_path"].format(
                video_key=key,
                chunk_index=row[f"videos/{key}/chunk_index"],
                file_index=row[f"videos/{key}/file_index"],
            )
            if not path.is_file():
                raise FileNotFoundError(path)
            videos.add(str(path))
    if previous_end != len(table):
        raise ValueError("Episode metadata does not cover all frames")
    stats = {}
    for key in ("state", "actions"):
        values = np.asarray(table[key].to_pylist(), dtype=np.float32)
        if values.shape != (len(table), 8) or not np.isfinite(values).all():
            raise ValueError(f"Invalid {key} shape or nonfinite values")
        values = values[~demo]
        stats[KEYS[key]] = {
            "min": values.min(0),
            "max": values.max(0),
            "mean": values.mean(0),
            "std": values.std(0),
            "count": np.array([len(values)]),
        }
    report = {
        "source_root": str(root.resolve()),
        "source_revision": ROBOMME_REVISION,
        "episodes": len(rows),
        "frames": len(table),
        "demo_frames": int(demo.sum()),
        "execution_frames": int((~demo).sum()),
        "fps": info["fps"],
        "data_files": len(paths),
        "video_files": len(videos),
        "language_variants": info["total_tasks"],
        "execution_frames_per_episode_min": min(lengths),
        "execution_frames_per_episode_max": max(lengths),
        "action_min": stats["action"]["min"].tolist(),
        "action_max": stats["action"]["max"].tolist(),
        "video_validation": "referenced files exist; decoding requires separate probes",
        "revision_note": (
            "expected pinned revision; local file content is audited, not verified against Hub"
        ),
        "meta_info_sha256": hashlib.sha256((root / "meta/info.json").read_bytes()).hexdigest(),
    }
    return np.flatnonzero(~demo), np.asarray(lengths), stats, report


def select_execution_episodes(indices, lengths, episodes):
    """Select original episode IDs without changing source action-query boundaries."""
    ids = list(range(len(lengths))) if episodes is None else list(episodes)
    if (
        not ids
        or any(type(ep) is not int or not 0 <= ep < len(lengths) for ep in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("Expected unique, in-range integer episode IDs")
    ids.sort()
    ends = np.concatenate(([0], np.cumsum(lengths)))
    selected = np.concatenate([indices[ends[ep] : ends[ep + 1]] for ep in ids])
    return selected, lengths[ids], ids


class RoboMMEControlDataset(Dataset):
    def __init__(self, root, chunk_size=20, revision=ROBOMME_REVISION, episodes=None):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.indices, lengths, stats, self.report = inspect_source(root)
        self.indices, lengths, self.source_episode_ids = select_execution_episodes(
            self.indices, lengths, episodes
        )
        # Retain full execution-only normalization statistics across init comparisons.
        self.report.update(
            selected_episode_ids=self.source_episode_ids,
            selected_execution_frames=len(self.indices),
            normalization_scope="full_training_execution_frames",
        )
        self.source = LeRobotDataset(
            "lerobot/robomme",
            root=root,
            revision=revision,
            video_backend="pyav",
            return_uint8=True,
            delta_timestamps={"actions": [i / 10 for i in range(chunk_size)]},
        )
        self.meta = copy.copy(self.source.meta)
        self.meta.info = copy.deepcopy(self.source.meta.info)
        self.meta.info.features = {
            KEYS[key]: copy.deepcopy(self.source.meta.features[key]) for key in KEYS
        }
        self.meta.info.total_frames = len(self.indices)
        self.meta.info.total_episodes = len(lengths)
        self.meta.stats = stats
        ends = np.cumsum(lengths)
        self.meta.episodes = {
            "dataset_from_index": np.concatenate(([0], ends[:-1])).tolist(),
            "dataset_to_index": ends.tolist(),
        }
        self.episodes = None
        self.absolute_to_relative_idx = None
        self.num_frames = len(self.indices)
        self.num_episodes = len(lengths)
        self.fps = 10

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        frame = self.source[int(self.indices[index])]
        if bool(frame["is_demo"].item()):
            raise ValueError("Demo frame reached control supervision")
        return {
            **{KEYS[key]: frame[key] for key in KEYS},
            "action_is_pad": frame["actions_is_pad"],
            "task": frame["task"],
        }


def make_control_datasets(cfg):
    if cfg.dataset.streaming or cfg.dataset.eval_split != 0:
        raise ValueError("RoboMME baseline supports no streaming or custom split")
    if cfg.dataset.image_transforms.enable or cfg.rename_map or cfg.dataset.use_imagenet_stats:
        raise ValueError(
            "RoboMME control view handles naming; use no augmentation/rename/imagenet stats"
        )
    if cfg.dataset.root is None or cfg.dataset.repo_id != "lerobot/robomme":
        raise ValueError("Provide the local lerobot/robomme training root")
    if cfg.policy.type != "evo1_groot" or cfg.policy.n_obs_steps != 1:
        raise ValueError("This entry point is for the memoryless Evo1-GROOT baseline")
    return RoboMMEControlDataset(
        cfg.dataset.root, cfg.policy.chunk_size, cfg.dataset.revision, cfg.dataset.episodes
    ), None


def train():
    """Explicit RoboMME runner; keep ordinary LIBERO training untouched."""
    from importlib import import_module

    from lerobot_policy_vla_ttt.cli import local_processor_routing

    runner = import_module("lerobot.scripts.lerobot_train")
    evaluator = import_module("lerobot.scripts.lerobot_eval")
    original = runner.make_train_eval_datasets
    try:
        runner.make_train_eval_datasets = make_control_datasets
        with local_processor_routing(runner, evaluator):
            runner.main()
    finally:
        runner.make_train_eval_datasets = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--decode-probes", action="store_true")
    args = parser.parse_args()
    if args.decode_probes:
        view = RoboMMEControlDataset(args.root)
        report = view.report
        probes = []
        for ep in range(0, view.num_episodes, 100):
            start = view.meta.episodes["dataset_from_index"][ep]
            end = view.meta.episodes["dataset_to_index"][ep]
            for index in (start, (start + end) // 2, end - 1):
                frame = view[index]
                if frame["action"].shape != (20, 8):
                    raise ValueError("Action chunk query failed")
                for key in ("observation.images.image", "observation.images.image2"):
                    if frame[key].shape != (3, 256, 256):
                        raise ValueError("Decoded image shape mismatch")
                if index == end - 1 and frame["action_is_pad"].sum().item() != 19:
                    raise ValueError("Episode tail padding mask is invalid")
                probes.append({"episode": ep, "source_index": int(view.indices[index])})
        report["decode_probes"] = probes
        report["video_validation"] = (
            "48 training samples / 96 camera frames decoded; not full video scan"
        )
    else:
        report = inspect_source(args.root)[-1]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
