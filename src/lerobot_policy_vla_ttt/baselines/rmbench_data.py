"""Convert official RMBench episodes without changing control or RGB semantics."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import h5py
import numpy as np

from .rmbench import CAMERAS, DATA_REVISION, UPSTREAM_REVISION, load_codec, resolve_tasks


def episode_frames(path, instruction, decode):
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Missing task-level instruction")
    with h5py.File(path, "r") as f:
        if "joint_action/vector" in f:
            vectors = np.asarray(f["joint_action/vector"], dtype=np.float32)
            states, actions = vectors[:-1], vectors[1:]
            cameras = {target: f[f"observation/{source}/rgb"] for source, target in CAMERAS.items()}
            expected_camera_length = len(vectors)
        elif "state" in f and "action" in f:
            fields = (
                "left_arm_joint_states",
                "left_ee_joint_states",
                "right_arm_joint_states",
                "right_ee_joint_states",
            )

            def pack(group):
                parts = []
                for key, width in zip(fields, (6, 1, 6, 1), strict=True):
                    value = np.asarray(f[f"{group}/{key}"], dtype=np.float32)
                    if value.ndim == 1:
                        value = value[:, None]
                    if value.ndim != 2 or value.shape[1] != width:
                        raise ValueError(f"Unexpected {group}/{key} shape")
                    parts.append(value)
                return np.concatenate(parts, axis=1)

            states, actions = pack("state"), pack("action")
            cameras = {
                target: f[f"vision/{source}/colors"]
                for source, target in zip(
                    ("cam_head", "cam_left_wrist", "cam_right_wrist"), CAMERAS.values(), strict=True
                )
            }
            expected_camera_length = len(states)
        else:
            raise ValueError(f"Unsupported RMBench schema: {path}")
        if (
            states.ndim != 2
            or states.shape[1] != 14
            or states.shape != actions.shape
            or not len(states)
        ):
            raise ValueError("Expected nonempty aligned 14D states/actions")
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise ValueError("Nonfinite state/action")
        if any(len(camera) != expected_camera_length for camera in cameras.values()):
            raise ValueError("Camera/state length mismatch")
        for t in range(len(states)):
            frame = {
                "observation.state": states[t].copy(),
                "action": actions[t].copy(),
                "task": instruction,
            }
            for key, camera in cameras.items():
                image = decode(camera[t])
                if image.dtype != np.uint8 or image.shape != (240, 320, 3):
                    raise ValueError(f"Expected RGB 240x320 for {key}")
                frame[f"observation.images.{key}"] = image.copy()
            yield frame


def find_episodes(raw, task):
    candidates = [
        Path(raw) / "data" / task / "demo_clean",
        Path(raw) / "demo_clean" / task / "aloha_agilex",
    ]
    directories = [p for p in candidates if (p / "data").is_dir()]
    if len(directories) != 1:
        raise ValueError(f"Expected exactly one official data layout for {task}")
    base = directories[0]
    files = sorted(
        (base / "data").glob("episode*.hdf5"),
        key=lambda p: int(re.search(r"(\d+)\.hdf5$", p.name)[1]),
    )
    if not files:
        raise ValueError(f"No episodes for {task}")
    return base, files


def task_instruction(base, episode):
    for directory in ("instructions", "instruction"):
        path = base / directory / (episode.stem + ".json")
        if path.is_file():
            value = json.loads(path.read_text())
            values = value.get("seen") if isinstance(value, dict) else value
            if (
                not isinstance(values, list)
                or not values
                or not isinstance(values[0], str)
                or not values[0].strip()
            ):
                raise ValueError(f"Invalid task-level instruction file {path}")
            return values[0]
    with h5py.File(episode) as f:
        if "instructions" in f:
            values = json.loads(f["instructions"][()])
            if (
                isinstance(values, list)
                and values
                and isinstance(values[0], str)
                and values[0].strip()
            ):
                return values[0]
    raise ValueError(f"Missing official instruction for {episode}")


def prepare(source, raw, output, tasks, *, max_episodes=None):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    tasks = resolve_tasks(tasks)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if max_episodes is not None and max_episodes <= 0:
        raise ValueError("max-episodes must be positive")
    codec = load_codec(source)
    features = {
        **{
            f"observation.images.{name}": {
                "dtype": "image",
                "shape": (240, 320, 3),
                "names": ["height", "width", "channels"],
            }
            for name in CAMERAS.values()
        },
        "observation.state": {"dtype": "float32", "shape": (14,), "names": ["state"]},
        "action": {"dtype": "float32", "shape": (14,), "names": ["action"]},
    }
    # Upstream frequency=15 is an indexing convention, not a resampling operation.
    dataset = LeRobotDataset.create(
        "local/rmbench",
        fps=15,
        root=output,
        features=features,
        robot_type="aloha_agilex",
        use_videos=False,
    )
    mappings = []
    try:
        for task in tasks:
            base, files = find_episodes(raw, task)
            for path in files[:max_episodes]:
                instruction = task_instruction(base, path)
                count = 0
                for frame in episode_frames(path, instruction, codec.decode_image_bit):
                    dataset.add_frame(frame)
                    count += 1
                dataset.save_episode()
                with path.open("rb") as stream:
                    sha = hashlib.file_digest(stream, "sha256").hexdigest()
                mappings.append(
                    {
                        "episode": len(mappings),
                        "task": task,
                        "source": str(path.relative_to(raw)),
                        "sha256": sha,
                        "frames": count,
                        "instruction": instruction,
                    }
                )
    finally:
        dataset.finalize()
    manifest = {
        "benchmark": "rmbench",
        "protocol": "aloha-joint-next-state-v1",
        "upstream_revision": UPSTREAM_REVISION,
        "reference_data_revision": DATA_REVISION,
        "tasks": tasks,
        "episodes": mappings,
        "partial": max_episodes is not None,
        "camera_order": list(CAMERAS.values()),
        "fps_indexing": 15,
    }
    (output / "preparation.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    from ..portability import read_machine

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--machine", type=Path, required=True)
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-episodes", type=int)
    args = p.parse_args()
    assets = read_machine(args.machine)["assets"]
    print(
        json.dumps(
            prepare(
                assets["rmbench_source"],
                Path(assets["rmbench_raw"]),
                args.output,
                args.tasks,
                max_episodes=args.max_episodes,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
