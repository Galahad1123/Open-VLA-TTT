"""Canonicalize downloaded RoboMME training episodes using the LeRobot writer."""

import argparse
import json
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ROBOMME_REVISION = "92796abbc67db4ea6447e5031d04bf47aa3770bb"


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def control_frame(frame):
    if bool(_numpy(frame["is_demo"]).item()):
        return None
    result = {"task": frame["task"]}
    for source, target in (("state", "observation.state"), ("actions", "action")):
        value = _numpy(frame[source]).astype(np.float32)
        if value.shape != (8,) or not np.isfinite(value).all():
            raise ValueError(f"Invalid RoboMME {source}: expected finite shape (8,)")
        result[target] = value
    for source, target in (("image", "image"), ("wrist_image", "image2")):
        value = _numpy(frame[source])
        if value.shape != (3, 256, 256) or value.dtype != np.uint8:
            raise ValueError("Expected LeRobot uint8 CHW camera frames")
        result[f"observation.images.{target}"] = value.transpose(1, 2, 0).copy()
    return result


def prepare(source_root, output, episodes, revision=ROBOMME_REVISION):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    source = LeRobotDataset(
        "lerobot/robomme",
        root=source_root,
        revision=revision,
        episodes=episodes,
        return_uint8=True,
        video_backend="pyav",
    )
    features = {
        **{
            f"observation.images.{key}": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channels"],
            }
            for key in ("image", "image2")
        },
        "observation.state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
        "action": {"dtype": "float32", "shape": (8,), "names": ["action"]},
    }
    # Lossless image-backed output: no re-encoding loss, no external video tools required.
    target = LeRobotDataset.create(
        "local/robomme-evo1",
        fps=source.meta.fps,
        root=output,
        features=features,
        robot_type="panda",
        use_videos=False,
    )
    current = None
    kept = 0
    total_kept = 0
    mappings = []
    try:
        for index in range(len(source)):
            frame = source[index]
            episode = int(frame["episode_index"].item())
            if episode != current:
                if current is not None:
                    if not kept:
                        raise ValueError(f"Episode {current} has no control frames")
                    target.save_episode()
                    mappings.append({"source_episode": current, "control_frames": kept})
                current, kept = episode, 0
            canonical = control_frame(frame)
            if canonical is not None:
                target.add_frame(canonical)
                kept += 1
                total_kept += 1
        if not kept:
            raise ValueError("No final control episode")
        target.save_episode()
        mappings.append({"source_episode": current, "control_frames": kept})
    finally:
        target.finalize()
    manifest = {
        "source_repo": "lerobot/robomme",
        "source_revision": revision,
        "source_root": str(source_root),
        "source_episodes": episodes,
        "episode_mapping": mappings,
        "control_frames": total_kept,
        "excluded": ["is_demo=true", "all privileged subgoal columns"],
        "observation_mode": "latest_frame",
        "action_space": "joint_angle",
    }
    (output / "preparation.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", nargs="+", type=int, required=True)
    args = parser.parse_args()
    print(prepare(args.source_root, args.output, args.episodes))


if __name__ == "__main__":
    main()
