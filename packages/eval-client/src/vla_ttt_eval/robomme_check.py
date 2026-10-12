"""Offline metadata audit and optional one-step simulation probes (not evaluation)."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from .robomme import MME_VLA_MAX_STEPS, TASKS, RoboMMEEnvironment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--simulate", action="store_true")
    args = parser.parse_args()
    from robomme.env_record_wrapper import BenchmarkEnvBuilder

    if BenchmarkEnvBuilder.get_task_list() != TASKS:
        raise ValueError("Official task list differs from pinned protocol")
    report = {
        "kind": "preflight_not_success_rate",
        "max_steps": MME_VLA_MAX_STEPS,
        "versions": {
            key: importlib.metadata.version(key)
            for key in ("robomme", "mani-skill", "sapien", "torch")
        },
        "metadata": [],
        "simulation": [],
    }
    for split, count in (("train", 100), ("val", 50), ("test", 50)):
        for task in TASKS:
            builder = BenchmarkEnvBuilder(task, dataset=split)
            ids = {ep for name, ep in builder.metadata_index if name == task}
            if ids != set(range(count)):
                raise ValueError(f"Invalid episode IDs: {task}/{split}")
            if any(builder.resolve_episode(ep)[0] is None for ep in ids):
                raise ValueError(f"Missing environment seed: {task}/{split}")
            payload = json.dumps(sorted(builder.metadata_index.items()), sort_keys=True).encode()
            report["metadata"].append(
                {
                    "task": task,
                    "split": split,
                    "episodes": count,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Persist each successful probe so an interrupted run retains evidence.
    with args.output.open("x") as stream:
        if args.simulate:
            for task in TASKS:
                env = RoboMMEEnvironment(task, "test", MME_VLA_MAX_STEPS)
                try:
                    obs = env.reset(0, 1000)
                    for image in obs["images"].values():
                        assert image.shape == (256, 256, 3) and image.dtype == np.uint8
                    assert obs["state"].shape == (8,) and np.isfinite(obs["state"]).all()
                    initial_steps = int(env.env.steps_without_demonstration)
                    action = obs["state"].copy()
                    action[-1] = 1.0  # Open gripper, not a learned policy.
                    _, success, done = env.step(action)
                    report["simulation"].append(
                        {
                            "task": task,
                            "episode": 0,
                            "env_seed": env.env_seed,
                            "initial_internal_steps": initial_steps,
                            "internal_step_limit": int(env.env.max_steps_without_demonstration),
                            "success": bool(success),
                            "done": bool(done),
                        }
                    )
                    print(f"PASS reset/render/step {task}", flush=True)
                finally:
                    env.close()
                stream.seek(0)
                json.dump(report, stream, indent=2)
                stream.truncate()
                stream.flush()
        else:
            json.dump(report, stream, indent=2)


if __name__ == "__main__":
    main()
