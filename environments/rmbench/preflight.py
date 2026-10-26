"""Exercise the actual pinned RMBench environment and official expert."""

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from vla_ttt_eval.rmbench import UPSTREAM_REVISION, canonical_observation, resolve_tasks
from vla_ttt_eval.rmbench_runner import task_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--task", default="put_back_block")
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    resolve_tasks([args.task])
    runtime, output = args.runtime.resolve(), args.output.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(runtime), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError("Runtime revision differs")
    if output.exists():
        raise FileExistsError(output)
    os.chdir(runtime)
    sys.path.insert(0, str(runtime))
    module = importlib.import_module("envs." + args.task)
    env = getattr(module, args.task)()
    started = time.monotonic()
    try:
        env.setup_demo(
            now_ep_num=0, seed=args.seed, is_test=True, **task_config(runtime, args.task)
        )
        obs = canonical_observation(env.get_obs(), "Preflight observation")
        output.parent.mkdir(parents=True, exist_ok=True)
        image_stats = {}
        for name, frame in obs["images"].items():
            Image.fromarray(frame).save(output.with_name(output.stem + "-" + name + ".png"))
            image_stats[name] = {
                "min": int(frame.min()),
                "max": int(frame.max()),
                "mean": float(frame.mean()),
                "std": float(frame.std()),
            }
        print(
            "OBSERVATION OK",
            obs["state"].shape,
            {k: v.shape for k, v in obs["images"].items()},
            flush=True,
        )
        env.play_once()
        success = bool(env.plan_success and env.check_success())
        result = {
            "seed": args.seed,
            "task": args.task,
            "expert_success": success,
            "plan_success": bool(env.plan_success),
            "elapsed_seconds": time.monotonic() - started,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "upstream_revision": revision,
            "state_finite": bool(np.isfinite(obs["state"]).all()),
            "image_stats": image_stats,
            "camera_shapes": {k: list(v.shape) for k, v in obs["images"].items()},
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
    finally:
        env.close_env()


if __name__ == "__main__":
    main()
