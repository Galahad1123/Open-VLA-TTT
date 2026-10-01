"""Closed-loop evaluation in a simulator-only environment."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .client import PolicyClient


def run_episode(env, client, task, episode, seed, max_steps):
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    start = time.perf_counter()
    observation = env.reset(episode, seed)
    client.reset(f"{task}/{episode}", seed)
    latencies = []
    success, done = False, False
    for step in range(1, max_steps + 1):  # noqa: B007 - step is recorded after termination
        before = time.perf_counter()
        action = client.infer(observation)
        latencies.append(time.perf_counter() - before)
        observation, success, done = env.step(action)
        if success or done:
            break
    return {
        "task": task,
        "episode": episode,
        "policy_seed": seed,
        "success": bool(success),
        "steps": step,
        "termination": "success" if success else ("terminated" if done else "timeout"),
        "elapsed_s": time.perf_counter() - start,
        "rpc_latency_mean_s": float(np.mean(latencies)),
        "rpc_latency_p95_s": float(np.percentile(latencies, 95)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=["libero", "robomme"], required=True)
    parser.add_argument("--server", default="ws://127.0.0.1:8000")
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--libero-assets", type=Path)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes <= 0 or args.start_episode < 0:
        parser.error("episodes must be positive; start-episode must be nonnegative")
    if args.benchmark == "libero":
        if args.libero_assets is None:
            parser.error("--libero-assets must point to a downloaded pinned asset snapshot")
        from .libero_setup import configure

        configure(args.libero_assets)
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    from .libero import HORIZONS

    max_steps = args.max_steps or (HORIZONS[args.suite] if args.benchmark == "libero" else 300)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidentally overwriting prior results.
    with (
        args.output.open("x") as output,
        PolicyClient(args.server, benchmark=args.benchmark) as client,
    ):
        run = {
            "config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "policy": client.metadata,
        }
        output.write(json.dumps({"type": "run", **run}) + "\n")
        output.flush()
        for task in args.tasks:
            if args.benchmark == "libero":
                from .libero import LiberoEnvironment

                env = LiberoEnvironment(args.suite, task)
            else:
                from .robomme import RoboMMEEnvironment

                env = RoboMMEEnvironment(task, args.split, max_steps)
            try:
                for episode in range(args.start_episode, args.start_episode + args.episodes):
                    try:
                        result = run_episode(
                            env, client, task, episode, args.seed + episode, max_steps
                        )
                    except Exception as exc:
                        output.write(
                            json.dumps(
                                {
                                    "type": "error",
                                    "task": task,
                                    "episode": episode,
                                    "error": str(exc),
                                }
                            )
                            + "\n"
                        )
                        output.flush()
                        raise
                    output.write(json.dumps({"type": "episode", **result}) + "\n")
                    output.flush()
                    print(json.dumps(result), flush=True)
            finally:
                env.close()


if __name__ == "__main__":
    main()
