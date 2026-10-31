"""RMBench official expert-filtered seed protocol with a remote VLA policy."""

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from .client import PolicyClient
from .rmbench import UPSTREAM_REVISION, canonical_observation, resolve_tasks, validate_action
from .rmbench_diagnostics import EpisodeDiagnostics
from .rmbench_video import H264Video


def append(path, row):
    with path.open("a") as stream:
        stream.write(json.dumps(row) + "\n")
        stream.flush()


def task_config(source, task):
    import yaml

    config_root = source / "env_cfg" / "task_config"
    cfg = yaml.safe_load((config_root / "demo_clean.yml").read_text())
    embodiments = yaml.safe_load((config_root / "_embodiment_config.yml").read_text())
    robot = source / embodiments["aloha-agilex"]["file_path"]
    embodiment = yaml.safe_load((robot / "config.yml").read_text())
    cfg.update(
        task_name=task,
        task_config="demo_clean",
        ckpt_setting="remote",
        policy_name="vla_ttt_remote",
        eval_mode=True,
        render_freq=0,
        left_robot_file=str(robot),
        right_robot_file=str(robot),
        left_embodiment_config=embodiment,
        right_embodiment_config=embodiment,
        dual_arm_embodied=True,
        head_camera_h=240,
        head_camera_w=320,
        eval_video_log=False,
        collect_data=False,
    )
    return cfg


def merge_results(paths):
    identity = None
    seen = set()
    rows = []
    expected = {}
    for path in map(Path, paths):
        manifest = json.loads((path / "manifest.json").read_text())
        current = {k: manifest[k] for k in ("upstream_revision", "protocol", "seed")}
        current["policy"] = {
            k: manifest["policy"][k] for k in ("checkpoint_sha256", "n_action_steps")
        }
        if identity is not None and identity != current:
            raise ValueError("Cannot merge incompatible checkpoint/protocol/seed")
        identity = current
        task = manifest["task"]
        if task in expected and expected[task] != manifest["episodes"]:
            raise ValueError("Inconsistent total episode counts")
        expected[task] = manifest["episodes"]
        for line in (path / "episodes.jsonl").read_text().splitlines():
            row = json.loads(line)
            key = (row["task"], row["episode"])
            if row["task"] != task or key in seen or not 0 <= row["episode"] < expected[task]:
                raise ValueError("Duplicate or invalid evaluation episode")
            if any(r["task"] == task and r["seed"] == row["seed"] for r in rows):
                raise ValueError("Duplicate evaluation seed")
            seen.add(key)
            rows.append(row)
    if not rows:
        raise ValueError("No evaluation episodes")
    successes = sum(row["success"] for row in rows)
    return {
        "identity": identity,
        "completed_episodes": len(rows),
        "expected_episodes": sum(expected.values()),
        "complete": len(rows) == sum(expected.values()),
        "successes": successes,
        "success_rate": successes / len(rows),
        "tasks": {
            task: {
                "episodes": sum(r["task"] == task for r in rows),
                "successes": sum(r["success"] for r in rows if r["task"] == task),
            }
            for task in expected
        },
    }


def evaluate(
    task,
    env,
    cfg,
    client,
    output,
    *,
    seed,
    episodes,
    max_candidates,
    describe,
    start_episode=0,
    end_episode=None,
    diagnostics=True,
):
    """Match official seed filtering; exceptions are recorded, never policy failures."""
    accepted = successes = 0
    end_episode = episodes if end_episode is None else end_episode
    started = time.monotonic()
    for candidate in range(100000 * (1 + seed), 100000 * (1 + seed) + max_candidates):
        begin = time.monotonic()
        try:
            env.setup_demo(now_ep_num=accepted, seed=candidate, is_test=True, **cfg)
            info = env.play_once()
            env.close_env()
            feasible = bool(env.plan_success and env.check_success())
        except Exception as error:
            append(
                output / "candidates.jsonl",
                {"seed": candidate, "status": "expert_error", "error": repr(error)},
            )
            env.close_env()
            continue
        if not feasible:
            append(output / "candidates.jsonl", {"seed": candidate, "status": "expert_invalid"})
            continue
        append(
            output / "candidates.jsonl",
            {"seed": candidate, "status": "accepted", "episode": accepted},
        )
        if accepted < start_episode:
            accepted += 1
            continue
        video = output / "videos" / f"episode-{accepted:03d}-seed-{candidate}.mp4"
        try:
            env.setup_demo(now_ep_num=accepted, seed=candidate, is_test=True, **cfg)
            descriptions = describe(task, [info["info"]], episodes)
            instruction = str(np.random.choice(descriptions[0]["seen"]))
            env.set_instruction(instruction=instruction)
            client.reset(f"{task}:{candidate}", candidate)
            observation = env.get_obs()
            frame = observation["observation"]["head_camera"]["rgb"]
            # Decision-rate head-camera video; policy still receives all three cameras.
            diagnostic_path = (
                output / "diagnostics" / f"episode-{accepted:03d}-seed-{candidate}.jsonl"
            )
            recorder = (
                EpisodeDiagnostics(
                    env, diagnostic_path, task=task, episode=accepted, seed=candidate
                )
                if diagnostics
                else nullcontext(None)
            )
            frame_index = 0
            with recorder as diagnostic, H264Video(video, frame.shape[:2]) as writer:
                writer.write(frame)
                while env.take_action_cnt < env.step_lim:
                    action = validate_action(
                        client.infer(canonical_observation(observation, instruction))
                    )
                    before = env.take_action_cnt
                    env.take_action(action, action_type="qpos")
                    if env.take_action_cnt <= before:
                        raise RuntimeError("Environment did not advance its action counter")
                    observation = env.get_obs()
                    writer.write(observation["observation"]["head_camera"]["rgb"])
                    frame_index += 1
                    if diagnostic is not None:
                        diagnostic.action(
                            action, observation["joint_action"]["vector"], frame=frame_index
                        )
                    if env.eval_success:
                        break
            success = bool(env.eval_success)
            row = {
                "task": task,
                "episode": accepted,
                "seed": candidate,
                "success": success,
                "steps": int(env.take_action_cnt),
                "max_reward": np.asarray(env.max_reward).tolist(),
                "instruction": instruction,
                "elapsed_seconds": time.monotonic() - begin,
                "video": str(video),
                "diagnostics": str(diagnostic_path) if diagnostics else None,
            }
            append(output / "episodes.jsonl", row)
            successes += int(success)
            accepted += 1
            progress = {
                "completed_episodes": accepted - start_episode,
                "expected_episodes": end_episode - start_episode,
                "successes": successes,
                "success_rate": successes / (accepted - start_episode),
                "elapsed_seconds": time.monotonic() - started,
            }
            temporary = output / "progress.partial.json"
            temporary.write_text(json.dumps(progress, indent=2))
            temporary.replace(output / "progress.json")
            print(json.dumps(progress), flush=True)
        except Exception as error:
            append(
                output / "errors.jsonl",
                {"seed": candidate, "episode": accepted, "error": repr(error)},
            )
            raise
        finally:
            env.close_env(clear_cache=(accepted + 1) % cfg["clear_cache_freq"] == 0)
        if accepted == end_episode:
            return
    raise RuntimeError(f"Only {accepted}/{episodes} accepted seeds within candidate limit")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--merge":
        print(json.dumps(merge_results(sys.argv[2:]), indent=2))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Dedicated pinned RMBench runtime checkout with assets attached",
    )
    parser.add_argument(
        "--no-diagnostics", action="store_true", help="Disable execution JSONL diagnostics"
    )
    parser.add_argument("--task", required=True)
    parser.add_argument("--uri", default="ws://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--end-episode", type=int)
    parser.add_argument("--max-candidates", type=int, default=10000)
    args = parser.parse_args()
    if args.end_episode is None:
        args.end_episode = args.episodes
    if not 0 <= args.start_episode < args.end_episode <= args.episodes:
        raise ValueError("Expected 0 <= start < end <= total episodes")
    resolve_tasks([args.task])
    if args.seed < 0 or args.episodes <= 0 or args.max_candidates < args.episodes:
        raise ValueError("Invalid seed/episode/candidate count")
    source, output = args.source.resolve(), args.output.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError("RMBench source revision mismatch")
    dirty = subprocess.check_output(["git", "-C", str(source), "diff", "HEAD", "--"], text=True)
    if dirty:
        raise ValueError("Use an unmodified pinned runtime checkout")
    output.mkdir(parents=True, exist_ok=False)
    os.chdir(source)
    sys.path[:0] = [str(source), str(source / "description" / "utils")]
    module = importlib.import_module(f"envs.{args.task}")
    describe = importlib.import_module(
        "generate_episode_instructions"
    ).generate_episode_descriptions
    env = getattr(module, args.task)()
    with PolicyClient(args.uri, benchmark="rmbench") as client:
        if client.metadata["action_dim"] != 14 or client.metadata["actions_normalized"]:
            raise ValueError("Expected denormalized 14D joint-target policy")
        (output / "manifest.json").write_text(
            json.dumps(
                {
                    "upstream_revision": revision,
                    "protocol": "official-expert-filter-seen-v1",
                    "task": args.task,
                    "seed": args.seed,
                    "episodes": args.episodes,
                    "start_episode": args.start_episode,
                    "end_episode": args.end_episode,
                    "policy": client.metadata,
                    "video": "h264-yuv420p-head-decision-rate-10fps",
                    "diagnostics": None if args.no_diagnostics else "execution-jsonl-v1",
                },
                indent=2,
            )
        )
        evaluate(
            args.task,
            env,
            task_config(source, args.task),
            client,
            output,
            seed=args.seed,
            episodes=args.episodes,
            start_episode=args.start_episode,
            end_episode=args.end_episode,
            max_candidates=args.max_candidates,
            describe=describe,
            diagnostics=not args.no_diagnostics,
        )


if __name__ == "__main__":
    main()
