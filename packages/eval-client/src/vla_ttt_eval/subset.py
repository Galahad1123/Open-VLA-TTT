"""Serial RoboMME subset evaluation with diagnostic videos, never privileged policy inputs."""

import argparse
import hashlib
import html
import json
import shutil
from pathlib import Path

from .client import PolicyClient
from .recorded import Recorded, base_env
from .robomme import RoboMMEEnvironment
from .runner import run_episode

TASKS = ("PickXtimes", "VideoUnmask", "PickHighlight", "ButtonUnmask", "BinFill", "SwingXtimes")


def eligible(task, env, repeats=None):
    if task in ("PickXtimes", "SwingXtimes"):
        return int(env.num_repeats) in ([1] if repeats is None else repeats)
    if task in ("VideoUnmask", "ButtonUnmask"):
        return int(env.configs[env.difficulty]["pick"]) == 1
    if task == "PickHighlight":
        return int(env.configs[env.difficulty]["pickup"]) == 1
    if task == "BinFill":
        return (
            sum(
                int(getattr(env, f"{color}_cubes_target_number"))
                for color in ("red", "blue", "green")
            )
            == 1
        )
    raise ValueError(f"Unsupported subset: {task}")


def load_plan(path):
    plan = json.loads(path.read_text())
    if plan.get("version") != 1 or not plan.get("tasks"):
        raise ValueError("Expected version 1 and nonempty tasks")
    for task, spec in plan["tasks"].items():
        if task not in TASKS:
            raise ValueError(f"Unknown task: {task}")
        repeats = spec.get("repeats")
        if repeats is not None and (
            task not in ("PickXtimes", "SwingXtimes")
            or not isinstance(repeats, list)
            or not repeats
            or any(type(n) is not int or n not in range(1, 6) for n in repeats)
            or len(set(repeats)) != len(repeats)
        ):
            raise ValueError(f"Invalid repeat selection: {task}")
        ids = spec["train_episodes"]
        if (
            not ids
            or any(type(x) is not int or not 0 <= x < 100 for x in ids)
            or len(ids) != len(set(ids))
        ):
            raise ValueError(f"Invalid task-local train episode IDs: {task}")
    return plan


def summarize(rows):
    episodes = [r for r in rows if r["type"] == "episode"]
    result = dict(
        episodes=len(episodes),
        successes=sum(r["success"] for r in episodes),
        errors=sum(r["type"] == "error" for r in rows),
        excluded=sum(r["type"] == "excluded" for r in rows),
    )
    result["success_rate"] = result["successes"] / len(episodes) if episodes else None
    result["tasks"] = {}
    for task in sorted({r["task"] for r in episodes}):
        group = [r for r in episodes if r["task"] == task]
        result["tasks"][task] = dict(
            episodes=len(group),
            successes=sum(r["success"] for r in group),
            success_rate=sum(r["success"] for r in group) / len(group),
        )
    result["macro_success_rate"] = (
        sum(x["success_rate"] for x in result["tasks"].values()) / len(result["tasks"])
        if result["tasks"]
        else None
    )
    repeat_groups = {}
    for row in episodes:
        if row.get("num_repeats") is not None:
            key = f"{row['task']}/{row['num_repeats']}"
            group = repeat_groups.setdefault(key, dict(episodes=0, successes=0))
            group["episodes"] += 1
            group["successes"] += int(row["success"])
    if repeat_groups:
        for group in repeat_groups.values():
            group["success_rate"] = group["successes"] / group["episodes"]
        result["repeat_groups"] = repeat_groups
        result["repeat_macro_success_rate"] = sum(
            g["success_rate"] for g in repeat_groups.values()
        ) / len(repeat_groups)
    return result


def report(output, rows, complete):
    summary = dict(complete=complete, **summarize(rows))
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    parts = [
        '<!doctype html><meta charset="utf-8"><title>RoboMME 子集评测</title>',
        "<h1>RoboMME 子集评测</h1>",
        "<pre>" + html.escape(json.dumps(summary, indent=2)) + "</pre>",
        "<p>阶段进展为仿真器诊断信息，不输入模型；不是独立的动作质量或目标选择评分。</p>",
    ]
    for row in rows:
        if row["type"] != "episode":
            continue
        parts.append(
            f"<details><summary>{html.escape(row['task'])} / {row['episode']} / "
            f"{row['termination']}</summary>"
        )
        for key in ("video", "demo_video"):
            if row.get(key):
                parts.append(
                    f'<p>{key}</p><video controls preload="none" width="640" '
                    f'src="{html.escape(row[key], quote=True)}"></video>'
                )
        parts.append("<pre>" + html.escape(json.dumps(row, indent=2)) + "</pre></details>")
    (output / "report.html").write_text("\n".join(parts))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--server", default="ws://127.0.0.1:8795")
    p.add_argument("--split", choices=("train", "val", "test"), default="val")
    p.add_argument("--tasks", nargs="+", choices=TASKS)
    p.add_argument("--selection", choices=("matched", "all"), default="matched")
    p.add_argument(
        "--limit-per-task", type=int, help="First N eligible episodes, in official episode order"
    )
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--max-steps", type=int, default=1300)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.max_steps < 1 or (args.limit_per_task is not None and args.limit_per_task < 1):
        p.error("Step and episode limits must be positive")
    plan = load_plan(args.plan)
    tasks = args.tasks or list(plan["tasks"])
    if len(set(tasks)) != len(tasks) or any(t not in plan["tasks"] for t in tasks):
        p.error("Tasks must be unique and present in plan")
    candidates = {
        t: (
            plan["tasks"][t]["train_episodes"]
            if args.split == "train" and args.selection == "matched"
            else list(range(100 if args.split == "train" else 50))
        )
        for t in tasks
    }
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(
        candidates=candidates,
        plan=plan,
        plan_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        protocol="mme-vla-1300" if args.max_steps == 1300 else "custom-smoke",
    )
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return
    if not shutil.which("ffmpeg"):
        p.error("ffmpeg with libx264 is required on PATH")
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []

    def emit(row):
        rows.append(row)
        with (args.output / "results.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        report(args.output, rows, False)
        print(json.dumps(row), flush=True)

    report(args.output, rows, False)
    with PolicyClient(args.server, benchmark="robomme", timeout=180) as client:
        if client.metadata["implementation"] not in {"evo1_groot", "aqmem"}:
            raise ValueError("Subset evaluation supports evo1_groot and aqmem")
        (args.output / "run.json").write_text(
            json.dumps(dict(config=config, policy=client.metadata), indent=2)
        )
        for task in tasks:
            env = RoboMMEEnvironment(task, args.split, args.max_steps)
            selected = 0
            try:
                for ep in candidates[task]:
                    try:
                        obs = env.reset(ep, args.seed + ep)
                        matches = eligible(task, base_env(env), plan["tasks"][task].get("repeats"))
                        if args.selection == "matched" and not matches:
                            if args.split == "train":
                                raise ValueError(f"Train plan/scene mismatch: {task}/{ep}")
                            emit(
                                dict(
                                    type="excluded",
                                    task=task,
                                    episode=ep,
                                    split=args.split,
                                    reason="scene does not match single-target/repeat rule",
                                    env_seed=env.env_seed,
                                    difficulty=env.difficulty,
                                )
                            )
                            continue
                        directory = args.output / "videos" / task
                        directory.mkdir(parents=True, exist_ok=True)
                        rec = Recorded(env, obs, directory / f"episode{ep:03d}.mp4")
                        try:
                            result = run_episode(
                                rec, client, task, ep, args.seed + ep, args.max_steps
                            )
                        finally:
                            rec.video.finish()
                        result.update(
                            type="episode",
                            split=args.split,
                            matches_subset=matches,
                            instruction=env.task,
                            num_repeats=int(base_env(env).num_repeats)
                            if task in ("PickXtimes", "SwingXtimes")
                            else None,
                            video=str(rec.path.relative_to(args.output)),
                            demo_video=str(rec.demo_path.relative_to(args.output))
                            if rec.demo_path
                            else None,
                            task_names=rec.task_names,
                            stage_transitions=rec.transitions,
                            max_stage_index=max(x["index"] for x in rec.transitions),
                            video_frames=rec.video.frames,
                        )
                        rec.path.with_suffix(".json").write_text(json.dumps(result, indent=2))
                        emit(result)
                        selected += 1
                        if args.limit_per_task and selected >= args.limit_per_task:
                            break
                    except Exception as exc:
                        emit(dict(type="error", task=task, episode=ep, error=str(exc)))
                        raise
            finally:
                env.close()
    report(args.output, rows, True)


if __name__ == "__main__":
    main()
