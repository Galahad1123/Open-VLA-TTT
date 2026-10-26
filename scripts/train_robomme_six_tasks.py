"""单卡串行运行六任务 Stage1/Stage2；失败立即停止，拒绝覆盖已有输出。"""

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

from lerobot_policy_vla_ttt.portability import read_machine

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_CHECKPOINT_FILES = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
)


def build_plan(machine, prefix, variant="legacy"):
    if variant not in {"legacy", "official"}:
        raise ValueError("Unknown recipe variant")
    machine = Path(machine).expanduser().resolve()
    if prefix in ("", ".", "..") or Path(prefix).name != prefix:
        raise ValueError("run-prefix 必须是单个目录名")
    storage = read_machine(machine).get("storage", {})
    if "runs_root" not in storage:
        raise ValueError("机器配置缺少 storage.runs_root")
    runs = Path(storage["runs_root"])
    plan = []
    for stage in ("stage1", "stage2"):
        tag = "official-" if variant == "official" else ""
        recipe = ROOT / f"configs/training/robomme-screen/multitask-six-{tag}{stage}.json"
        config = json.loads(recipe.read_text())
        if config["training_stage"] != stage:
            raise ValueError(f"{recipe} 的训练阶段不匹配")
        output = runs / f"{prefix}-{stage}"
        if output.exists():
            raise FileExistsError(f"输出已存在，请换 run-prefix 或单独启动后续阶段：{output}")
        command = [
            sys.executable,
            "-u",
            "-m",
            "lerobot_policy_vla_ttt.training.train_task_subset",
            "--machine",
            str(machine),
            "--recipe",
            str(recipe),
            "--output",
            str(output),
        ]
        if stage == "stage1":
            command += ["--fresh"]
        else:
            command += ["--checkpoint", str(plan[0]["checkpoint"])]
        plan.append(
            dict(
                command=command,
                output=output,
                steps=config["steps"],
                checkpoint=output / f"checkpoints/{config['steps']:06d}/pretrained_model",
            )
        )
    return plan


def validate_stage1(stage):
    summary = stage["output"] / "summary.json"
    if not summary.is_file():
        raise RuntimeError("Stage1 没有完成记录，停止串行训练")
    result = json.loads(summary.read_text())
    if result.get("status") != "complete" or result.get("updates") != stage["steps"]:
        raise RuntimeError("Stage1 未完成全部更新，停止串行训练")
    missing = [
        name for name in REQUIRED_CHECKPOINT_FILES if not (stage["checkpoint"] / name).is_file()
    ]
    if missing:
        raise RuntimeError(f"Stage1 检查点缺少必要文件：{missing}")


def execute(plan, runner=subprocess.run):
    # Recheck both destinations before doing expensive work.
    for stage in plan:
        if stage["output"].exists():
            raise FileExistsError(stage["output"])
    for index, stage in enumerate(plan):
        if index:
            validate_stage1(plan[0])
        print(f"开始 Stage{index + 1}: {shlex.join(stage['command'])}", flush=True)
        runner(stage["command"], cwd=ROOT, check=True)
    print("两个阶段的训练命令均成功退出。", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine", type=Path, default=Path("configs/machine.local.toml"))
    parser.add_argument("--run-prefix")
    parser.add_argument("--variant", choices=("legacy", "official"), default="legacy")
    parser.add_argument(
        "--dry-run", action="store_true", help="仅打印两阶段命令，不创建目录或启动训练"
    )
    args = parser.parse_args(argv)
    try:
        prefix = args.run_prefix or (
            "robomme-six-official-seed1000"
            if args.variant == "official"
            else "robomme-six-fresh-seed1000"
        )
        plan = build_plan(args.machine, prefix, args.variant)
        if args.dry_run:
            for stage in plan:
                print(shlex.join(stage["command"]))
        else:
            execute(plan)
    except subprocess.CalledProcessError as exc:
        print(f"训练进程失败（退出码 {exc.returncode}），后续阶段不会启动。", file=sys.stderr)
        raise SystemExit(exc.returncode if exc.returncode > 0 else 1) from exc
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
