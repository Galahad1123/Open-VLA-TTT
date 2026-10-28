"""Offline RMBench baseline runner, sharing LeRobot's training/checkpoint loop."""

import argparse
import functools
import hashlib
import json
import math
import os
import subprocess
import sys
from importlib import import_module
from pathlib import Path

import torch
from lerobot.datasets.sampler import EpisodeAwareSampler

from lerobot_policy_vla_ttt.baselines.rmbench import UPSTREAM_REVISION, resolve_tasks
from lerobot_policy_vla_ttt.cli import local_processor_routing
from lerobot_policy_vla_ttt.portability import configure_cache, read_machine


def training_budget(chunks_per_task, batch_size, tasks, world_size=1):
    if min(chunks_per_task, batch_size, tasks, world_size) <= 0:
        raise ValueError("Budget, batch size, task count and world size must be positive")
    return math.ceil(chunks_per_task * tasks / (batch_size * world_size))


def stage_budget(settings, tasks, world_size, *, smoke_steps=None, smoke_use_recipe_batch=False):
    if ("steps" in settings) == ("chunks_per_task" in settings):
        raise ValueError("Each stage requires exactly one of steps or chunks_per_task")
    batch = settings["batch_size"]
    budget = settings.get("steps", settings.get("chunks_per_task"))
    if any(type(value) is not int or value <= 0 for value in (batch, budget, tasks, world_size)):
        raise ValueError("Training budgets and batch sizes must be positive integers")
    steps = settings.get("steps") or training_budget(budget, batch, tasks, world_size)
    if smoke_steps is not None:
        if type(smoke_steps) is not int or smoke_steps <= 0:
            raise ValueError("Smoke steps must be positive")
        return batch if smoke_use_recipe_batch else 1, smoke_steps
    return batch, steps


def select_episodes(preparation, tasks):
    """Select a read-only view; normalization stays tied to the source dataset."""
    missing = set(tasks) - set(preparation["tasks"])
    if missing:
        raise ValueError(f"Requested tasks missing from dataset: {sorted(missing)}")
    if set(tasks) == set(preparation["tasks"]):
        return None  # Preserve existing all-data runs and their resume configuration.
    episodes = preparation["episodes"]
    if [item["episode"] for item in episodes] != list(range(len(episodes))):
        raise ValueError("Preparation episodes must use contiguous source indices")
    if set(tasks) - {item["task"] for item in episodes}:
        raise ValueError("Selected tasks have no episodes in preparation")
    return [item["episode"] for item in episodes if item["task"] in tasks]


def validate_lineage(checkpoint, identity, *, resume):
    saved = json.loads((Path(checkpoint) / "rmbench_run.json").read_text())
    if resume:
        matches = saved == identity
    else:
        keys = ("tasks", "recipe", "seed", "preparation_sha256", "run_name")
        matches = saved["stage"] == "stage1" and all(
            saved.get(key) == identity.get(key) for key in keys
        )
    if not matches:
        raise ValueError("Checkpoint does not belong to this RMBench stage/data/run")


class BalancedTaskSampler(EpisodeAwareSampler):
    """Uniform task, then uniform eligible frame; deterministic resumable epochs."""

    def __init__(self, starts, ends, *, task_by_episode, global_batch_size=1, **kwargs):
        subset = kwargs.get("episode_indices_to_use")
        if len(task_by_episode) != len(starts) or (
            subset is not None and any(i < 0 or i >= len(starts) for i in subset)
        ):
            raise ValueError("Episode/task mapping mismatch")
        super().__init__(starts, ends, **kwargs)
        self._resume_epoch = 0
        self._padding = (-self._num_frames) % global_batch_size
        selected = set(range(len(starts))) if subset is None else set(subset)
        self.task_samplers = []
        for task in sorted(set(task_by_episode)):
            ids = [i for i, value in enumerate(task_by_episode) if value == task and i in selected]
            if not ids:
                continue
            options = dict(kwargs, episode_indices_to_use=ids)
            self.task_samplers.append(EpisodeAwareSampler(starts, ends, **options))

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self._resume_epoch = state["epoch"]

    def set_epoch(self, epoch):
        # DataLoaderShard starts its local iteration at zero even on resume.
        super().set_epoch(self._resume_epoch + epoch)

    def _iter_epoch(self, epoch, start):
        generator = self._epoch_generator(epoch)
        prefix = []
        # Regenerate consumed positions too: resume must preserve both random draws.
        for position in range(self._num_frames):
            task = int(torch.randint(len(self.task_samplers), (), generator=generator))
            sampler = self.task_samplers[task]
            frame = int(torch.randint(len(sampler), (), generator=generator))
            index = sampler._frame_index(frame)
            if position < self._padding:
                prefix.append(index)
            if position >= start:
                yield index
        # Pad from the ORIGINAL epoch prefix, not the resumed suffix. Otherwise
        # Accelerate's even_batches changes the last rank's final batch on resume.
        for position in range(self._padding):
            yield prefix[position % len(prefix)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--mode", choices=["single", "multi"], required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--stage", choices=["stage1", "stage2", "both"], default="both")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--resume", type=Path, help="Same-stage pretrained_model checkpoint directory"
    )
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--smoke-steps", type=int)
    parser.add_argument("--smoke-use-recipe-batch", action="store_true")
    parser.add_argument("--num-workers", type=int, help="DataLoader workers per GPU")
    parser.add_argument(
        "--num-processes", type=int, default=1, help="Local GPU count; batch_size is per GPU"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    machine = read_machine(args.machine)
    configure_cache(machine, __spec__.name, sys.argv[1:])
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[key] = "1"
    worker_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if args.num_processes < 1 or worker_world_size < 1:
        raise ValueError("Process count must be positive")
    if worker_world_size > 1 and (args.num_processes != 1 or args.stage == "both"):
        raise ValueError("Use --num-processes in a plain Python launch for two-stage DDP")
    world_size = max(worker_world_size, args.num_processes)
    launch_workers = args.num_processes > 1
    if args.smoke_use_recipe_batch and args.smoke_steps is None:
        raise ValueError("--smoke-use-recipe-batch requires --smoke-steps")
    if args.num_workers is not None and args.num_workers < 0:
        raise ValueError("Worker count must be nonnegative")
    num_workers = (
        args.num_workers if args.num_workers is not None else (0 if args.smoke_steps else 4)
    )
    tasks = resolve_tasks(args.tasks)
    if args.mode == "single" and len(tasks) != 1:
        raise ValueError("Single-task entry requires exactly one task")
    if Path(args.run_name).name != args.run_name or args.run_name in ("", ".", ".."):
        raise ValueError("run-name must be a single directory name")
    if args.smoke_steps is not None and args.smoke_steps <= 0:
        raise ValueError("smoke-steps must be positive")
    if args.resume and (args.stage == "both" or args.checkpoint):
        raise ValueError("Resume requires one stage and no initialization checkpoint")
    assets = machine["assets"]
    dataset = (args.dataset_root or Path(assets["rmbench_dataset"])).resolve()
    vlm = Path(assets["vlm"])
    if not (vlm / "config.json").is_file():
        raise ValueError("A local VLM directory is required")
    preparation_bytes = (dataset / "preparation.json").read_bytes()
    preparation = json.loads(preparation_bytes)
    episodes = select_episodes(preparation, tasks)
    if preparation["upstream_revision"] != UPSTREAM_REVISION:
        raise ValueError("Dataset protocol revision mismatch")
    if preparation["partial"] and not args.smoke_steps:
        raise ValueError("Partial preparation is only allowed with --smoke-steps")
    recipe = json.loads(args.recipe.read_text())
    root = Path(machine["storage"]["runs_root"])
    stages = ["stage1", "stage2"] if args.stage == "both" else [args.stage]
    checkpoint = args.checkpoint.resolve() if args.checkpoint else None
    if checkpoint and args.stage != "stage2":
        raise ValueError("--checkpoint is only for Stage1 to Stage2 initialization")
    distributed = None
    if worker_world_size > 1 and not args.dry_run:
        from accelerate import PartialState

        distributed = PartialState()
    is_main = distributed is None or distributed.is_main_process
    original_argv = sys.argv
    for stage in stages:
        settings = recipe[stage]
        batch, steps = stage_budget(
            settings,
            len(tasks),
            world_size,
            smoke_steps=args.smoke_steps,
            smoke_use_recipe_batch=args.smoke_use_recipe_batch,
        )
        output = root / f"{args.run_name}-{stage}"
        identity = {
            "run_name": args.run_name,
            "tasks": tasks,
            "stage": stage,
            "recipe": recipe,
            "seed": args.seed,
            "steps": steps,
            "batch_size": batch,
            "preparation_sha256": hashlib.sha256(preparation_bytes).hexdigest(),
        }
        if world_size > 1:
            identity.update(world_size=world_size, global_batch_size=batch * world_size)
        manifest_path = root / f"{args.run_name}-{stage}.rmbench.json"
        if args.resume:
            saved = json.loads(manifest_path.read_text())
            if saved != identity:
                raise ValueError("Resume configuration/data identity differs from original run")
            if not (args.resume / "train_config.json").is_file():
                raise ValueError("Resume needs a complete LeRobot checkpoint")
            validate_lineage(args.resume, identity, resume=True)
        elif is_main and (output.exists() or manifest_path.exists()):
            raise ValueError(f"Run already exists: {output}")
        common = {
            "dataset.repo_id": "local/rmbench",
            "dataset.root": str(dataset),
            "dataset.return_uint8": True,
            "dataset.use_imagenet_stats": False,
            "dataset.image_transforms.enable": False,
            "dataset.eval_split": 0.0,
            "policy.vlm_model_name": str(vlm),
            "policy.training_stage": stage,
            "policy.apply_training_stage_defaults": True,
            "policy.vlm_dtype": "bfloat16" if stage == "stage1" else "float32",
            "policy.device": "cuda",
            "policy.use_amp": True,
            "policy.dtype": "bfloat16",
            "policy.use_flash_attn": False,
            "policy.enable_gradient_checkpointing": True,
            "policy.gradient_checkpointing_use_reentrant": False,
            "policy.image_resolution": [448, 448],
            "policy.chunk_size": recipe["chunk_size"],
            "policy.n_action_steps": recipe["n_action_steps"],
            "policy.max_state_dim": 24,
            "policy.max_action_dim": 24,
            "policy.postprocess_action_dim": 14,
            "policy.binarize_gripper": False,
            "policy.dropout": 0.2,
            "policy.optimizer_lr": recipe["action_lr"],
            "policy.optimizer_vlm_lr": recipe["vlm_lr"],
            "policy.optimizer_weight_decay": 0.001,
            "policy.optimizer_grad_clip_norm": 1.0,
            "policy.scheduler_warmup_steps": 1 if args.smoke_steps else 1000,
            "policy.push_to_hub": False,
            "use_policy_training_preset": True,
            "resume": bool(args.resume),
            "save_checkpoint": True,
            "save_checkpoint_to_hub": False,
            "env_eval_freq": 0,
            "log_freq": 1 if args.smoke_steps else 50,
            "num_workers": num_workers,
            "persistent_workers": num_workers > 0,
            "seed": args.seed,
            "cudnn_deterministic": True,
            "wandb.enable": False,
            "batch_size": batch,
            "steps": steps,
            "save_freq": max(1, steps // 2)
            if args.smoke_steps
            else min(recipe.get("save_freq", 10000), steps),
            "output_dir": str(output),
        }
        if episodes is not None:
            common["dataset.episodes"] = episodes
        if args.resume:
            common["config_path"] = str(args.resume / "train_config.json")
        elif checkpoint:
            if not (args.dry_run and args.stage == "both"):
                validate_lineage(checkpoint, identity, resume=False)
            common["policy.path"] = str(checkpoint)
        elif stage == "stage1":
            common["policy.type"] = "evo1_groot"
        else:
            raise ValueError("Stage2 requires --checkpoint from this experiment's Stage1")
        argv = ["vla-ttt-train"] + [
            f"--{key}={json.dumps(value) if not isinstance(value, str) else value}"
            for key, value in common.items()
        ]
        if is_main:
            print(
                json.dumps(
                    {
                        "output": str(output),
                        "identity": identity,
                        "selected_episodes": episodes,
                        "normalization": "source_dataset_stats",
                        "world_size": world_size,
                        "global_batch_size": batch * world_size,
                        "expected_chunks_per_task": steps * batch * world_size / len(tasks),
                        "argv": argv,
                    },
                    indent=2,
                ),
                flush=True,
            )
        if launch_workers and not args.dry_run:
            # A fresh torchrun job per stage releases model, optimizer and process group.
            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={world_size}",
                "--module",
                __spec__.name,
                "--machine",
                str(args.machine),
                "--recipe",
                str(args.recipe),
                "--tasks",
                *tasks,
                "--mode",
                args.mode,
                "--run-name",
                args.run_name,
                "--stage",
                stage,
                "--seed",
                str(args.seed),
            ]
            for flag, value in (
                ("--checkpoint", checkpoint),
                ("--resume", args.resume),
                ("--dataset-root", args.dataset_root),
                ("--smoke-steps", args.smoke_steps),
                ("--num-workers", args.num_workers),
            ):
                if value is not None:
                    command.extend([flag, str(value)])
            if args.smoke_use_recipe_batch:
                command.append("--smoke-use-recipe-batch")
            subprocess.run(command, check=True)
        elif not args.dry_run:
            root.mkdir(parents=True, exist_ok=True)
            if is_main and not args.resume:
                with manifest_path.open("x") as stream:
                    json.dump(identity, stream, indent=2)
            if distributed is not None:
                distributed.wait_for_everyone()
            runner = import_module("lerobot.scripts.lerobot_train")
            evaluator = import_module("lerobot.scripts.lerobot_eval")
            original_sampler = runner.EpisodeAwareSampler
            original_save = runner.save_checkpoint

            def save_with_identity(
                *save_args,
                original_save=original_save,
                identity=identity,
                checkpoint=checkpoint,
                **save_kwargs,
            ):
                result = original_save(*save_args, **save_kwargs)
                saved_policy = Path(save_kwargs["checkpoint_dir"]) / "pretrained_model"
                (saved_policy / "rmbench_run.json").write_text(json.dumps(identity, indent=2))
                if checkpoint:
                    from lerobot_policy_vla_ttt.evaluation.policy import checkpoint_identity

                    (saved_policy / "initialization.json").write_text(
                        json.dumps(
                            {"stage1_checkpoint_sha256": checkpoint_identity(checkpoint)}, indent=2
                        )
                    )
                return result

            try:
                sys.argv = argv
                runner.EpisodeAwareSampler = functools.partial(
                    BalancedTaskSampler,
                    task_by_episode=[item["task"] for item in preparation["episodes"]],
                    global_batch_size=batch * world_size if world_size > 1 else 1,
                )
                runner.save_checkpoint = save_with_identity
                with local_processor_routing(runner, evaluator):
                    runner.main()
            finally:
                sys.argv = original_argv
                runner.EpisodeAwareSampler = original_sampler
                runner.save_checkpoint = original_save
        checkpoint = output / "checkpoints" / f"{steps:06d}" / "pretrained_model"


if __name__ == "__main__":
    main()
