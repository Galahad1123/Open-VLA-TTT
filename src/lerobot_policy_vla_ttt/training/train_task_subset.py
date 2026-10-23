"""Portable random-chunk RoboMME screening using the standard LeRobot trainer.

Only weight initialization is supported. --stop-after is a profiling-only early
stop (no final checkpoint); it retains the recipe's complete LR schedule.
"""

import argparse
import dataclasses
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig

from ..baselines.prepare import ROBOMME_REVISION
from ..baselines.robomme_data import make_control_datasets
from ..cli import local_processor_routing
from ..portability import configure_cache, local_vlm, read_machine, relocate_vlm, resolve_machine


@dataclasses.dataclass(frozen=True)
class ScreenRecipe:
    episodes: list[int]
    steps: int = 2000
    batch_size: int = 4
    seed: int = 1000
    lr: float = 1e-5
    warmup_steps: int = 100
    save_every: int = 500
    num_workers: int = 2
    training_stage: str = "stage2"
    sampling_mode: str = "random_frames"
    task_groups: dict[str, list[int]] | None = None

    def __post_init__(self):
        if self.training_stage not in {"stage1", "stage2"}:
            raise ValueError("training_stage must be stage1 or stage2")
        if self.sampling_mode not in {"random_frames", "task_balanced"}:
            raise ValueError("Unknown sampling_mode")
        if self.sampling_mode == "task_balanced":
            groups = self.task_groups
            if not isinstance(groups, dict) or not groups:
                raise ValueError("task_balanced requires nonempty task_groups")
            if any(
                not isinstance(k, str) or not k or not isinstance(v, list) or not v
                for k, v in groups.items()
            ):
                raise ValueError("task_groups must contain named nonempty episode lists")
            ids = [ep for group in groups.values() for ep in group]
            if (
                any(type(ep) is not int for ep in ids)
                or len(set(ids)) != len(ids)
                or set(ids) != set(self.episodes)
            ):
                raise ValueError("task_groups must partition episodes exactly")
            if self.batch_size < len(groups):
                raise ValueError("batch_size must cover every task group")
        elif self.task_groups is not None:
            raise ValueError("task_groups requires task_balanced sampling")
        if (
            not self.episodes
            or any(type(x) is not int or not 0 <= x < 1600 for x in self.episodes)
            or len(set(self.episodes)) != len(self.episodes)
        ):
            raise ValueError("episodes must contain unique integer original IDs in [0, 1600)")
        for name in ("steps", "batch_size", "save_every"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.num_workers) is not int or self.num_workers < 0:
            raise ValueError("num_workers must be a nonnegative integer")
        if type(self.warmup_steps) is not int or not 0 <= self.warmup_steps <= self.steps:
            raise ValueError("warmup_steps must be in [0, steps]")
        if not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("lr must be finite and positive")


def adapt_policy(source, recipe, checkpoint, *, fresh=False):
    if source.type != "evo1_groot":
        raise ValueError("Task screening requires an evo1_groot checkpoint")
    if source.chunk_size != 20 or source.n_action_steps != 10:
        raise ValueError("Expected RoboMME checkpoint with horizon=20, execution=10")
    finetune = recipe.training_stage == "stage2"
    return dataclasses.replace(
        source,
        pretrained_path=None if fresh else str(checkpoint),
        training_stage=recipe.training_stage,
        apply_training_stage_defaults=True,
        finetune_vlm=finetune,
        finetune_language_model=finetune,
        finetune_vision_model=finetune,
        finetune_action_head=True,
        dtype="bfloat16",
        vlm_dtype="float32",
        device="cuda",
        use_amp=True,
        use_flash_attn=False,
        enable_gradient_checkpointing=True,
        gradient_checkpointing_use_reentrant=False,
        optimizer_lr=recipe.lr,
        scheduler_warmup_steps=recipe.warmup_steps,
        task_field="task",
        push_to_hub=False,
    )


def frame_groups_for_recipe(dataset, recipe):
    """Map original episode IDs to the execution-only view's compact indices."""
    import numpy as np

    starts = dataset.meta.episodes["dataset_from_index"]
    ends = dataset.meta.episodes["dataset_to_index"]
    ranges = dict(zip(dataset.source_episode_ids, zip(starts, ends, strict=True), strict=True))
    return {
        name: np.concatenate([np.arange(*ranges[ep], dtype=np.int64) for ep in ids])
        for name, ids in recipe.task_groups.items()
    }


class ProfileComplete(Exception):
    """Intentional profiling stop, not a training failure."""


def run(cfg, identity, stop_after):
    import torch
    from lerobot.scripts import lerobot_eval, lerobot_train

    original_factory = lerobot_train.make_train_eval_datasets
    original_update = lerobot_train.update_policy
    original_sampler = lerobot_train.EpisodeAwareSampler
    recipe = ScreenRecipe(**identity["recipe"])
    owns_output = False
    rows = []
    started = time.perf_counter()
    last_end = None
    status = "failed"

    def factory(config):
        nonlocal owns_output
        datasets = make_control_datasets(config)
        if recipe.sampling_mode == "task_balanced":
            from .task_balanced_sampler import TaskBalancedSampler

            groups = frame_groups_for_recipe(datasets[0], recipe)
            identity["task_sampling"] = {
                name: {"episodes": len(recipe.task_groups[name]), "execution_frames": len(indices)}
                for name, indices in groups.items()
            }

            # LeRobot still owns DataLoader/optimizer/checkpoint logic. Replace
            # only its sampler constructor during this scoped training call.
            def sampler_factory(*args, **kwargs):
                return TaskBalancedSampler(groups, recipe.batch_size, recipe.steps, recipe.seed)

            lerobot_train.EpisodeAwareSampler = sampler_factory
        config.output_dir.mkdir(parents=True, exist_ok=False)
        owns_output = True
        identity["data_audit"] = datasets[0].report
        (config.output_dir / "run.json").write_text(json.dumps(identity, indent=2))
        return datasets

    def update(*args, **kwargs):
        nonlocal last_end
        if stop_after is not None and len(rows) >= stop_after:
            raise ProfileComplete
        torch.cuda.synchronize()
        before = time.perf_counter()
        tracker, output = original_update(*args, **kwargs)
        torch.cuda.synchronize()
        end = time.perf_counter()
        row = {
            "step": len(rows) + 1,
            "loss": float(tracker.loss.val),
            "grad_norm": float(tracker.grad_norm.val),
            "lr": float(tracker.lr.val),
            "update_seconds": end - before,
            "step_seconds": end - last_end if last_end is not None else None,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            "timestamp": time.time(),
        }
        if not math.isfinite(row["loss"]) or not math.isfinite(row["grad_norm"]):
            raise FloatingPointError("Nonfinite loss or gradient norm")
        rows.append(row)
        with (cfg.output_dir / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        last_end = end
        return tracker, output

    lerobot_train.make_train_eval_datasets = factory
    lerobot_train.update_policy = update
    try:
        with local_processor_routing(lerobot_train, lerobot_eval):
            lerobot_train.train(cfg)
        status = "complete"
    except ProfileComplete:
        status = "profile_complete"
    finally:
        lerobot_train.make_train_eval_datasets = original_factory
        lerobot_train.update_policy = original_update
        lerobot_train.EpisodeAwareSampler = original_sampler
        if owns_output:
            timings = [x["step_seconds"] for x in rows[5:] if x["step_seconds"] is not None]
            median = statistics.median(timings) if timings else None
            summary = {
                "status": status,
                "updates": len(rows),
                "wall_seconds": time.perf_counter() - started,
                "steady_median_step_seconds": median,
                "estimated_2000_update_hours": median * 2000 / 3600 if median else None,
                "estimated_recipe_hours": median * cfg.steps / 3600 if median else None,
                "peak_allocated_gib": max((x["peak_allocated_gib"] for x in rows), default=0),
                "peak_reserved_gib": max((x["peak_reserved_gib"] for x in rows), default=0),
                "note": "Estimate excludes startup/evaluation; profiles omit final checkpoint.",
            }
            (cfg.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--vlm-path", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "Use checkpoint configuration only; initialize action modules randomly "
            "and load the generic VLM"
        ),
    )
    args = parser.parse_args(argv)
    args.benchmark, args.resume = "robomme", None
    try:
        configure_cache(
            read_machine(args.machine), __spec__.name, sys.argv[1:] if argv is None else argv
        )
        resolve_machine(args)
        if any(x is None for x in (args.checkpoint, args.dataset_root, args.output)):
            raise ValueError("Provide checkpoint, dataset-root and output via CLI or machine TOML")
        if args.output.exists():
            raise ValueError("Use a new output directory")
        recipe = ScreenRecipe(**json.loads(args.recipe.read_text()))
        if args.stop_after is not None and not 1 <= args.stop_after < recipe.steps:
            raise ValueError("Profiling stop-after must be positive and less than recipe.steps")
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise ValueError("Screening uses one GPU per process; do not use distributed launch")
        source = PreTrainedConfig.from_pretrained(args.checkpoint, local_files_only=True)
        source = relocate_vlm(source, args.vlm_path)
        if not args.dry_run:
            source = local_vlm(source)
        policy = adapt_policy(source, recipe, args.checkpoint, fresh=args.fresh)
        dataset = DatasetConfig(
            repo_id="lerobot/robomme",
            root=str(args.dataset_root),
            revision=ROBOMME_REVISION,
            episodes=sorted(recipe.episodes),
            video_backend="pyav",
            return_uint8=True,
            eval_split=0.0,
            use_imagenet_stats=False,
        )
        dataset.image_transforms.enable = False
        cfg = TrainPipelineConfig(
            policy=policy,
            dataset=dataset,
            output_dir=args.output,
            seed=recipe.seed,
            steps=recipe.steps,
            batch_size=recipe.batch_size,
            num_workers=recipe.num_workers,
            persistent_workers=recipe.num_workers > 0,
            prefetch_factor=2,
            env=None,
            env_eval_freq=0,
            log_freq=10,
            save_freq=recipe.save_every,
            save_checkpoint=args.stop_after is None,
            save_checkpoint_to_hub=False,
            resume=False,
            cudnn_deterministic=True,
        )
        identity = {
            "recipe": dataclasses.asdict(recipe),
            "checkpoint": None if args.fresh else str(args.checkpoint),
            "config_template": str(args.checkpoint),
            "fresh_policy": args.fresh,
            "dataset_root": str(args.dataset_root),
            "vlm_path": policy.vlm_model_name,
            "initialization_only": True,
            "source_training_stage": None if args.fresh else source.training_stage,
            "training_stage": recipe.training_stage,
            "sampling": recipe.sampling_mode,
            "profile_stop_after": args.stop_after,
        }
        if args.dry_run:
            print(json.dumps(identity, indent=2))
            return
    except (ValueError, TypeError, OSError) as exc:
        parser.error(str(exc))
    run(cfg, identity, args.stop_after)


if __name__ == "__main__":
    main()
