"""Portable random-chunk RoboMME screening using the standard LeRobot trainer.

Only weight initialization is supported. --stop-after is a profiling-only early
stop (no final checkpoint); it retains the recipe's complete LR schedule.
"""

import argparse
import copy
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

    def __post_init__(self):
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


def adapt_policy(source, recipe, checkpoint, *, precision="bf16"):
    dtype = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}[precision]
    if source.type != "evo1_groot":
        raise ValueError("Task screening requires an evo1_groot checkpoint")
    if source.chunk_size != 20 or source.n_action_steps != 10:
        raise ValueError("Expected RoboMME checkpoint with horizon=20, execution=10")
    return dataclasses.replace(
        source,
        pretrained_path=str(checkpoint),
        training_stage="stage2",
        apply_training_stage_defaults=True,
        finetune_vlm=True,
        finetune_language_model=True,
        finetune_vision_model=True,
        finetune_action_head=True,
        dtype=dtype,
        vlm_dtype="float32",
        device="cuda",
        use_amp=precision != "fp32",
        use_flash_attn=False,
        enable_gradient_checkpointing=True,
        gradient_checkpointing_use_reentrant=False,
        optimizer_lr=recipe.lr,
        scheduler_warmup_steps=recipe.warmup_steps,
        task_field="task",
        push_to_hub=False,
    )


def per_rank_batch(global_batch, world):
    """Recipes always specify a global batch, including legacy batch_size."""
    if world < 1 or global_batch < 1 or global_batch % world:
        raise ValueError("recipe batch_size (global) must be divisible by WORLD_SIZE")
    return global_batch // world


def successful_update(update_fn, *args, accelerator, lr_scheduler=None, **kwargs):
    """Use Accelerate's sole scaler; count only successful optimizer updates.

    DDP synchronizes gradients before GradScaler inspects them, so overflow is
    shared by all ranks. Replaying torch RNG preserves flow time/noise/dropout.
    The random-chunk trainer does not use the episode trainer's custom scaler.
    """
    import torch

    from .episode_parallel.runtime import collective_error

    started = time.perf_counter()
    metric_snapshot = copy.deepcopy(args[0].metrics) if args else None
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    device = getattr(accelerator, "device", torch.device("cpu"))
    for skipped in range(17):
        if skipped:
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng)
        # MetricsTracker owns an Accelerator and is not deepcopy-safe. Only
        # restore its meters; counters advance outside this function.
        if skipped and metric_snapshot is not None:
            args[0].metrics = copy.deepcopy(metric_snapshot)
        tracker, output = update_fn(*args, accelerator=accelerator, lr_scheduler=None, **kwargs)
        error = None if math.isfinite(float(tracker.loss.val)) else "Nonfinite forward loss"
        collective_error(error, device)
        if not accelerator.optimizer_step_was_skipped:
            error = (
                None if math.isfinite(float(tracker.grad_norm.val)) else "Nonfinite gradient norm"
            )
            collective_error(error, device)
            if lr_scheduler is not None:
                lr_scheduler.step()
            if hasattr(tracker, "update_s"):
                meter = tracker.update_s
                elapsed = time.perf_counter() - started
                meter.sum += elapsed - meter.val
                meter.val = elapsed
                meter.avg = meter.sum / meter.count
            return tracker, output, skipped
        if skipped == 16:
            raise FloatingPointError("FP16 gradient overflow persisted for 17 attempts")


class ProfileComplete(Exception):
    """Intentional profiling stop, not a training failure."""


def run(cfg, identity, stop_after):
    import torch
    from lerobot.scripts import lerobot_eval, lerobot_train

    original_factory = lerobot_train.make_train_eval_datasets
    original_update = lerobot_train.update_policy
    rows = []
    is_main = int(os.environ.get("RANK", "0")) == 0
    skipped_total = 0
    owns_output = False
    started = time.perf_counter()
    last_end = None
    status = "failed"

    def factory(config):
        nonlocal owns_output
        datasets = make_control_datasets(config)
        if is_main:
            config.output_dir.mkdir(parents=True, exist_ok=False)
            owns_output = True
            identity["data_audit"] = datasets[0].report
            (config.output_dir / "run.json").write_text(json.dumps(identity, indent=2))
        return datasets

    def update(*args, **kwargs):
        nonlocal last_end, skipped_total
        if stop_after is not None and len(rows) >= stop_after:
            raise ProfileComplete
        torch.cuda.synchronize()
        before = time.perf_counter()
        tracker, output, skipped = successful_update(original_update, *args, **kwargs)
        skipped_total += skipped
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
            "skipped_updates": skipped_total,
            "loss_scale": kwargs["accelerator"].scaler.get_scale()
            if kwargs["accelerator"].scaler
            else None,
        }
        if not math.isfinite(row["loss"]) or not math.isfinite(row["grad_norm"]):
            raise FloatingPointError("Nonfinite loss or gradient norm")
        # All ranks participate; only rank zero writes shared artifacts.
        import torch.distributed as dist

        if dist.is_initialized():
            loss = torch.tensor(row["loss"], device=kwargs["accelerator"].device)
            dist.all_reduce(loss)
            row["loss"] = loss.item() / dist.get_world_size()
            keys = ["update_seconds", "step_seconds", "peak_allocated_gib", "peak_reserved_gib"]
            values = torch.tensor(
                [row[k] or 0 for k in keys], dtype=torch.float64, device=loss.device
            )
            dist.all_reduce(values, op=dist.ReduceOp.MAX)
            for k, value in zip(keys, values.tolist(), strict=True):
                if row[k] is not None:
                    row[k] = value
        rows.append(row)
        if is_main:
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
                "skipped_updates": skipped_total,
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
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
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
        world = int(os.environ.get("WORLD_SIZE", "1"))
        local_batch = per_rank_batch(recipe.batch_size, world)
        source = PreTrainedConfig.from_pretrained(args.checkpoint, local_files_only=True)
        source = relocate_vlm(source, args.vlm_path)
        if not args.dry_run:
            source = local_vlm(source)
        policy = adapt_policy(source, recipe, args.checkpoint, precision=args.precision)
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
            batch_size=local_batch,
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
            "checkpoint": str(args.checkpoint),
            "dataset_root": str(args.dataset_root),
            "vlm_path": policy.vlm_model_name,
            "initialization_only": True,
            "precision": args.precision,
            "world_size": world,
            "global_batch_size": recipe.batch_size,
            "per_rank_batch_size": local_batch,
            "source_training_stage": source.training_stage,
            "training_stage": "stage2",
            "sampling": "random_execution_chunk",
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
