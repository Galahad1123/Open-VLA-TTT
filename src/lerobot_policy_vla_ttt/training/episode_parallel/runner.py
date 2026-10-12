"""Single and multi-process outer loop, independent of the legacy trainer."""

import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist

from .checkpoint import canonical, load_checkpoint, model_contract, save_checkpoint
from .data import EpisodeSampler
from .engine import backward_episode, episode_rng
from .optim import PhaseOptimizers
from .runtime import broadcast_model, collective_error, distributed, synchronize_gradients

COUNTERS = ("episodes", "events", "decisions", "supervised_scalars")


def run(
    policy,
    source,
    pre,
    post,
    *,
    recipe,
    accelerator,
    output,
    identity,
    resume=None,
    stop_after=None,
):
    """Run recipe budget, or stop early at a resumable boundary for a smoke test."""
    if stop_after is not None and stop_after < 1:
        raise ValueError("stop_after must be positive")
    if accelerator.gradient_accumulation_steps != 1 or accelerator.mixed_precision != "no":
        raise ValueError(
            'Use Accelerator(mixed_precision="no", gradient_accumulation_steps=1); '
            "the policy owns BF16 autocast and the engine owns normalization"
        )
    sampler = EpisodeSampler(
        source.episodes.values(), seed=recipe.seed, global_batch_size=recipe.global_batch_size
    )
    sampler.local_batch(0, accelerator.process_index, accelerator.num_processes)
    policy.to(accelerator.device)
    broadcast_model(policy)
    control = PhaseOptimizers(policy, recipe)
    step, totals = 0, dict.fromkeys(COUNTERS, 0)
    if resume:
        step, totals = load_checkpoint(
            resume, policy, control, accelerator, recipe=recipe, identity=identity
        )
    if step >= recipe.steps:
        raise ValueError("Checkpoint already completed this recipe")
    end = recipe.steps if stop_after is None else min(recipe.steps, step + stop_after)
    output = Path(output)
    error = None
    if accelerator.is_main_process:
        try:
            output.mkdir(parents=True, exist_ok=False)
            manifest = {
                "recipe": asdict(recipe),
                "world_size": accelerator.num_processes,
                "microbatch_episodes": 1,
                "gradient_reduction": "explicit_sum",
                "local_episodes_per_update": recipe.global_batch_size // accelerator.num_processes,
                "identity": identity,
                "model_config": model_contract(policy),
                "resume": str(resume) if resume else None,
                "start_step": step,
                "device_type": accelerator.device.type,
            }
            (output / "run.json").write_text(json.dumps(canonical(manifest), indent=2) + "\n")
        except Exception as exc:
            error = f"Output initialization failed: {exc}"
    collective_error(error, accelerator.device)
    accelerator.wait_for_everyone()
    start_time = time.monotonic()
    while step < end:
        started = time.monotonic()
        local = dict.fromkeys(("loss_sum", *COUNTERS), 0.0)
        error = None
        try:
            phase = control.begin(step)
            for sample in sampler.local_batch(
                step, accelerator.process_index, accelerator.num_processes
            ):
                with episode_rng(sample.seed, accelerator.device):
                    metrics, memory = backward_episode(
                        policy,
                        source.events(sample.episode_id, seed=sample.seed),
                        decisions=source.decision_count(sample.episode_id),
                        recipe=recipe,
                        backward=accelerator.backward,
                        preprocess=pre,
                    )
                del memory
                for key in local:
                    local[key] += metrics[key]
        except Exception as exc:
            error = f"Local episode computation at step {step} failed: {type(exc).__name__}: {exc}"
        collective_error(error, accelerator.device)
        synchronize_gradients(policy)
        error = None
        try:
            grad_norm = control.step()
        except Exception as exc:
            error = f"Optimizer update failed: {exc}"
        collective_error(error, accelerator.device)
        step += 1
        values = torch.tensor(list(local.values()), dtype=torch.float64, device=accelerator.device)
        if distributed():
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
        metrics = dict(zip(local, values.tolist(), strict=True))
        metrics["loss"] = metrics.pop("loss_sum") / recipe.global_batch_size
        for key in COUNTERS:
            metrics[key] = int(metrics[key])
            totals[key] += metrics[key]
        metrics.update(
            step=step,
            phase=phase,
            phase_updates=control.counts.copy(),
            grad_norm=grad_norm,
            totals=totals.copy(),
            update_seconds=time.monotonic() - started,
            run_seconds=time.monotonic() - start_time,
            learning_rates={
                str(i): g["lr"] for i, g in enumerate(control.optimizers[phase].param_groups)
            },
        )
        error = None
        if accelerator.is_main_process:
            try:
                with (output / "metrics.jsonl").open("a") as log:
                    log.write(json.dumps(metrics, allow_nan=False) + "\n")
                print(json.dumps(metrics), flush=True)
            except Exception as exc:
                error = f"Metrics write failed: {exc}"
        collective_error(error, accelerator.device)
        if step % recipe.save_every == 0 or step == end:
            save_checkpoint(
                output / f"step-{step:08d}",
                policy,
                control,
                pre,
                post,
                accelerator,
                recipe=recipe,
                identity=identity,
                totals=totals,
            )
    return {"step": step, "counts": control.counts.copy(), "totals": totals.copy()}
