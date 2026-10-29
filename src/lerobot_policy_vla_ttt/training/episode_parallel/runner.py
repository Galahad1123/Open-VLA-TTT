"""Single and multi-process outer loop, independent of the legacy trainer."""

import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist

from .checkpoint import canonical, load_checkpoint, model_contract, save_checkpoint
from .data import EpisodeSampler, MixedTaskEpisodeSampler
from .engine import backward_episode, episode_rng
from .microbatch import backward_microbatch
from .optim import PhaseOptimizers
from .runtime import broadcast_model, collective_error, distributed, synchronize_gradients
from .streams import ChunkStreamSampler, backward_chunks, prepare_chunks

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
    progress_mode="auto",
    launch=None,
    output_prepared=False,
):
    """Run recipe budget, or stop early at a resumable boundary for a smoke test."""
    if stop_after is not None and stop_after < 1:
        raise ValueError("stop_after must be positive")
    if accelerator.gradient_accumulation_steps != 1 or accelerator.mixed_precision != "no":
        raise ValueError(
            'Use Accelerator(mixed_precision="no", gradient_accumulation_steps=1); '
            "the policy owns BF16 autocast and the engine owns normalization"
        )
    chunk_updates = recipe.update_unit == "chunk"
    if chunk_updates and (policy.config.type != "evo1_groot" or recipe.training_mode != "policy"):
        raise ValueError("Chunk updates require stateless evo1_groot policy training")
    if recipe.global_batch_size % accelerator.num_processes:
        raise ValueError("global batch must be divisible by world size")
    sampler_cls = (
        ChunkStreamSampler
        if chunk_updates
        else (MixedTaskEpisodeSampler if recipe.sampling_mode == "mixed_tasks" else EpisodeSampler)
    )
    sampler = sampler_cls(
        source if chunk_updates else source.episodes.values(),
        seed=recipe.seed,
        global_batch_size=recipe.global_batch_size,
    )
    policy.to(accelerator.device)
    broadcast_model(policy)
    control = PhaseOptimizers(policy, recipe)
    step, totals = 0, dict.fromkeys(COUNTERS, 0)
    if resume:
        step, totals = load_checkpoint(
            resume, policy, control, accelerator, recipe=recipe, identity=identity
        )
    if chunk_updates:
        sampler.seek(step)
    if step >= recipe.steps:
        raise ValueError("Checkpoint already completed this recipe")
    end = recipe.steps if stop_after is None else min(recipe.steps, step + stop_after)
    output = Path(output)
    error = None
    if accelerator.is_main_process:
        try:
            output.mkdir(parents=True, exist_ok=output_prepared)
            if (output / "run.json").exists() or (output / "metrics.jsonl").exists():
                raise ValueError("Output already contains a run")
            manifest = {
                "recipe": asdict(recipe),
                "world_size": accelerator.num_processes,
                "microbatch_episodes": recipe.episode_microbatch_size or 1,
                "flow_rng_protocol": "row_event_v1" if recipe.episode_microbatch_size else "legacy",
                "update_unit": recipe.update_unit,
                "loss_normalization": "valid_elements" if chunk_updates else "equal_episodes",
                "sampler_protocol": "persistent_streams_v1" if chunk_updates else "episode_v1",
                "gradient_reduction": "explicit_sum",
                (
                    "local_chunks_per_update" if chunk_updates else "local_episodes_per_update"
                ): recipe.global_batch_size // accelerator.num_processes,
                "identity": identity,
                "model_config": model_contract(policy),
                "resume": str(resume) if resume else None,
                "start_step": step,
                "device_type": accelerator.device.type,
                "launch": launch,
            }
            (output / "run.json").write_text(json.dumps(canonical(manifest), indent=2) + "\n")
        except Exception as exc:
            error = f"Output initialization failed: {exc}"
    collective_error(error, accelerator.device)
    accelerator.wait_for_everyone()
    from .diagnostics import Diagnostics
    from .progress import Progress

    progress = Progress(recipe, step, end, mode=progress_mode, enabled=accelerator.is_main_process)
    start_time = time.monotonic()
    with (
        progress,
        Diagnostics(recipe.diagnostics_every, accelerator.is_main_process) as diagnostics,
    ):
        while step < end:
            started = time.monotonic()
            local = dict.fromkeys(("loss_sum", *COUNTERS), 0.0)
            error = None
            try:
                phase = control.begin(step)
                diagnostics.begin(step + 1)
                global_samples = sampler.batch(step)
                task_counts = Counter(source.episodes[s.episode_id].task for s in global_samples)
                local_samples = global_samples[
                    accelerator.process_index :: accelerator.num_processes
                ]
                prepared = (
                    prepare_chunks(policy, source, local_samples, pre) if chunk_updates else None
                )
            except Exception as exc:
                error = (
                    f"Local batch preparation at step {step} failed: {type(exc).__name__}: {exc}"
                )
            collective_error(error, accelerator.device)
            if chunk_updates:
                valid_count = torch.tensor(
                    sum(n for _, _, n in prepared), device=accelerator.device, dtype=torch.int64
                )
                if distributed():
                    dist.all_reduce(valid_count)
                global_valid_count = int(valid_count)
            error = None
            try:
                if chunk_updates:
                    local = backward_chunks(
                        policy,
                        prepared,
                        global_valid_count=global_valid_count,
                        backward=accelerator.backward,
                    )
                    del prepared
                elif recipe.episode_microbatch_size is not None:
                    size = recipe.episode_microbatch_size
                    for offset in range(0, len(local_samples), size):
                        samples = local_samples[offset : offset + size]
                        metrics, memory = backward_microbatch(
                            policy,
                            [source.events(s.episode_id, seed=s.seed) for s in samples],
                            decisions=[source.decision_count(s.episode_id) for s in samples],
                            seeds=[s.seed for s in samples],
                            recipe=recipe,
                            backward=accelerator.backward,
                            preprocess=pre,
                        )
                        del memory
                        for key in local:
                            local[key] += metrics[key]
                else:
                    for sample in local_samples:
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
                error = (
                    f"Local episode computation at step {step} failed: {type(exc).__name__}: {exc}"
                )
            collective_error(error, accelerator.device)
            synchronize_gradients(policy)
            error = None
            try:
                diagnostics.before_step(control.optimizers[phase], policy)
                grad_norm = control.step()
                diagnostic_row = diagnostics.after_step()
            except Exception as exc:
                error = f"Optimizer update failed: {exc}"
            collective_error(error, accelerator.device)
            step += 1
            values = torch.tensor(
                list(local.values()), dtype=torch.float64, device=accelerator.device
            )
            if distributed():
                dist.all_reduce(values, op=dist.ReduceOp.SUM)
            metrics = dict(zip(local, values.tolist(), strict=True))
            metrics["loss"] = metrics.pop("loss_sum") / (
                metrics["supervised_scalars"] if chunk_updates else recipe.global_batch_size
            )
            for key in COUNTERS:
                metrics[key] = int(metrics[key])
                totals[key] += metrics[key]
            if accelerator.device.type == "cuda":
                metrics["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
                metrics["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
            metrics.update(
                step=step,
                phase=phase,
                phase_updates=control.counts.copy(),
                sampled_episode_ids=[s.episode_id for s in global_samples],
                global_task_count=len(task_counts),
                global_task_counts=dict(task_counts),
                grad_norm=grad_norm,
                totals=totals.copy(),
                update_seconds=time.monotonic() - started,
                run_seconds=time.monotonic() - start_time,
                learning_rates={
                    str(i): g["lr"] for i, g in enumerate(control.optimizers[phase].param_groups)
                },
                optimizer_groups=[
                    {k: g[k] for k in ("category", "lr", "peak_lr", "weight_decay")}
                    for g in control.optimizers[phase].param_groups
                ],
            )
            if chunk_updates:
                metrics["stream_positions"] = [
                    {
                        "slot": sample.slot,
                        "episode_id": sample.episode_id,
                        "decision": sample.decision,
                    }
                    for sample in global_samples
                ]
            error = None
            if accelerator.is_main_process:
                try:
                    with (output / "metrics.jsonl").open("a") as log:
                        log.write(json.dumps(metrics, allow_nan=False) + "\n")
                    if diagnostic_row is not None:
                        with (output / "diagnostics.jsonl").open("a") as log:
                            log.write(
                                json.dumps(
                                    dict(step=step, phase=phase, **diagnostic_row), allow_nan=False
                                )
                                + "\n"
                            )
                    progress.update(metrics)
                except Exception as exc:
                    error = f"Metrics write failed: {exc}"
            collective_error(error, accelerator.device)
            if step % recipe.save_every == 0 or step == end:
                progress.saving()
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
                progress.saved()
    return {"step": step, "counts": control.counts.copy(), "totals": totals.copy()}
