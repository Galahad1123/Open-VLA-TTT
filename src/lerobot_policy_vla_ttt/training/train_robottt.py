"""Local sequential RoboTTT training. Run with python -m ...training.train_robottt.

--checkpoint initializes weights and normalization (warm start, not exact resume).
Trajectory state resets per sampled window; TBPTT carries state within a window.
"""

import argparse
import dataclasses
import json
import random
from pathlib import Path

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader

from ..cli import make_pre_post_processors
from ..models.evo1_groot.modeling_evo1_groot import Evo1GrootPolicy
from ..models.robottt.configuration_robottt import RoboTTTConfig
from ..models.robottt.modeling_robottt import RoboTTTPolicy
from .sequence import TrajectoryWindows, collate_windows
from .tbptt import train_sequence


def run_training(
    policy,
    pre,
    post,
    windows,
    output,
    *,
    steps,
    batch_size,
    segment_length,
    learning_rate,
    save_every,
    seed=1000,
    warmup_steps=0,
):
    if min(steps, batch_size, segment_length, save_every) < 1 or learning_rate <= 0:
        raise ValueError("Training sizes and learning rate must be positive")
    if not 0 <= warmup_steps <= steps:
        raise ValueError("warmup_steps must be between zero and steps")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metrics.jsonl").exists():
        raise ValueError("Output already contains a training run; choose a new directory")
    loader = DataLoader(
        windows,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_windows,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(
        policy.get_optim_params(),
        lr=learning_rate,
        betas=policy.config.optimizer_betas,
        eps=policy.config.optimizer_eps,
        weight_decay=policy.config.optimizer_weight_decay,
    )
    iterator = iter(loader)
    policy.train()
    with (output / "metrics.jsonl").open("w") as log:
        for step in range(1, steps + 1):
            try:
                sequence = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                sequence = next(iterator)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate * min(1.0, step / max(1, warmup_steps))
            frames = [pre(frame) for frame in sequence["frames"]]
            metrics, memory = train_sequence(
                policy,
                frames,
                optimizer,
                segment_length=segment_length,
                valid_mask=sequence["valid_mask"],
                grad_clip_norm=policy.config.optimizer_grad_clip_norm,
            )
            metrics.update(
                step=step,
                learning_rate=optimizer.param_groups[0]["lr"],
                memory_updates=None if memory is None else memory[0].position.tolist(),
            )
            line = json.dumps(metrics)
            log.write(line + "\n")
            log.flush()
            print(line, flush=True)
            if step % save_every == 0 or step == steps:
                checkpoint = output / f"step-{step:08d}"
                checkpoint.mkdir(exist_ok=False)
                policy.save_pretrained(checkpoint)
                pre.save_pretrained(checkpoint)
                post.save_pretrained(checkpoint)
                torch.save(
                    {"step": step, "optimizer": optimizer.state_dict()}, checkpoint / "optimizer.pt"
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-repo", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--benchmark", choices=["libero", "robomme"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--memory-training", choices=["memory_only", "joint"], default="memory_only"
    )
    parser.add_argument("--memory-dim", type=int, default=128)
    parser.add_argument("--memory-hidden-dim", type=int, default=256)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--decision-stride", type=int, default=10)
    parser.add_argument("--tbptt-length", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()
    if not args.checkpoint.is_dir() or not args.dataset_root.is_dir():
        parser.error("Pinned local checkpoint and dataset directories are required")
    if args.output.exists():
        parser.error("Use a new output directory")
    if (
        min(
            args.steps,
            args.batch_size,
            args.sequence_length,
            args.decision_stride,
            args.tbptt_length,
            args.save_every,
        )
        < 1
    ):
        parser.error("All training sizes must be positive")
    info = json.loads((args.dataset_root / "meta/info.json").read_text())
    dimension = 7 if args.benchmark == "libero" else 8
    if info["features"]["action"]["shape"] != [dimension]:
        parser.error("Dataset action dimension does not match benchmark")
    if args.benchmark == "robomme" and not (args.dataset_root / "preparation.json").is_file():
        parser.error("RoboMME requires audited canonical preparation.json")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    source = PreTrainedConfig.from_pretrained(args.checkpoint, local_files_only=True)
    if source.output_features["action"].shape != (dimension,):
        parser.error("Checkpoint action schema does not match dataset")
    common = dict(
        device=args.device,
        memory_training=args.memory_training,
        training_stage="stage2" if args.memory_training == "joint" else "stage1",
        apply_training_stage_defaults=True,
        push_to_hub=False,
    )
    mapping = None
    if source.type == "evo1_groot":
        values = {f.name: getattr(source, f.name) for f in dataclasses.fields(source)}
        config = RoboTTTConfig(
            **(
                values
                | common
                | dict(
                    num_register_tokens=16,
                    memory_dim=args.memory_dim,
                    memory_hidden_dim=args.memory_hidden_dim,
                )
            )
        )
        baseline_config = dataclasses.replace(source, device=args.device)
        baseline = Evo1GrootPolicy.from_pretrained(
            args.checkpoint, config=baseline_config, local_files_only=True
        )
        policy = RoboTTTPolicy(config)
        mapping = policy.initialize_from_baseline(baseline)
        del baseline
    elif source.type == "robottt":
        config = dataclasses.replace(source, **common)
        policy = RoboTTTPolicy.from_pretrained(
            args.checkpoint, config=config, local_files_only=True
        )
    else:
        parser.error("Expected evo1_groot or robottt checkpoint")
    policy.to(args.device)
    pre, post = make_pre_post_processors(
        config,
        pretrained_path=args.checkpoint,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    fps = info["fps"]
    dataset = LeRobotDataset(
        args.dataset_repo,
        root=args.dataset_root,
        revision=args.revision,
        delta_timestamps={"action": [i / fps for i in range(config.chunk_size)]},
        video_backend="pyav",
    )
    ranges = [
        (int(ep["dataset_from_index"]), int(ep["dataset_to_index"])) for ep in dataset.meta.episodes
    ]
    windows = TrajectoryWindows(
        dataset, ranges, length=args.sequence_length, stride=args.decision_stride
    )
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = vars(args) | {
        "initialization": mapping,
        "policy_type": config.type,
        "update_protocol": config.memory_update_protocol,
        "chunk_size": config.chunk_size,
        "n_action_steps": config.n_action_steps,
        "num_inference_timesteps": config.num_inference_timesteps,
        "overlapping_label_path": args.decision_stride < config.chunk_size,
        "resume_mode": "weights_and_processors_only",
        "dataset_info": info,
    }
    (args.output / "run.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    run_training(
        policy,
        pre,
        post,
        windows,
        args.output,
        steps=args.steps,
        batch_size=args.batch_size,
        segment_length=args.tbptt_length,
        learning_rate=args.learning_rate,
        save_every=args.save_every,
        seed=args.seed,
        warmup_steps=args.warmup_steps,
    )


if __name__ == "__main__":
    main()
