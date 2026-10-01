"""Standard LeRobot training, with explicit baseline configuration and no simulator imports."""

import argparse
import dataclasses
import json
from pathlib import Path

from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig

from ..cli import local_processor_routing


def make_config(policy, dataset, output, *, steps, batch_size, seed, stage):
    if steps <= 0 or batch_size <= 0:
        raise ValueError("steps and batch_size must be positive")
    policy = dataclasses.replace(
        policy,
        training_stage=stage,
        apply_training_stage_defaults=True,
        task_field="task",
        push_to_hub=False,
        scheduler_warmup_steps=min(300, max(0, steps // 10)),
    )
    return TrainPipelineConfig(
        policy=policy,
        dataset=dataset,
        output_dir=Path(output),
        seed=seed,
        steps=steps,
        batch_size=batch_size,
        num_workers=0,
        env=None,
        env_eval_freq=0,
        log_freq=1 if steps < 100 else 100,
        save_freq=max(1, steps // 5),
        cudnn_deterministic=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--benchmark", choices=["libero", "robomme"], required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-repo", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--episodes", type=int, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--stage", choices=["stage1", "stage2"], default="stage1")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.checkpoint.is_dir() or not args.dataset_root.is_dir():
        parser.error("Download/prepare checkpoint and dataset first; local directories required")
    info = json.loads((args.dataset_root / "meta/info.json").read_text())
    dimension = 7 if args.benchmark == "libero" else 8
    if info["features"].get("action", {}).get("shape") != [dimension]:
        parser.error("Dataset action schema mismatch: prepare canonical data first")
    if args.benchmark == "robomme" and not (args.dataset_root / "preparation.json").exists():
        parser.error("RoboMME requires audited preparation.json (no demo control labels)")
    config = PreTrainedConfig.from_pretrained(args.checkpoint, local_files_only=True)
    if config.output_features["action"].shape != (dimension,):
        parser.error("Adapt checkpoint and dataset statistics with baselines.initialize first")
    config.pretrained_path = str(args.checkpoint.resolve())
    dataset = DatasetConfig(
        repo_id=args.dataset_repo,
        root=str(args.dataset_root.resolve()),
        episodes=args.episodes,
        revision=args.revision,
        video_backend="pyav",
        use_imagenet_stats=False,
    )
    cfg = make_config(
        config,
        dataset,
        args.output,
        steps=args.steps,
        batch_size=args.batch_size,
        seed=args.seed,
        stage=args.stage,
    )
    if args.dry_run:
        print(json.dumps(cfg.to_dict(), indent=2, default=str))
        return
    from lerobot.scripts import lerobot_eval, lerobot_train

    # train(cfg) retains the upstream optimizer, loader, checkpoint and logging loop.
    with local_processor_routing(lerobot_train, lerobot_eval):
        lerobot_train.train(cfg)


if __name__ == "__main__":
    main()
