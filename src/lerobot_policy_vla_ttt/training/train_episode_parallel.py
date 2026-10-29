"""Independent Accelerate episode-parallel trainer for LIBERO and RoboMME memory recipes.

Only outer slow gradients are reduced. --dry-run reads configs, never weights.
Use --cpu for CPU smoke tests; GPU execution must be explicitly scheduled by the user.
"""

import argparse
import dataclasses
import hashlib
import json
import os
import sys
from pathlib import Path

from lerobot.configs.policies import PreTrainedConfig

from ..models.evo1_groot.configuration_evo1_groot import Evo1GrootConfig
from ..portability import (
    configure_cache,
    content_digest,
    local_vlm,
    read_machine,
    relocate_vlm,
    resolve_machine,
)
from .episode_parallel.checkpoint import canonical, normalize_saved_recipe
from .episode_parallel.config import Recipe
from .train_robottt import ARCHITECTURES

KINDS = ("evo1_groot", "mac", "mag", "mal", "aqmem", "t2mem")
LIBERO_REVISION = "86958911c0f959db2bbbdb107eb3e17c5f9c798e"


def resolve_config(source, architecture, recipe, *, device, overrides=None):
    if architecture not in KINDS or source.type not in (architecture, "evo1_groot"):
        raise ValueError("Expected evo1_groot or a checkpoint of the selected architecture")
    baseline = architecture == "evo1_groot"
    if baseline and recipe.training_mode != "policy":
        raise ValueError("evo1_groot requires training_mode=policy")
    if not baseline and recipe.training_mode == "policy":
        raise ValueError("training_mode=policy is only supported for evo1_groot")
    if baseline and overrides:
        raise ValueError("evo1_groot does not accept memory overrides")
    cls = Evo1GrootConfig if baseline else ARCHITECTURES[architecture][0]
    values = {f.name: getattr(source, f.name) for f in dataclasses.fields(source)}
    changes = dict(overrides or {})
    # Overrides describe new memory structure only, never silently alter a backbone.
    allowed = {f.name for f in dataclasses.fields(cls)} - {
        f.name for f in dataclasses.fields(Evo1GrootConfig)
    }
    allowed |= {"vlm_layer_indices"}
    if set(changes) - allowed:
        raise ValueError(f"Unsupported memory overrides: {sorted(set(changes) - allowed)}")
    if source.type != "evo1_groot" and changes:
        raise ValueError("Memory overrides apply only when initializing from baseline")
    values.update(changes)
    values.update(
        # LeRobot capability checks take a device type, not an indexed device.
        # CUDA's current device is set from Accelerate before model creation.
        device=str(device).split(":")[0],
        dtype="bfloat16",
        vlm_dtype="float32",
        use_amp=True,
        use_flash_attn=False,
        training_stage="stage2",
        apply_training_stage_defaults=True,
        finetune_vlm=True,
        finetune_language_model=True,
        finetune_vision_model=True,
        finetune_action_head=True,
        memory_training="joint" if recipe.training_mode == "joint" else "memory_only",
        memory_update_protocol="half_execution",
        n_action_steps=recipe.execution,
        enable_gradient_checkpointing=True,
        gradient_checkpointing_use_reentrant=False,
        push_to_hub=False,
    )
    if baseline:
        for key in ("memory_training", "memory_update_protocol"):
            values.pop(key, None)
    if source.type == "evo1_groot" and not baseline:
        values.setdefault("memory_update_rule", "t2mem_adaptive")
    if values["chunk_size"] != recipe.horizon:
        raise ValueError("Recipe horizon must match checkpoint chunk_size")
    return cls(**values)


def augmentation_config(values):
    from lerobot.transforms import ImageTransformsConfig
    from lerobot.transforms.transforms import ImageTransformConfig

    values = dict(values)
    if "tfs" in values:
        values["tfs"] = {k: ImageTransformConfig(**v) for k, v in values["tfs"].items()}
    return ImageTransformsConfig(**values)


def metadata_digest(root):
    """Hash small metadata files, never scan videos or frame parquet payloads."""
    digest = hashlib.sha256()
    for path in sorted((root / "meta").rglob("*")):
        if path.is_file() and path.suffix in (".json", ".jsonl", ".parquet"):
            digest.update(str(path.relative_to(root)).encode())
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--checkpoint", type=Path, help="Weight initialization, not exact resume")
    sources.add_argument("--resume", type=Path, help="Trusted complete trainer checkpoint")
    parser.add_argument("--architecture", choices=KINDS)
    parser.add_argument("--machine", type=Path, help="Machine-local TOML paths")
    parser.add_argument("--run-name", help="New directory name under machine storage.runs_root")
    parser.add_argument(
        "--vlm-path", type=Path, help="Local VLM override, without editing checkpoints"
    )
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--benchmark", choices=("libero", "robomme"))
    parser.add_argument(
        "--demo-stride", type=int, help="RoboMME demo frame stride; defaults to E/2"
    )
    parser.add_argument("--dataset-repo")
    parser.add_argument("--revision")
    parser.add_argument(
        "--output", type=Path, help="New directory, including on resume; alternative to --run-name"
    )
    parser.add_argument(
        "--recipe", type=Path, help="JSON Recipe overrides; defaults follow LIBERO recipe"
    )
    parser.add_argument(
        "--memory-config", type=Path, help="JSON memory structure overrides for baseline init"
    )
    parser.add_argument(
        "--augmentation-config", type=Path, help="JSON LeRobot image transform config"
    )
    parser.add_argument(
        "--stop-after", type=int, help="Stop after this many updates, retaining full schedule"
    )
    parser.add_argument(
        "--progress",
        choices=["auto", "bar", "plain", "off"],
        default="auto",
        help="Console progress only; auto selects bar on TTY, periodic text otherwise",
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        machine = read_machine(args.machine)
        configure_cache(
            machine, __spec__.name if __spec__ else __name__, sys.argv[1:] if argv is None else argv
        )
        resume_state = None
        if args.resume:
            args.resume = args.resume.expanduser().resolve()
            import torch

            if not (args.resume / "COMPLETE").is_file():
                raise ValueError("Incomplete resume checkpoint")
            resume_state = torch.load(
                args.resume / "training.pt", map_location="cpu", weights_only=False
            )
        old_identity = resume_state["identity"] if resume_state else {}
        protocol = old_identity.get("sequence_protocol", {})
        benchmark = args.benchmark or protocol.get("benchmark", "libero")
        args.benchmark = benchmark
        machine = resolve_machine(args)
        if not (args.checkpoint or args.resume) or args.dataset_root is None or args.output is None:
            raise ValueError(
                "Provide checkpoint/resume, dataset-root and output via CLI or --machine"
            )
        if args.output.exists():
            raise ValueError("Use a new output directory")
        if args.stop_after is not None and args.stop_after < 1:
            raise ValueError("stop-after must be positive")
        checkpoint = args.checkpoint or (args.resume / "pretrained_model")
        source = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
        source = relocate_vlm(source, args.vlm_path)
        portable = resume_state is None or resume_state["identity"].get("version") == 2
        if not args.dry_run and portable:
            source = local_vlm(source)
        architecture = args.architecture or (
            source.type
            if source.type in KINDS and (source.type != "evo1_groot" or args.resume)
            else "t2mem"
        )
        from ..baselines.prepare import ROBOMME_REVISION

        old_dataset = old_identity.get("dataset", {})
        args.dataset_repo = (
            args.dataset_repo
            or old_dataset.get("repo")
            or ("lerobot/robomme" if benchmark == "robomme" else "HuggingFaceVLA/libero")
        )
        args.revision = (
            args.revision
            or old_dataset.get("revision")
            or (ROBOMME_REVISION if benchmark == "robomme" else LIBERO_REVISION)
        )
        recipe_values = (
            resume_state["recipe"]
            if resume_state
            else (
                {"horizon": source.chunk_size, "execution": source.n_action_steps}
                if benchmark == "robomme"
                else {}
            )
        )
        if args.recipe:
            recipe_values = recipe_values | json.loads(args.recipe.read_text())
        recipe = Recipe(**recipe_values)
        demo_stride = (
            args.demo_stride
            if args.demo_stride is not None
            else protocol.get("demo_stride", recipe.execution // 2)
        )
        if demo_stride < 1 or (benchmark == "libero" and args.demo_stride is not None):
            raise ValueError("demo-stride must be positive and is only supported for RoboMME")
        args.benchmark, args.demo_stride = benchmark, demo_stride
        world = int(os.environ.get("WORLD_SIZE", "1"))
        if recipe.global_batch_size % world:
            raise ValueError("global_batch_size must be divisible by WORLD_SIZE")
        overrides = json.loads(args.memory_config.read_text()) if args.memory_config else {}
        # CPU here avoids GPU discovery during --dry-run.
        resolved = resolve_config(source, architecture, recipe, device="cpu", overrides=overrides)
        info = json.loads((args.dataset_root / "meta/info.json").read_text())
        action_key, action_dim = ("actions", 8) if benchmark == "robomme" else ("action", 7)
        if info["features"][action_key]["shape"] != [action_dim]:
            raise ValueError(f"Dataset action dimension must be {action_dim} for {benchmark}")
        if source.output_features and source.output_features["action"].shape != (action_dim,):
            raise ValueError(f"Checkpoint action dimension must match {benchmark}")
        if args.augmentation_config:
            transforms = json.loads(args.augmentation_config.read_text())
        elif resume_state:
            transforms = resume_state["identity"]["augmentation"]
        elif (checkpoint / "train_config.json").is_file():
            transforms = json.loads((checkpoint / "train_config.json").read_text())["dataset"][
                "image_transforms"
            ]
        else:
            raise ValueError(
                "No saved augmentation config; provide --augmentation-config explicitly"
            )
        transform_config = augmentation_config(transforms)
        identity = {
            "dataset": {
                "root": str(args.dataset_root.resolve()),
                "repo": args.dataset_repo,
                "revision": args.revision,
                "metadata_sha256": metadata_digest(args.dataset_root),
            },
            "augmentation": canonical(dataclasses.asdict(transform_config)),
            # Dict equality ignores order, but seeded transform selection does not.
            "augmentation_order": list(transform_config.tfs),
            "initialization": (
                resume_state["identity"]["initialization"]
                if resume_state
                else str(checkpoint.resolve())
            ),
        }
        if benchmark == "robomme":
            identity["sequence_protocol"] = {
                "benchmark": "robomme",
                "version": 1,
                "demo_stride": demo_stride,
                "demo_supervision": False,
                "reset_at_execution": False,
                "execution_clock": "exec_start_idx",
            }
        paths = {
            name: str(getattr(args, name)) if getattr(args, name) else None
            for name in ("checkpoint", "resume", "dataset_root", "output")
        }
        paths["vlm"] = source.vlm_model_name
        if portable:
            identity["version"] = 2
            identity["dataset"].pop("root")
            # Dry-run inspects small metadata only; it never claims byte validation.
            if not args.dry_run:
                print(
                    "Validating dataset, VLM and initialization content (SHA256)...",
                    file=sys.stderr,
                )
                identity["dataset"]["content_sha256"] = content_digest(args.dataset_root)
                identity["vlm_sha256"] = content_digest(Path(source.vlm_model_name))
                if resume_state is None:
                    identity["initialization"] = {"sha256": content_digest(checkpoint)}
        expected_identity = resume_state["identity"] if resume_state else None
        if resume_state and portable and args.dry_run:
            expected_identity = dict(expected_identity)
            expected_identity.pop("vlm_sha256", None)
            expected_identity["dataset"] = {
                k: v for k, v in expected_identity["dataset"].items() if k != "content_sha256"
            }
        if resume_state and (
            normalize_saved_recipe(resume_state["recipe"]) != dataclasses.asdict(recipe)
            or expected_identity != identity
            or resume_state["world_size"] != world
        ):
            raise ValueError("Resume recipe, identity or world_size mismatch")
        if args.dry_run:
            print(
                json.dumps(
                    canonical(
                        {
                            "recipe": dataclasses.asdict(recipe),
                            "policy": dataclasses.asdict(resolved),
                            "augmentation": dataclasses.asdict(transform_config),
                            "identity": identity,
                            "world_size": world,
                            "gradient_reduction": "explicit_sum",
                            "paths": paths,
                            "machine": machine,
                            "hf_home": os.environ.get("HF_HOME"),
                            "asset_validation": "metadata_only",
                        }
                    ),
                    indent=2,
                )
            )
            return
    except (ValueError, KeyError, OSError, TypeError) as exc:
        parser.error(str(exc))

    execute(
        args,
        source,
        architecture,
        recipe,
        overrides,
        checkpoint,
        info,
        transform_config,
        identity,
        machine,
        paths,
    )


def execute(
    args,
    source,
    architecture,
    recipe,
    overrides,
    checkpoint,
    info,
    transform_config,
    identity,
    machine,
    paths,
):

    from datetime import timedelta

    import torch
    from accelerate import Accelerator
    from accelerate.utils import InitProcessGroupKwargs

    from .episode_parallel.run_files import environment, run_directory

    accelerator = Accelerator(
        cpu=args.cpu,
        mixed_precision="no",
        gradient_accumulation_steps=1,
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(minutes=30))],
    )
    if accelerator.device.type != "cuda" and not args.cpu:
        raise RuntimeError("CUDA unavailable; use --cpu explicitly for CPU smoke tests")
    # Single-process Accelerate may return bare `cuda`; set_device and the
    # episode RNG context require a concrete index. Preserve assigned DDP ranks.
    if accelerator.device.type == "cuda" and accelerator.device.index is None:
        accelerator.state.device = torch.device("cuda", torch.cuda.current_device())
    with run_directory(args.output, accelerator):
        _execute_run(
            args,
            source,
            architecture,
            recipe,
            overrides,
            checkpoint,
            info,
            transform_config,
            identity,
            accelerator,
            {"machine": machine, "paths": paths, "environment": environment(accelerator)},
        )
    accelerator.end_training()


def _execute_run(
    args,
    source,
    architecture,
    recipe,
    overrides,
    checkpoint,
    info,
    transform_config,
    identity,
    accelerator,
    launch,
):
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.transforms import ImageTransforms

    from ..cli import make_pre_post_processors
    from ..models.evo1_groot.modeling_evo1_groot import Evo1GrootPolicy
    from .episode_parallel.data import Episode, EpisodeSource
    from .episode_parallel.runner import run
    from .episode_parallel.runtime import collective_error

    error = None
    try:
        # Set only this process's random streams and current CUDA device.
        torch.random.default_generator.manual_seed(recipe.seed)
        if accelerator.device.type == "cuda":
            torch.cuda.set_device(accelerator.device)
            torch.cuda.manual_seed(recipe.seed)
        resolved = resolve_config(
            source, architecture, recipe, device=accelerator.device, overrides=overrides
        )
        policy_cls = (
            Evo1GrootPolicy if architecture == "evo1_groot" else ARCHITECTURES[architecture][1]
        )
        if source.type == "evo1_groot" and architecture != "evo1_groot":
            baseline_config = dataclasses.replace(source, device="cpu", vlm_dtype="float32")
            baseline = Evo1GrootPolicy.from_pretrained(
                checkpoint, config=baseline_config, local_files_only=True
            )
            policy = policy_cls(resolved, vlm_hub_kwargs={"local_files_only": True})
            policy.initialize_from_baseline(baseline)
            del baseline
        else:
            policy = policy_cls.from_pretrained(checkpoint, config=resolved, local_files_only=True)
        policy.float()
        pre, post = make_pre_post_processors(
            resolved,
            pretrained_path=checkpoint,
            preprocessor_overrides={"device_processor": {"device": str(accelerator.device)}},
        )
        if args.benchmark == "robomme":
            from .episode_parallel.robomme import RoboMMEMemoryDataset

            dataset = RoboMMEMemoryDataset(
                args.dataset_root,
                horizon=recipe.horizon,
                revision=args.revision,
                repo_id=args.dataset_repo,
            )
            episodes = dataset.episodes
        else:
            dataset = LeRobotDataset(
                args.dataset_repo,
                root=args.dataset_root,
                revision=args.revision,
                delta_timestamps={"action": [i / info["fps"] for i in range(recipe.horizon)]},
                video_backend="pyav",
            )
            episodes = []
            for row in dataset.meta.episodes:
                if len(row["tasks"]) != 1:
                    raise ValueError("Recipe requires one task per episode")
                episodes.append(
                    Episode(
                        int(row["episode_index"]),
                        str(row["tasks"][0]),
                        int(row["dataset_from_index"]),
                        int(row["dataset_to_index"]),
                    )
                )
        source_data = EpisodeSource(
            dataset,
            episodes,
            execution=recipe.execution,
            horizon=recipe.horizon,
            transform=ImageTransforms(transform_config),
            demo_stride=args.demo_stride,
        )
    except Exception as exc:
        error = f"Initialization failed: {type(exc).__name__}: {exc}"
    collective_error(error, accelerator.device)
    run(
        policy,
        source_data,
        pre,
        post,
        recipe=recipe,
        accelerator=accelerator,
        output=args.output,
        identity=identity,
        resume=args.resume,
        stop_after=args.stop_after,
        progress_mode=getattr(args, "progress", "auto"),
        launch=launch,
        output_prepared=True,
    )


if __name__ == "__main__":
    main()
