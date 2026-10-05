"""Initialize an independent evo1_groot checkpoint from a local VLM and dataset stats."""

import argparse
import json
from pathlib import Path

import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

from ..cli import make_pre_post_processors
from ..models.evo1_groot.configuration_evo1_groot import Evo1GrootConfig
from ..models.evo1_groot.modeling_evo1_groot import Evo1GrootPolicy


def validate_dataset_schema(features, benchmark):
    if benchmark not in {"libero", "robomme"}:
        raise ValueError("benchmark must be libero or robomme")
    action_dim = 7 if benchmark == "libero" else 8
    for name, width in (("action", action_dim), ("observation.state", 8)):
        if tuple(features.get(name, {}).get("shape", ())) != (width,):
            raise ValueError(f"Dataset {name} must have shape ({width},) for {benchmark}")
    for name in ("observation.images.image", "observation.images.image2"):
        shape = tuple(features.get(name, {}).get("shape", ()))
        if len(shape) != 3 or (shape[0] != 3 and shape[-1] != 3):
            raise ValueError(f"Dataset requires RGB camera {name}")
    return action_dim


def initialize(
    config, dataset_root, dataset_repo, benchmark, vlm_path, output, *, device="cuda", seed=1000
):
    output, dataset_root, vlm_path = Path(output), Path(dataset_root), Path(vlm_path)
    if output.exists():
        raise ValueError("Initialization destination must not exist")
    if not dataset_root.is_dir() or not vlm_path.is_dir():
        raise ValueError("Download/prepare local dataset and VLM directories first")
    if benchmark == "robomme" and not (dataset_root / "preparation.json").is_file():
        raise ValueError("RoboMME requires audited preparation.json (no demo control labels)")
    policy_config = PreTrainedConfig.from_pretrained(config, local_files_only=True)
    if not isinstance(policy_config, Evo1GrootConfig):
        raise ValueError("Initialization requires an evo1_groot config directory")
    metadata = LeRobotDatasetMetadata(dataset_repo, root=dataset_root)
    dimension = validate_dataset_schema(metadata.features, benchmark)
    if policy_config.output_features["action"].shape != (
        dimension,
    ) or policy_config.input_features["observation.state"].shape != (8,):
        raise ValueError("Preset dimensions do not match benchmark")
    policy_config.device = device
    policy_config.vlm_model_name = str(vlm_path.resolve())
    policy_config.pretrained_path = None
    policy_config.push_to_hub = False
    torch.manual_seed(seed)
    policy = Evo1GrootPolicy(policy_config, vlm_hub_kwargs={"local_files_only": True})
    pre, post = make_pre_post_processors(policy_config, dataset_stats=metadata.stats)
    output.mkdir(parents=True)
    policy.save_pretrained(output)
    pre.save_pretrained(output)
    post.save_pretrained(output)
    (output / "baseline_provenance.json").write_text(
        json.dumps(
            {
                "implementation": "evo1_groot",
                "benchmark": benchmark,
                "seed": seed,
                "preset": str(Path(config).resolve()),
                "vlm_path": str(vlm_path.resolve()),
                "dataset_root": str(dataset_root.resolve()),
                "dataset_repo": dataset_repo,
                "action_head_initialization": "random",
                "normalization": "fresh target dataset stats",
                "trained_on_target": False,
            },
            indent=2,
        )
        + "\n"
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, required=True, help="Directory containing config.json"
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-repo", required=True)
    parser.add_argument("--benchmark", choices=["libero", "robomme"], required=True)
    parser.add_argument(
        "--vlm-path", type=Path, required=True, help="Downloaded/pinned InternVL3 path"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1000)
    print(initialize(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
