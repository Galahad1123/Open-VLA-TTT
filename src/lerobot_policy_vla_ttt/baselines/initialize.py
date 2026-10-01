"""Strict weight warm-start with benchmark-specific action dimensions and fresh statistics."""

import argparse
import dataclasses
import json
from pathlib import Path

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.evo1.modeling_evo1 import Evo1Policy

from ..cli import make_pre_post_processors
from ..modeling_vla_ttt import VLATTTPolicy


def initialize(source, dataset_root, dataset_repo, benchmark, implementation, output):
    source, output = Path(source), Path(output)
    if not source.is_dir() or output.exists():
        raise ValueError("Source must exist locally; destination must not exist")
    metadata = LeRobotDatasetMetadata(dataset_repo, root=dataset_root)
    action_dim = 7 if benchmark == "libero" else 8
    if metadata.features["action"]["shape"] != (action_dim,) and list(
        metadata.features["action"]["shape"]
    ) != [action_dim]:
        raise ValueError("Dataset action dimension mismatch")
    config = PreTrainedConfig.from_pretrained(source, local_files_only=True)
    if config.type != "evo1":
        raise ValueError("Warm-start source must be a native Evo-1 checkpoint")
    config = dataclasses.replace(
        config,
        output_features={"action": PolicyFeature(FeatureType.ACTION, (action_dim,))},
        postprocess_action_dim=action_dim,
        binarize_gripper=False,
        gripper_index=action_dim - 1,
        training_stage="stage1",
        n_action_steps=10,
    )
    if implementation == "native":
        policy = Evo1Policy.from_pretrained(
            source, config=config, local_files_only=True, strict=True
        )
    elif implementation == "local":
        policy = VLATTTPolicy.from_evo1_pretrained(source, config=config, local_files_only=True)
    else:
        raise ValueError("Unknown implementation")
    pre, post = make_pre_post_processors(policy.config, dataset_stats=metadata.stats)
    output.mkdir(parents=True)
    policy.save_pretrained(output)
    pre.save_pretrained(output)
    post.save_pretrained(output)
    (output / "baseline_provenance.json").write_text(
        json.dumps(
            {
                "source_checkpoint": str(source.resolve()),
                "dataset_root": str(Path(dataset_root).resolve()),
                "benchmark": benchmark,
                "implementation": implementation,
                "action_dim": action_dim,
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
    parser.add_argument("--source", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--dataset-repo", required=True)
    parser.add_argument("--benchmark", choices=["libero", "robomme"], required=True)
    parser.add_argument("--implementation", choices=["native", "local"], required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(initialize(**vars(args)))


if __name__ == "__main__":
    main()
