import dataclasses
import json
import sys

import numpy as np
import pytest
import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from lerobot_policy_vla_ttt.cli import make_pre_post_processors
from lerobot_policy_vla_ttt.models.aqmem.configuration_aqmem import AQMemConfig
from lerobot_policy_vla_ttt.models.aqmem.modeling_aqmem import AQMemPolicy
from lerobot_policy_vla_ttt.models.evo1_groot.configuration_evo1_groot import Evo1GrootConfig
from lerobot_policy_vla_ttt.models.evo1_groot.modeling_evo1_groot import Evo1GrootPolicy
from lerobot_policy_vla_ttt.models.t2mem.configuration_t2mem import T2MemConfig
from lerobot_policy_vla_ttt.models.t2mem.modeling_t2mem import T2MemPolicy
from lerobot_policy_vla_ttt.training.train_robottt import main

from .assets import patch_assets
from .test_policy import policy_config


@pytest.mark.parametrize(
    "architecture,cls,policy_cls",
    [("aqmem", AQMemConfig, AQMemPolicy), ("t2mem", T2MemConfig, T2MemPolicy)],
)
def test_cli_real_dataset(monkeypatch, tmp_path, architecture, cls, policy_cls):
    segment_transfer = True
    import datasets

    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", tmp_path / "hf-cache")
    patch_assets(monkeypatch)
    config = policy_config(
        cls,
        chunk_size=6,
        n_action_steps=4,
        max_state_dim=8,
        max_action_dim=8,
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (8,)),
            "observation.images.image": PolicyFeature(FeatureType.VISUAL, (3, 16, 16)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (7,))},
    )
    values = {f.name: getattr(config, f.name) for f in dataclasses.fields(Evo1GrootConfig)}
    baseline = Evo1GrootPolicy(Evo1GrootConfig(**(values | {"num_register_tokens": 0})))
    source = tmp_path / "source"
    baseline.save_pretrained(source)
    stats = {
        k: {"min": -torch.ones(n), "max": torch.ones(n)}
        for k, n in [("action", 7), ("observation.state", 8)]
    }
    pre, post = make_pre_post_processors(baseline.config, dataset_stats=stats)
    pre.save_pretrained(source)
    post.save_pretrained(source)
    features = {
        "observation.state": {"dtype": "float32", "shape": (8,), "names": None},
        "action": {"dtype": "float32", "shape": (7,), "names": None},
    }
    features.update(
        {
            f"observation.images.{camera}": {
                "dtype": "image",
                "shape": (16, 16, 3),
                "names": ["height", "width", "channels"],
            }
            for camera in ["image", "image2"]
        }
    )
    root = tmp_path / "data"
    dataset = LeRobotDataset.create(
        "test/robottt", fps=10, root=root, features=features, use_videos=False
    )
    for _episode in range(2):
        for t in range(4):
            frame = {
                "observation.state": np.zeros(8, np.float32),
                "action": np.full(7, t / 10, np.float32),
                "task": "move",
            }
            frame.update(
                {
                    f"observation.images.{camera}": np.zeros((16, 16, 3), np.uint8)
                    for camera in ["image", "image2"]
                }
            )
            dataset.add_frame(frame)
        dataset.save_episode()
    dataset.finalize()
    output = tmp_path / "run"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_robottt",
            "--architecture",
            architecture,
            "--memory-training",
            "alternating",
            "--phase-length",
            "1",
            "--checkpoint",
            str(source),
            "--dataset-root",
            str(root),
            "--dataset-repo",
            "test/robottt",
            "--benchmark",
            "libero",
            "--output",
            str(output),
            "--device",
            "cpu",
            "--steps",
            "2",
            "--batch-size",
            "2",
            "--sequence-length",
            "3",
            "--decision-stride",
            "1",
            "--tbptt-length",
            "2",
            "--memory-heads",
            "2",
            "--memory-head-dim",
            "4",
            "--memory-hidden-dim",
            "6",
        ],
    )
    if segment_transfer:
        sys.argv.append("--segment-transfer")
    main()
    loaded = policy_cls.from_pretrained(output / "step-00000002", local_files_only=True)
    assert loaded.config.num_register_tokens == 0
    manifest = json.loads((output / "run.json").read_text())
    assert manifest["segment_transfer"] is segment_transfer
    rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert [r["memory_training"] for r in rows] == ["memory_only", "policy_only"]
    metrics = rows[-1]
    assert metrics["segment_transfer"] is segment_transfer
    assert manifest["overlapping_label_path"] is False
    assert manifest["observation_stride"] == 2
    assert manifest["initialization"]["copied"]
