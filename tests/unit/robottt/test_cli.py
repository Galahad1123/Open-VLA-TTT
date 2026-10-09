"""Actual LeRobot image dataset -> processors -> sequential optimizer -> checkpoint."""

import dataclasses
import json
import sys

import numpy as np
import pytest
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from lerobot_policy_vla_ttt.cli import make_pre_post_processors
from lerobot_policy_vla_ttt.models.evo1_groot import model as baseline_model
from lerobot_policy_vla_ttt.models.evo1_groot.configuration_evo1_groot import Evo1GrootConfig
from lerobot_policy_vla_ttt.models.evo1_groot.modeling_evo1_groot import Evo1GrootPolicy
from lerobot_policy_vla_ttt.models.robottt import model
from lerobot_policy_vla_ttt.models.robottt.modeling_robottt import RoboTTTPolicy
from lerobot_policy_vla_ttt.training.train_robottt import main


@pytest.mark.parametrize("segment_transfer", [False, True])
def test_cli_with_real_local_dataset(config, monkeypatch, tmp_path, segment_transfer):
    import datasets

    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", tmp_path / "hf-cache")
    monkeypatch.setattr(baseline_model, "InternVL3Embedder", model.InternVL3Embedder)
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
            "1",
            "--batch-size",
            "2",
            "--sequence-length",
            "3",
            "--decision-stride",
            "1",
            "--tbptt-length",
            "2",
            "--memory-dim",
            "4",
            "--memory-hidden-dim",
            "6",
        ],
    )
    if segment_transfer:
        sys.argv.append("--segment-transfer")
    main()
    loaded = RoboTTTPolicy.from_pretrained(output / "step-00000001", local_files_only=True)
    assert loaded.config.num_register_tokens == 16
    manifest = json.loads((output / "run.json").read_text())
    assert manifest["segment_transfer"] is segment_transfer
    metrics = json.loads((output / "metrics.jsonl").read_text())
    assert metrics["segment_transfer"] is segment_transfer
    assert manifest["overlapping_label_path"] is True
    assert manifest["initialization"]["copied"]
