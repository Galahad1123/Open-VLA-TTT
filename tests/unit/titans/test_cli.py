"""Local dataset -> real tiny InternVL -> sequence optimizer -> saved policy -> reload."""

import json
import sys
from dataclasses import fields

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from lerobot_policy_vla_ttt.cli import make_pre_post_processors
from lerobot_policy_vla_ttt.models.evo1_groot.configuration_evo1_groot import Evo1GrootConfig
from lerobot_policy_vla_ttt.models.evo1_groot.modeling_evo1_groot import Evo1GrootPolicy
from lerobot_policy_vla_ttt.training.train_robottt import main

from .conftest import stats
from .test_internvl import patch_assets


def test_sequence_cli_from_baseline_and_warm_start(setup, monkeypatch, tmp_path):
    import datasets

    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", tmp_path / "hf-cache")
    patch_assets(monkeypatch)
    cfg = setup.config
    source = tmp_path / "baseline"
    baseline = Evo1GrootPolicy(
        Evo1GrootConfig(**{f.name: getattr(cfg, f.name) for f in fields(Evo1GrootConfig)})
    )
    baseline.save_pretrained(source)
    pre, post = make_pre_post_processors(baseline.config, dataset_stats=stats())
    pre.save_pretrained(source)
    post.save_pretrained(source)
    root = tmp_path / "dataset"
    features = {
        "observation.state": {"dtype": "float32", "shape": (8,), "names": None},
        "action": {"dtype": "float32", "shape": (7,), "names": None},
        **{
            f"observation.images.{cam}": {
                "dtype": "image",
                "shape": (16, 16, 3),
                "names": ["height", "width", "channels"],
            }
            for cam in ("image", "image2")
        },
    }
    data = LeRobotDataset.create(
        "test/titans", fps=10, root=root, features=features, use_videos=False
    )
    for _ in range(2):
        for t in range(3):
            data.add_frame(
                {
                    "observation.state": np.zeros(8, np.float32),
                    "action": np.full(7, t / 10, np.float32),
                    "task": "move",
                    **{
                        f"observation.images.{cam}": np.full((16, 16, 3), t, np.uint8)
                        for cam in ("image", "image2")
                    },
                }
            )
        data.save_episode()
    data.finalize()
    for stage in range(2):
        output = tmp_path / f"run-{stage}"
        args = [
            "train_titans",
            "--architecture",
            cfg.type,
            "--checkpoint",
            str(source),
            "--dataset-root",
            str(root),
            "--dataset-repo",
            "test/titans",
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
            "--segment-transfer",
        ]
        if stage == 0:
            args += [
                "--memory-layers",
                "0",
                "1",
                "--memory-dim",
                "4",
                "--memory-hidden-dim",
                "6",
                "--memory-update-rule",
                "momentum_decay",
                "--memory-input",
                "visual",
                "--num-persistent-tokens",
                "2",
            ]
        monkeypatch.setattr(sys, "argv", args)
        main()
        source = output / "step-00000001"
        loaded = setup.policy_class.from_pretrained(source, local_files_only=True)
        assert loaded.config.memory_input == "visual"
        assert loaded.config.memory_update_rule == "momentum_decay"
        manifest = json.loads((output / "run.json").read_text())
        assert manifest["policy_type"] == cfg.type
        assert manifest["overlapping_label_path"] is False
        assert manifest["overlapping_action_targets"] is True
        metrics = json.loads((output / "metrics.jsonl").read_text())
        assert metrics["memory_updates"] == [3, 3]
        assert np.isfinite(metrics["loss"])
        assert loaded.inference_state is None
        assert (source / "optimizer.pt").is_file()
        assert all(torch.isfinite(p).all() for p in loaded.parameters())
