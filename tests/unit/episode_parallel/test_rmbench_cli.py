import json
import os
import subprocess
import sys
from pathlib import Path

from lerobot.configs.types import FeatureType, PolicyFeature

from lerobot_policy_vla_ttt.baselines.rmbench import UPSTREAM_REVISION
from lerobot_policy_vla_ttt.models.evo1_groot.configuration_evo1_groot import Evo1GrootConfig


def test_rmbench_dry_run_frozen_and_portable(tmp_path):
    checkpoint = tmp_path / "model"
    cfg = Evo1GrootConfig(
        device="cpu",
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(14,))},
    )
    cfg.save_pretrained(checkpoint)
    (checkpoint / "train_config.json").write_text(
        json.dumps({"dataset": {"image_transforms": {"enable": False}}})
    )
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text(
        json.dumps({"features": {"action": {"shape": [14]}}, "fps": 10})
    )
    (root / "preparation.json").write_text(json.dumps({"upstream_revision": UPSTREAM_REVISION}))
    repo = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "lerobot_policy_vla_ttt.training.train_episode_parallel",
            "--checkpoint",
            str(checkpoint),
            "--dataset-root",
            str(root),
            "--benchmark",
            "rmbench",
            "--tasks",
            "put_back_block",
            "swap_blocks",
            "--architecture",
            "aqmem",
            "--recipe",
            str(repo / "configs/ttt/aqmem/rmbench/recipe.json"),
            "--output",
            str(tmp_path / "out"),
            "--dry-run",
        ],
        env={
            **os.environ,
            "PYTHONPATH": str(repo / "src") + ":" + str(repo / "packages/eval-client/src"),
            "HF_HUB_OFFLINE": "1",
            "CUDA_VISIBLE_DEVICES": "",
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    result = json.loads(result.stdout)
    assert result["identity"]["sequence_protocol"]["tasks"] == ["put_back_block", "swap_blocks"]
    assert not result["policy"]["finetune_language_model"]
    assert not result["policy"]["finetune_vision_model"]
    assert result["policy"]["n_action_steps"] == 24
    assert not (tmp_path / "out").exists()
