from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

from lerobot_policy_vla_ttt.configuration_vla_ttt import VLATTTConfig
from lerobot_policy_vla_ttt.modeling_vla_ttt import VLATTTPolicy


def test_standard_lerobot_surfaces_discover_plugin_without_dataset_wrapper(tmp_path: Path) -> None:
    config = VLATTTConfig(device="cpu", use_amp=False)
    config.save_pretrained(tmp_path)

    restored = PreTrainedConfig.from_pretrained(tmp_path)
    preprocessor, postprocessor = make_pre_post_processors(restored)

    assert isinstance(restored, VLATTTConfig)
    assert get_policy_class(restored.type) is VLATTTPolicy
    assert preprocessor.name == "policy_preprocessor"
    assert postprocessor.name == "policy_postprocessor"
    assert LeRobotDataset.__module__ == "lerobot.datasets.lerobot_dataset"


def test_lerobot_rollout_cli_is_available() -> None:
    executable = shutil.which("lerobot-rollout")
    assert executable is not None

    result = subprocess.run(
        [executable, "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "--strategy.type" in result.stdout
    assert "--policy.path" in result.stdout


@pytest.mark.integration
@pytest.mark.libero
def test_one_libero_episode_through_standard_lerobot_eval() -> None:
    checkpoint = os.getenv("VLA_TTT_LIBERO_CHECKPOINT")
    if not checkpoint:
        pytest.skip("VLA_TTT_LIBERO_CHECKPOINT is not set; no benchmark assets are downloaded")
    checkpoint_path = Path(checkpoint)
    revision = os.getenv("VLA_TTT_LIBERO_REVISION")
    if not checkpoint_path.exists():
        if not revision:
            pytest.fail("VLA_TTT_LIBERO_REVISION is required for a Hub checkpoint")
        if os.getenv("VLA_TTT_ALLOW_DOWNLOAD") != "1":
            pytest.skip("Hub downloads remain disabled until their dry-run size is reported")
    pytest.importorskip("libero", reason="install with: uv sync --extra libero")
    executable = shutil.which("vla-ttt-eval")
    assert executable is not None

    command = [
        executable,
        f"--policy.path={checkpoint}",
        "--env.type=libero",
        "--env.task=libero_spatial",
        "--env.task_ids=[0]",
        "--eval.n_episodes=1",
        "--eval.batch_size=1",
        "--seed=20260930",
    ]
    if revision:
        command.append(f"--policy.pretrained_revision={revision}")
    result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=1800)

    assert result.returncode == 0, result.stderr
