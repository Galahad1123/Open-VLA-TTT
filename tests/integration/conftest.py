from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import torch
from lerobot.configs.types import FeatureType
from lerobot.policies.evo1.configuration_evo1 import Evo1Config


@pytest.fixture
def evo1_checkpoint() -> tuple[str, str | None]:
    checkpoint = os.getenv("VLA_TTT_EVO1_CHECKPOINT")
    if not checkpoint:
        pytest.skip(
            "VLA_TTT_EVO1_CHECKPOINT is not set; integration tests never download "
            "a model implicitly"
        )
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support bf16")

    local_checkpoint = Path(checkpoint)
    revision = os.getenv("VLA_TTT_EVO1_REVISION")
    if not local_checkpoint.exists():
        if not revision:
            pytest.fail("VLA_TTT_EVO1_REVISION is required when the checkpoint is a Hub ID")
        if os.getenv("VLA_TTT_ALLOW_DOWNLOAD") != "1":
            pytest.skip(
                "Hub checkpoint is not cached and VLA_TTT_ALLOW_DOWNLOAD=1 is absent; "
                "report the dry-run size to the user before enabling downloads"
            )
    return checkpoint, revision


def make_synthetic_raw_batch(
    config: Evo1Config,
    *,
    batch_size: int = 1,
) -> dict[str, Any]:
    """Build deterministic raw observations from the checkpoint's declared features."""
    if batch_size != 1:
        raise ValueError("The raw Evo-1 integration fixture currently supports batch_size=1 only")

    batch: dict[str, Any] = {}
    visual_features = 0
    for key, feature in (config.input_features or {}).items():
        if feature.type in {FeatureType.VISUAL, FeatureType.STATE}:
            batch[key] = torch.zeros(feature.shape, dtype=torch.float32)
        if feature.type is FeatureType.VISUAL:
            visual_features += 1
    if visual_features == 0:
        raise ValueError("The Evo-1 checkpoint declares no visual input feature")
    batch[config.task_field] = "move the robot deterministically"
    return batch


@pytest.fixture
def synthetic_batch_factory():
    return make_synthetic_raw_batch
