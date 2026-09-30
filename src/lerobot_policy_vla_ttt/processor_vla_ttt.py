"""Processor entry points delegated to LeRobot's Evo-1 implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from lerobot.policies.evo1.processor_evo1 import make_evo1_pre_post_processors
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import (
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from .configuration_vla_ttt import VLATTTConfig


def make_vla_ttt_pre_post_processors(
    config: VLATTTConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """Build processors through the upstream Evo-1 factory."""
    return make_evo1_pre_post_processors(config, dataset_stats=dataset_stats)


def make_vla_ttt_pre_post_processors_from_pretrained(
    config: VLATTTConfig,
    pretrained_path: str | Path,
    *,
    revision: str | None = None,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    dataset_meta: Any | None = None,
    preprocessor_overrides: dict[str, Any] | None = None,
    postprocessor_overrides: dict[str, Any] | None = None,
    preprocessor_config_filename: str = f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
    postprocessor_config_filename: str = f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """Load processors through LeRobot's generic loader and Evo-1 reconciliation path."""
    return make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(pretrained_path),
        pretrained_revision=revision,
        dataset_stats=dataset_stats,
        dataset_meta=dataset_meta,
        preprocessor_overrides=preprocessor_overrides,
        postprocessor_overrides=postprocessor_overrides,
        preprocessor_config_filename=preprocessor_config_filename,
        postprocessor_config_filename=postprocessor_config_filename,
    )
