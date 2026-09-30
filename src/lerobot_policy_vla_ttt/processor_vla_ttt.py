"""Processor entry points delegated to LeRobot's Evo-1 implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from lerobot.policies.evo1.processor_evo1 import (
    make_evo1_pre_post_processors,
    reconcile_evo1_processors,
)
from lerobot.policies.factory import _reconnect_relative_absolute_steps, make_pre_post_processors
from lerobot.processor import (
    PolicyProcessorPipeline,
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
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
    force_download: bool = False,
    resume_download: bool | None = None,
    proxies: dict[str, str] | None = None,
    token: str | bool | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """Load processors through LeRobot's generic loader and Evo-1 reconciliation path."""
    needs_explicit_hub_options = any(
        (
            force_download,
            resume_download is not None,
            proxies is not None,
            token is not None,
            cache_dir is not None,
            local_files_only,
        )
    )
    if needs_explicit_hub_options:
        common_kwargs = {
            "pretrained_model_name_or_path": str(pretrained_path),
            "force_download": force_download,
            "resume_download": resume_download,
            "proxies": proxies,
            "token": token,
            "cache_dir": cache_dir,
            "local_files_only": local_files_only,
            "revision": revision,
        }
        preprocessor = PolicyProcessorPipeline.from_pretrained(
            **common_kwargs,
            config_filename=preprocessor_config_filename,
            overrides=preprocessor_overrides or {},
            to_transition=batch_to_transition,
            to_output=transition_to_batch,
        )
        postprocessor = PolicyProcessorPipeline.from_pretrained(
            **common_kwargs,
            config_filename=postprocessor_config_filename,
            overrides=postprocessor_overrides or {},
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        )
        _reconnect_relative_absolute_steps(preprocessor, postprocessor)
        return reconcile_evo1_processors(config, preprocessor, postprocessor)

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
