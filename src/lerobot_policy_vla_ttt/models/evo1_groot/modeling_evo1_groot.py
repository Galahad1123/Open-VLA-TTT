"""Independent policy identity; reuse the established Evo-1 I/O and training boundary."""

from dataclasses import asdict

import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy

from ...checkpoint import checkpoint_policy_type
from ...compatibility import ensure_supported_lerobot_version
from ...evo1.modeling_evo1 import Evo1Policy
from .configuration_evo1_groot import Evo1GrootConfig
from .model import Evo1GrootModel


class Evo1GrootPolicy(Evo1Policy):
    config_class = Evo1GrootConfig
    name = "evo1_groot"

    # Everything else describes model/input semantics and must match the saved
    # config even when its change would preserve every weight tensor's shape.
    _runtime_overrides = frozenset(
        {
            "device",
            "use_amp",
            "dtype",
            "vlm_dtype",
            "use_flash_attn",
            "vlm_model_name",
            "pretrained_path",
            "pretrained_revision",
            "push_to_hub",
            "repo_id",
            "private",
            "tags",
            "license",
            "training_stage",
            "n_action_steps",
            "num_inference_timesteps",
            "enable_gradient_checkpointing",
            "gradient_checkpointing_use_reentrant",
            "finetune_vlm",
            "finetune_language_model",
            "finetune_vision_model",
            "finetune_action_head",
            "apply_training_stage_defaults",
            "optimizer_lr",
            "optimizer_betas",
            "optimizer_eps",
            "optimizer_weight_decay",
            "optimizer_grad_clip_norm",
            "scheduler_warmup_steps",
            "scheduler_min_lr_ratio",
        }
    )

    def __init__(self, config, *, vlm_hub_kwargs=None, **kwargs):
        ensure_supported_lerobot_version()
        if not isinstance(config, Evo1GrootConfig):
            raise TypeError("Evo1GrootPolicy requires Evo1GrootConfig")
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        if len(config.image_features) > config.max_views:
            raise ValueError("Number of camera streams exceeds max_views")
        self.model = Evo1GrootModel(config, vlm_hub_kwargs=vlm_hub_kwargs)
        self.model.set_finetune_flags()
        self._keep_frozen_embedder_eval()
        self.init_rtc_processor()
        self.reset()

    def select_action(self, batch, **kwargs):
        if self.config.robomme_action_representation == "delta_joint":
            raise ValueError("delta_joint requires Evo1Inference to anchor the whole action chunk")
        return super().select_action(batch, **kwargs)

    def supports_rtc(self):
        return False

    @property
    def _amp_enabled(self):
        return self.config.dtype == "bfloat16" and self._device.type == "cuda"

    def _maybe_autocast(self):
        # Explicitly disable an enclosing autocast for float32 (or CPU). Generic
        # eval can otherwise enter fp16 based on the legacy use_amp flag.
        return torch.autocast(self._device.type, dtype=torch.bfloat16, enabled=self._amp_enabled)

    def init_rtc_processor(self):
        if self.config.rtc_config is not None:
            raise ValueError("evo1_groot does not support RTC")
        self.rtc_processor = None
        self.model.rtc_processor = None

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, **kwargs):
        if kwargs.get("strict", True) is not True:
            raise ValueError("evo1_groot checkpoint loading requires strict=True")
        kwargs["strict"] = True
        policy_type = checkpoint_policy_type(
            pretrained_name_or_path,
            **{
                key: kwargs[key]
                for key in ("revision", "local_files_only", "cache_dir", "token")
                if key in kwargs
            },
        )
        if policy_type != cls.name:
            raise ValueError(f"Checkpoint type {policy_type!r}; expected 'evo1_groot'")
        config = kwargs.get("config")
        if config is not None:
            if not isinstance(config, Evo1GrootConfig):
                raise TypeError("Evo1GrootPolicy requires Evo1GrootConfig")
            saved = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path,
                **{
                    key: kwargs[key]
                    for key in (
                        "revision",
                        "local_files_only",
                        "cache_dir",
                        "token",
                        "force_download",
                        "resume_download",
                        "proxies",
                    )
                    if key in kwargs
                },
            )
            supplied = asdict(config)
            mismatches = [
                key
                for key, value in asdict(saved).items()
                if key not in cls._runtime_overrides and supplied[key] != value
            ]
            if list(config.image_features) != list(saved.image_features):
                mismatches.append("camera order")
            if mismatches:
                raise ValueError(f"Checkpoint configuration mismatch: {', '.join(mismatches)}")
        return super().from_pretrained(pretrained_name_or_path, **kwargs)
