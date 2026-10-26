"""Independent baseline configuration and LeRobot registration."""

import math
from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig

from ...evo1.configuration_evo1 import Evo1Config
from ...training.schedulers import CosineFloorSchedulerConfig


@PreTrainedConfig.register_subclass("evo1_groot")
@dataclass
class Evo1GrootConfig(Evo1Config):
    # LeRobot's trainer consults dtype before use_amp; explicit bf16 avoids an
    # outer fp16 GradScaler trying to unscale the VLM's bf16 gradients.
    dtype: str = "bfloat16"
    num_register_tokens: int = 0
    flow_beta_alpha: float = 1.5
    flow_beta_beta: float = 1.0
    flow_time_scale: float = 0.999
    robomme_action_representation: str = "absolute"
    robomme_augmentation: str = "none"
    scheduler_min_lr_ratio: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        if self.robomme_action_representation not in {"absolute", "delta_joint"}:
            raise ValueError("Unknown robomme_action_representation")
        if self.robomme_augmentation not in {"none", "official_v1"}:
            raise ValueError("Unknown robomme_augmentation")
        if not 0 <= self.scheduler_min_lr_ratio <= 1:
            raise ValueError("scheduler_min_lr_ratio must be in [0, 1]")
        if self.dtype not in {"bfloat16", "float32"}:
            raise ValueError("dtype must be bfloat16 or float32 (fp16 GradScaler is unsupported)")
        for name in (
            "embed_dim",
            "hidden_dim",
            "state_hidden_dim",
            "num_heads",
            "num_layers",
            "chunk_size",
            "n_action_steps",
            "num_inference_timesteps",
            "num_categories",
            "max_state_dim",
            "max_action_dim",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.embed_dim % self.num_heads or self.embed_dim % 2:
            raise ValueError("embed_dim must be even and divisible by num_heads")
        if self.num_layers % 2:
            raise ValueError("num_layers must be even: complete cross/self attention pairs")
        if not isinstance(self.num_register_tokens, int) or self.num_register_tokens < 0:
            raise ValueError("num_register_tokens must be a nonnegative integer")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if any(not math.isfinite(x) or x <= 0 for x in (self.flow_beta_alpha, self.flow_beta_beta)):
            raise ValueError("flow Beta parameters must be finite and positive")
        if not 0 < self.flow_time_scale <= 1:
            raise ValueError("flow_time_scale must be in (0, 1]")
        if self.rtc_config is not None:
            raise ValueError("evo1_groot does not support RTC")
        if self.action_head != "flowmatching":
            raise ValueError("evo1_groot requires action_head='flowmatching'")

    def get_scheduler_preset(self):
        if self.scheduler_min_lr_ratio > 0:
            return CosineFloorSchedulerConfig(
                num_warmup_steps=self.scheduler_warmup_steps,
                min_lr_ratio=self.scheduler_min_lr_ratio,
            )
        return super().get_scheduler_preset()
