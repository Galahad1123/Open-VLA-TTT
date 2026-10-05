"""Independent RoboTTT architecture identity and explicit research protocol."""

import math
from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig

from ..evo1_groot.configuration_evo1_groot import Evo1GrootConfig


@PreTrainedConfig.register_subclass("robottt")
@dataclass
class RoboTTTConfig(Evo1GrootConfig):
    num_register_tokens: int = 16
    memory_dim: int = 128
    memory_hidden_dim: int = 256
    memory_learning_rate: float = 0.1
    memory_gate_init: float = 0.001
    memory_enabled: bool = True
    memory_training: str = "memory_only"
    memory_update_protocol: str = "cumulative_denoising"

    def __post_init__(self):
        super().__post_init__()
        if self.memory_dim <= 0 or self.memory_dim % 2 or self.memory_hidden_dim <= 0:
            raise ValueError("memory_dim must be positive and even; memory_hidden_dim positive")
        if not math.isfinite(self.memory_learning_rate) or self.memory_learning_rate <= 0:
            raise ValueError("memory_learning_rate must be positive and finite")
        if not math.isfinite(self.memory_gate_init):
            raise ValueError("memory_gate_init must be finite")
        if self.memory_training not in {"memory_only", "joint"}:
            raise ValueError("memory_training must be memory_only or joint")
        if self.memory_update_protocol != "cumulative_denoising":
            raise ValueError("Only cumulative_denoising is implemented")
