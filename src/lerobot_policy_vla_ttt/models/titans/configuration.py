"""Common VLA adaptation settings for the three independently registered policies."""

import math
from dataclasses import dataclass

from ...ttt.configuration import MemoryUpdateConfig
from ..evo1_groot.configuration_evo1_groot import Evo1GrootConfig


@dataclass
class TitansConfig(MemoryUpdateConfig, Evo1GrootConfig):
    use_flash_attn: bool = False
    attention_scope: str = "observation_causal"
    memory_dim: int = 128
    memory_hidden_dim: int = 256
    memory_learning_rate: float = 0.1
    memory_enabled: bool = True
    memory_training: str = "memory_only"
    memory_layers: tuple[int, ...] = (10, 12)  # zero-based retained Qwen2 layers
    memory_input: str = "all"
    memory_granularity: str = "observation"
    num_persistent_tokens: int = 4
    memory_update_protocol: str = "observation"
    rope_position_mode: str = "decision"

    def __post_init__(self):
        super().__post_init__()
        self.validate_memory_update()
        if self.attention_scope != "observation_causal":
            raise ValueError("Only observation_causal attention is implemented")
        # Normalize the inherited runtime preference: this backend requires double backward.
        self.use_flash_attn = False
        if not isinstance(self.vlm_num_layers, int) or self.vlm_num_layers < 1:
            raise ValueError("Titans requires an explicit positive vlm_num_layers")
        self.memory_layers = tuple(self.memory_layers)
        if not self.memory_layers or tuple(sorted(set(self.memory_layers))) != self.memory_layers:
            raise ValueError("memory_layers must be nonempty, unique and sorted")
        if any(
            not isinstance(i, int) or i < 0 or i >= self.vlm_num_layers for i in self.memory_layers
        ):
            raise ValueError("memory_layers must index retained VLM language layers")
        if self.memory_input not in {"all", "visual"}:
            raise ValueError("memory_input must be all or visual")
        if self.memory_granularity not in {"observation", "token"}:
            raise ValueError("memory_granularity must be observation or token")
        if self.memory_dim <= 0 or self.memory_dim % 2 or self.memory_hidden_dim <= 0:
            raise ValueError("Invalid memory dimensions")
        if not math.isfinite(self.memory_learning_rate) or self.memory_learning_rate <= 0:
            raise ValueError("Invalid memory learning rate")
        if not isinstance(self.num_persistent_tokens, int) or self.num_persistent_tokens < 0:
            raise ValueError("num_persistent_tokens must be nonnegative")
        if self.memory_training not in {"memory_only", "policy_only", "joint"}:
            raise ValueError("memory_training must be memory_only or joint")
        if (
            self.memory_update_protocol not in {"observation", "half_execution"}
            or self.rope_position_mode != "decision"
        ):
            raise ValueError("VLM memory uses observation calls and decision RoPE positions")
        if self.memory_update_protocol == "half_execution":
            if self.n_action_steps % 2 or self.memory_granularity != "observation":
                raise ValueError(
                    "half_execution needs even execution length and observation granularity"
                )
        if self.gradient_checkpointing_use_reentrant:
            raise ValueError("TTT requires non-reentrant gradient checkpointing")

    @property
    def write_stride(self):
        return self.n_action_steps // 2
