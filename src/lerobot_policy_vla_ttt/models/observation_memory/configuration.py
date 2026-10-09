"""Observation-memory settings; original backbone dimensions are inherited."""

import math
from dataclasses import dataclass

from ...ttt.configuration import MemoryUpdateConfig
from ..evo1_groot.configuration_evo1_groot import Evo1GrootConfig


@dataclass
class ObservationMemoryConfig(MemoryUpdateConfig, Evo1GrootConfig):
    num_register_tokens: int = 0
    num_interface_tokens: int = 16
    memory_heads: int = 16
    memory_head_dim: int = 64
    memory_hidden_dim: int = 256
    memory_learning_rate: float = 0.1
    memory_gate_init: float = 0.01
    memory_enabled: bool = True
    memory_training: str = "memory_only"
    memory_update_protocol: str = "half_execution"
    vlm_layer_indices: tuple[int, ...] | None = None
    use_flash_attn: bool = False

    @property
    def interface_depth(self):
        return self.num_layers

    @property
    def write_stride(self):
        return self.n_action_steps // 2

    def __post_init__(self):
        super().__post_init__()
        if self.memory_update_rule not in {"gradient_descent", "momentum_decay", "t2mem_adaptive"}:
            raise ValueError("Unknown memory_update_rule")
        if self.n_action_steps % 2 or self.n_action_steps < 2:
            raise ValueError("half_execution requires a positive even n_action_steps")
        if self.num_register_tokens != 0:
            raise ValueError("Observation memory uses interface tokens, no action registers")
        if self.memory_update_protocol != "half_execution":
            raise ValueError("Expected half_execution update protocol")
        if self.return_cls_only:
            raise ValueError("Multi-layer interfaces require full VL tokens")
        if self.memory_training not in {"memory_only", "policy_only", "joint"}:
            raise ValueError("Invalid memory_training phase")
        if self.gradient_checkpointing_use_reentrant:
            raise ValueError("Use non-reentrant checkpointing for memory training")
        for name in (
            "num_interface_tokens",
            "memory_heads",
            "memory_head_dim",
            "memory_hidden_dim",
            "vlm_num_layers",
        ):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.memory_head_dim % 2:
            raise ValueError("memory_head_dim must be even")
        if (
            not math.isfinite(self.memory_gate_init)
            or not math.isfinite(self.memory_learning_rate)
            or self.memory_learning_rate <= 0
        ):
            raise ValueError("Invalid memory gate or learning rate")
        for value in (self.memory_momentum_init, self.memory_decay_init):
            if not 0 < value < 1:
                raise ValueError("Invalid momentum/decay initialization")
        self.use_flash_attn = False
        depth = self.interface_depth
        if depth < 1:
            raise ValueError("Architecture requires at least one effective interface stage")
        if self.vlm_layer_indices is None:
            self.vlm_layer_indices = tuple(
                (
                    self.vlm_num_layers - 1
                    if depth == 1
                    else round(i * (self.vlm_num_layers - 1) / (depth - 1))
                )
                for i in range(depth)
            )
        else:
            self.vlm_layer_indices = tuple(self.vlm_layer_indices)
        if len(self.vlm_layer_indices) != depth or any(
            not isinstance(i, int) or not 0 <= i < self.vlm_num_layers
            for i in self.vlm_layer_indices
        ):
            raise ValueError(
                "vlm_layer_indices must provide a valid source for each interface layer"
            )
