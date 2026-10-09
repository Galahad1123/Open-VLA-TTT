"""Shared update settings, independent of the memory connection or its clock."""

import math
from dataclasses import dataclass


@dataclass
class MemoryUpdateConfig:
    memory_update_rule: str = "gradient_descent"
    memory_momentum: bool = True
    memory_forgetting: bool = True
    memory_momentum_init: float = 0.9
    memory_decay_init: float = 0.01
    memory_data_dependent: bool = True

    def validate_memory_update(self):
        if self.memory_update_rule not in {"gradient_descent", "momentum_decay"}:
            raise ValueError("Unknown memory_update_rule")
        for value in (self.memory_momentum_init, self.memory_decay_init):
            if not math.isfinite(value) or not 0 < value < 1:
                raise ValueError("Momentum and decay initial values must be in (0, 1)")

    def update_kwargs(self):
        return dict(
            update_rule=self.memory_update_rule,
            momentum=self.memory_momentum,
            forgetting=self.memory_forgetting,
            momentum_init=self.memory_momentum_init,
            decay_init=self.memory_decay_init,
            data_dependent=self.memory_data_dependent,
        )
