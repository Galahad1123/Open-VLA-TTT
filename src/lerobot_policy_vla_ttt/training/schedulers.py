"""Versioned local schedules; legacy recipes retain LeRobot's original schedule."""

import math
from dataclasses import dataclass

from lerobot.optim.schedulers import LRSchedulerConfig
from torch.optim.lr_scheduler import LambdaLR


@LRSchedulerConfig.register_subclass("vla_ttt_cosine_floor")
@dataclass
class CosineFloorSchedulerConfig(LRSchedulerConfig):
    num_warmup_steps: int = 300
    min_lr_ratio: float = 0.1

    def __post_init__(self):
        if self.num_warmup_steps < 0 or not 0 <= self.min_lr_ratio <= 1:
            raise ValueError("Invalid warmup or minimum LR ratio")

    def build(self, optimizer, num_training_steps):
        if num_training_steps <= self.num_warmup_steps:
            raise ValueError("Training must extend beyond warmup")

        def scale(step):
            if step < self.num_warmup_steps:
                return step / max(1, self.num_warmup_steps)
            progress = min(
                1.0, (step - self.num_warmup_steps) / (num_training_steps - self.num_warmup_steps)
            )
            return self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )

        return LambdaLR(optimizer, scale)
