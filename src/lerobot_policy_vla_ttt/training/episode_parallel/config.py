"""Auditable outer-loop settings for the LIBERO observation-memory recipe."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Recipe:
    steps: int = 2000
    phase_length: int = 500
    memory_phase_length: int | None = None
    policy_phase_length: int | None = None
    freeze_vlm: bool = False
    warmup_ratio: float | None = None
    save_phase_boundaries: bool = False
    max_run_seconds: float | None = None
    global_batch_size: int = 4
    tbptt_events: int = 8
    execution: int = 10
    horizon: int = 50
    warmup_steps: int = 50
    seed: int = 1000
    save_every: int = 250
    grad_clip: float = 1.0
    policy_lr: float = 1e-5
    memory_lr: float = 7.5e-5
    gate_lr: float = 5e-4
    policy_decay: float = 1e-3
    memory_decay: float = 1e-5
    minimum_lr_ratio: float = 0.1
    training_mode: str = "alternating"
    sampling_mode: str = "same_task"
    update_unit: str = "episode"

    def __post_init__(self):
        lengths = (self.memory_phase_length, self.policy_phase_length)
        if any(x is not None for x in lengths):
            if any(type(x) is not int or x < 1 for x in lengths):
                raise ValueError("Both phase lengths must be positive integers")
            if self.training_mode != "alternating":
                raise ValueError("Explicit phase lengths require alternating training")
        if type(self.freeze_vlm) is not bool:
            raise ValueError("freeze_vlm must be boolean")
        if self.warmup_ratio is not None and not 0 <= self.warmup_ratio < 1:
            raise ValueError("warmup_ratio must be in [0,1)")
        if self.max_run_seconds is not None and (
            not math.isfinite(self.max_run_seconds) or self.max_run_seconds <= 0
        ):
            raise ValueError("max_run_seconds must be positive and finite")
        if self.update_unit not in ("episode", "chunk"):
            raise ValueError("update_unit must be episode or chunk")
        if self.update_unit == "chunk":
            if self.training_mode != "policy":
                raise ValueError("chunk updates require policy training")
            if self.sampling_mode != "mixed_tasks":
                raise ValueError("chunk updates require mixed_tasks sampling")
        if self.training_mode not in ("alternating", "joint", "policy"):
            raise ValueError("training_mode must be alternating, joint or policy")
        if self.sampling_mode not in ("same_task", "mixed_tasks"):
            raise ValueError("sampling_mode must be same_task or mixed_tasks")
        for name in (
            "steps",
            "phase_length",
            "global_batch_size",
            "tbptt_events",
            "execution",
            "horizon",
            "save_every",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.execution < 2 or self.execution % 2:
            raise ValueError("execution must be even and >= 2")
        if type(self.warmup_steps) is not int or self.warmup_steps < 0:
            raise ValueError("warmup_steps must be nonnegative")
        for name in ("grad_clip", "policy_lr", "memory_lr", "gate_lr"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("policy_decay", "memory_decay"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 <= self.minimum_lr_ratio <= 1:
            raise ValueError("minimum_lr_ratio must be in [0, 1]")

    def phase(self, step):
        """Phase for the zero-based next outer update."""
        if self.training_mode in ("joint", "policy"):
            return self.training_mode
        memory = self.memory_phase_length or self.phase_length
        policy = self.policy_phase_length or self.phase_length
        return "memory" if step % (memory + policy) < memory else "policy"

    def phase_boundary(self, completed):
        return completed > 0 and (
            completed == self.steps or self.phase(completed - 1) != self.phase(completed)
        )

    def phase_budget(self, phase):
        if self.training_mode in ("policy", "joint"):
            return self.steps if phase == self.training_mode else 0
        return sum(self.phase(step) == phase for step in range(self.steps))
