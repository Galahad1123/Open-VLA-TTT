"""Disjoint phase optimizers with schedules driven only by actual phase updates."""

import math

import torch


def parameter_groups(policy, phase, recipe):
    policy.model.set_phase(f"{phase}_only")
    grouped = {}
    for name, p in policy.named_parameters():
        if not p.requires_grad:
            continue
        if phase == "policy":
            category = "vlm" if name.startswith("model.embedder.") else "action_expert"
            lr, decay = recipe.policy_lr, recipe.policy_decay
        else:
            category, lr, decay = "memory", recipe.memory_lr, recipe.memory_decay
            if ".fast." in name:
                category, decay = "w0", 0.0
            elif ".update." in name:
                category, decay = "inner_step", 0.0
            elif (
                policy.config.type == "t2mem" and ".interfaces." in name and name.endswith(".gate")
            ):
                category, lr, decay = "t2mem_gate", recipe.gate_lr, 0.0
        if p.ndim <= 1 or name.endswith(".bias"):
            decay = 0.0
        key = category, lr, decay
        group = grouped.setdefault(
            key, dict(params=[], names=[], category=category, lr=lr, peak_lr=lr, weight_decay=decay)
        )
        group["params"].append(p)
        group["names"].append(name)
    if not grouped:
        raise ValueError(f"No trainable {phase} parameters")
    return list(grouped.values())


class PhaseOptimizers:
    def __init__(self, policy, recipe):
        self.policy, self.recipe = policy, recipe
        self.optimizers = {
            phase: torch.optim.AdamW(
                parameter_groups(policy, phase, recipe), betas=(0.9, 0.999), eps=1e-8
            )
            for phase in ("memory", "policy")
        }
        a, b = (
            {id(p) for g in o.param_groups for p in g["params"]} for o in self.optimizers.values()
        )
        if a & b:
            raise ValueError("Phase parameter groups overlap")
        self.counts = {"memory": 0, "policy": 0}
        self.active = None

    def begin(self, step):
        if step != sum(self.counts.values()) or not 0 <= step < self.recipe.steps:
            raise ValueError("Outer step differs from optimizer counters or budget")
        self.active = self.recipe.phase(step)
        self.policy.train()
        self.policy.model.set_phase(f"{self.active}_only")
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        update = self.counts[self.active] + 1
        budget = self.recipe.phase_budget(self.active)
        warmup = min(self.recipe.warmup_steps, budget)
        if update <= warmup:
            factor = update / warmup
        else:
            progress = (update - warmup) / max(1, budget - warmup)
            factor = (
                self.recipe.minimum_lr_ratio
                + (1 - self.recipe.minimum_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2
            )
        for group in self.optimizers[self.active].param_groups:
            group["lr"] = group["peak_lr"] * factor
        return self.active

    def step(self):
        if self.active is None:
            raise RuntimeError("begin must precede step")
        parameters = [p for p in self.policy.parameters() if p.requires_grad]
        norm = torch.nn.utils.clip_grad_norm_(
            parameters, self.recipe.grad_clip, error_if_nonfinite=True
        )
        self.optimizers[self.active].step()
        self.counts[self.active] += 1
        self.active = None
        return float(norm)

    def state_dict(self):
        return {
            "counts": self.counts.copy(),
            "optimizers": {k: v.state_dict() for k, v in self.optimizers.items()},
        }

    def load_state_dict(self, state):
        counts = state["counts"]
        step = sum(counts.values())
        expected = {
            phase: sum(self.recipe.phase(i) == phase for i in range(step))
            for phase in self.optimizers
        }
        if counts != expected or not 0 <= step <= self.recipe.steps:
            raise ValueError("Invalid phase counters")
        for phase, optimizer in self.optimizers.items():
            saved = state["optimizers"][phase]
            if [g["names"] for g in saved["param_groups"]] != [
                g["names"] for g in optimizer.param_groups
            ]:
                raise ValueError("Optimizer parameter groups changed")
            optimizer.load_state_dict(saved)
        self.counts = counts.copy()
        self.active = None
