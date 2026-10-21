"""Outer-only dynamic scaling for explicit, episode-boundary gradient SUM.

Inner autograd.grad updates are deliberately never scaled. Unlike a wrapped
optimizer scaler, this controller reaches rank consensus before any optimizer
can step, including when overflow first appears during the SUM itself.
"""

import math

import torch
import torch.distributed as dist

from .runtime import distributed, synchronize_gradients

DTYPES = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}


def validate_device(dtype, device):
    if device.type == "cuda" and dtype == "bfloat16":
        major, _ = torch.cuda.get_device_capability(device)
        if major < 8:
            raise ValueError(
                "BF16 requires native GPU support; use --precision fp16 or fp32 on V100"
            )


class OuterPrecision:
    def __init__(self, dtype, *, initial_scale=65536.0, growth_interval=2000):
        if dtype not in DTYPES.values():
            raise ValueError(f"Unsupported precision: {dtype}")
        if not math.isfinite(initial_scale) or initial_scale <= 0 or growth_interval < 1:
            raise ValueError("Invalid loss scale settings")
        self.dtype = dtype
        self.enabled = dtype == "float16"
        self.scale = float(initial_scale) if self.enabled else 1.0
        self.growth_interval = growth_interval
        self.good_updates = 0
        self.skipped_updates = 0

    def backward(self, loss):
        if self.enabled and not torch.isfinite(loss.detach()).all():
            raise FloatingPointError(
                "Nonfinite forward loss; loss scaling cannot fix this. Try fp32."
            )
        (loss * self.scale if self.enabled else loss).backward()

    @torch.no_grad()
    def prepare_gradients(self, policy):
        if not self.enabled:
            synchronize_gradients(policy)
            return True
        parameters = [p for p in policy.parameters() if p.requires_grad]
        for p in parameters:
            if p.grad is not None:
                if p.grad.is_sparse:
                    # Let the existing reducer propagate an identical error on every rank.
                    continue
                p.grad.div_(self.scale)
        try:
            synchronize_gradients(policy)
        except FloatingPointError as exc:
            # Sparse gradients are not numerical overflow and must not be retried.
            sparse = torch.tensor(
                any(p.grad is not None and p.grad.is_sparse for p in parameters),
                device=parameters[0].device,
                dtype=torch.int64,
            )
            if distributed():
                dist.all_reduce(sparse, op=dist.ReduceOp.MAX)
            if sparse.item():
                raise exc
            return self._overflow()
        invalid = torch.zeros((), device=parameters[0].device, dtype=torch.int64)
        for p in parameters:
            if p.grad is not None:
                invalid.add_((~p.grad.isfinite().all()).long())
        if distributed():
            dist.all_reduce(invalid, op=dist.ReduceOp.MAX)
        return self._overflow() if invalid.item() else True

    def _overflow(self):
        self.scale *= 0.5
        self.good_updates = 0
        self.skipped_updates += 1
        if self.scale == 0:
            raise FloatingPointError("Loss scale underflow; try fp32")
        return False

    def success(self):
        if self.enabled:
            self.good_updates += 1
            if self.good_updates == self.growth_interval:
                self.scale = min(self.scale * 2, 2.0**24)
                self.good_updates = 0

    def state_dict(self):
        return {
            name: getattr(self, name)
            for name in ("dtype", "scale", "growth_interval", "good_updates", "skipped_updates")
        }

    def load_state_dict(self, state):
        if state is None:
            if self.enabled:
                raise ValueError("FP16 checkpoint is missing scaler state")
            return
        if state.get("dtype") != self.dtype:
            raise ValueError("Checkpoint precision mismatch")
        scale = state["scale"]
        interval, good, skipped = (
            state[k] for k in ("growth_interval", "good_updates", "skipped_updates")
        )
        if (
            not math.isfinite(scale)
            or not 0 < scale <= 2.0**24
            or any(type(x) is not int for x in (interval, good, skipped))
            or interval < 1
            or not 0 <= good < interval
            or skipped < 0
            or (not self.enabled and (scale != 1 or good != 0 or skipped != 0))
        ):
            raise ValueError("Invalid scaler state")
        self.scale, self.growth_interval = scale, interval
        self.good_updates, self.skipped_updates = good, skipped
