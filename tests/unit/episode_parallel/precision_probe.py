"""CPU/Gloo numerical control probe. No CUDA tensors or model downloads."""

import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from accelerate import Accelerator

from episode_parallel.test_precision import test_overflow_retries_same_batch_and_resume
from lerobot_policy_vla_ttt.training.episode_parallel.precision import OuterPrecision


def main():
    acc = Accelerator(cpu=True, mixed_precision="no")
    root = Path(sys.argv[1])
    for chunk in (False, True):
        with pytest.MonkeyPatch.context() as patch:
            test_overflow_retries_same_batch_and_resume(patch, root / str(chunk), chunk)
    model = torch.nn.Linear(1, 1, bias=False)
    scaler = OuterPrecision("float16", initial_scale=1)
    # Locally finite, SUM overflows. Every rank must reject this update.
    model.weight.grad = torch.full_like(model.weight, 2e38)
    assert not scaler.prepare_gradients(model)
    assert scaler.scale == 0.5 and scaler.skipped_updates == 1
    # A locally absent gradient is globally present. Scaling must be symmetric.
    model.weight.grad = torch.ones_like(model.weight) if acc.process_index == 0 else None
    assert scaler.prepare_gradients(model)
    assert model.weight.grad.item() == 2
    states = [None] * acc.num_processes
    dist.all_gather_object(states, scaler.state_dict())
    assert states[0] == states[1]
    acc.end_training()


if __name__ == "__main__":
    main()
