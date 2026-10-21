"""Run on the target GPU server: small FP16 backward and NCCL SUM smoke test.

Requires explicit --cuda; never runs automatically during installation or tests.
"""

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true", required=True)
    args = parser.parse_args()
    if not args.cuda:
        parser.error("--cuda is required")
    import torch
    import torch.distributed as dist

    from lerobot_policy_vla_ttt.training.episode_parallel.precision import OuterPrecision

    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    if "sm_70" not in torch.cuda.get_arch_list():
        raise RuntimeError(
            "This PyTorch wheel does not include sm_70; install the cu126 environment"
        )
    if torch.cuda.get_device_capability(device) != (7, 0):
        raise RuntimeError("Expected a Volta device for this V100 acceptance check")
    if world > 1:
        dist.init_process_group("nccl", device_id=device)
    try:
        torch.manual_seed(12)
        model = torch.nn.Linear(32, 32).to(device).float()
        scale = OuterPrecision("float16", initial_scale=128)
        with torch.autocast("cuda", dtype=torch.float16):
            output = model(torch.ones(8, 32, device=device))
        assert output.dtype == torch.float16
        scale.backward(output.float().square().mean() / world)
        assert scale.prepare_gradients(model)
        assert all(p.grad.dtype == torch.float32 for p in model.parameters())
        torch.optim.AdamW(model.parameters()).step()
        if world > 1:
            checksum = sum(p.detach().double().sum() for p in model.parameters())
            values = [torch.zeros_like(checksum) for _ in range(world)]
            dist.all_gather(values, checksum)
            assert all(torch.equal(values[0], value) for value in values)
        print(
            json.dumps(
                {
                    "rank": rank,
                    "world_size": world,
                    "gpu": torch.cuda.get_device_name(device),
                    "torch": torch.__version__,
                    "cuda_runtime": torch.version.cuda,
                    "architectures": torch.cuda.get_arch_list(),
                    "fp16_backward": "passed",
                    "gradient_sum": "passed",
                }
            ),
            flush=True,
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
