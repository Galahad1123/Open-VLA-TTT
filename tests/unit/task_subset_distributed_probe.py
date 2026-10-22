"""CPU-only subprocess probe using the installed LeRobot update implementation."""

import argparse
from pathlib import Path

import torch
import torch.distributed as dist
from accelerate import Accelerator
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.scripts.lerobot_train import update_policy
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from torch.utils.data import DataLoader, Dataset

from lerobot_policy_vla_ttt.training.train_task_subset import per_rank_batch, successful_update


class TinyDataset(Dataset):
    def __len__(self):
        return 48

    def __getitem__(self, index):
        x = torch.tensor([index / 48, (index % 7) / 7], dtype=torch.float64)
        return {"id": index, "x": x, "y": 0.7 * x[0] - 0.4 * x[1] + 0.2}


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1, dtype=torch.float64)

    def forward(self, batch):
        loss = (self.linear(batch["x"]).squeeze(-1) - batch["y"]).square().mean()
        return loss, {"mse": loss.detach().item()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    assert not torch.cuda.is_available(), "Probe must run with CUDA_VISIBLE_DEVICES=''"
    accelerator = Accelerator(cpu=True, mixed_precision="no", step_scheduler_with_optimizer=False)
    assert accelerator.device.type == "cpu"
    if accelerator.num_processes > 1:
        assert dist.get_backend() == "gloo"
    torch.manual_seed(1000)
    local_batch = per_rank_batch(12, accelerator.num_processes)
    policy = TinyPolicy()
    initial = {key: value.clone() for key, value in policy.state_dict().items()}
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.2, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)
    sampler = EpisodeAwareSampler([0, 24], [24, 48], shuffle=True, seed=1000)
    loader = DataLoader(TinyDataset(), batch_size=local_batch, sampler=sampler, num_workers=0)
    policy, optimizer, loader, scheduler = accelerator.prepare(policy, optimizer, loader, scheduler)
    tracker = MetricsTracker(
        local_batch,
        48,
        2,
        {name: AverageMeter(name) for name in ("loss", "grad_norm", "lr", "update_s")},
        accelerator=accelerator,
    )
    ids, learning_rates, scheduler_steps, skipped_updates = [], [], [], []
    # Two complete epochs exercise the sampler's deterministic epoch advancement.
    for _ in range(2):
        for batch in loader:
            ids.append(batch["id"].tolist())
            tracker, output, skipped = successful_update(
                update_policy,
                tracker,
                policy,
                batch,
                optimizer,
                grad_clip_norm=0.25,
                accelerator=accelerator,
                lr_scheduler=scheduler,
            )
            assert "mse" in output
            tracker.step()
            learning_rates.append(optimizer.param_groups[0]["lr"])
            scheduler_steps.append(scheduler.scheduler.last_epoch)
            skipped_updates.append(skipped)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "weights": accelerator.unwrap_model(policy).state_dict(),
            "initial": initial,
            "ids": ids,
            "learning_rates": learning_rates,
            "scheduler_steps": scheduler_steps,
            "skipped_updates": skipped_updates,
            "updates": tracker.steps,
            "samples": tracker.samples,
            "world_size": accelerator.num_processes,
            "local_batch": local_batch,
        },
        args.output / f"rank-{accelerator.process_index}.pt",
    )
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
