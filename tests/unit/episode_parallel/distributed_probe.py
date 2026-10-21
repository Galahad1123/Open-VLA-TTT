"""Small CPU/Gloo or explicit CUDA/NCCL probe; invoke with python or torchrun."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from accelerate import Accelerator

from episode_parallel.test_baseline import make_baseline
from episode_parallel.test_checkpoint import assert_control_equal, mixed_source, source
from episode_parallel.test_engine import episode_events, make_policy
from lerobot_policy_vla_ttt.training.episode_parallel.config import Recipe
from lerobot_policy_vla_ttt.training.episode_parallel.engine import backward_episode, episode_rng
from lerobot_policy_vla_ttt.training.episode_parallel.optim import PhaseOptimizers
from lerobot_policy_vla_ttt.training.episode_parallel.run_files import run_directory
from lerobot_policy_vla_ttt.training.episode_parallel.runner import run
from lerobot_policy_vla_ttt.training.episode_parallel.runtime import (
    broadcast_model,
    collective_error,
    synchronize_gradients,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    accelerator = Accelerator(cpu=args.device == "cpu", mixed_precision="no")
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    with run_directory(args.output / "logged", accelerator):
        print(f"rank {accelerator.process_index} ready", flush=True)
    result = {}
    for kind in ("evo1_groot", "t2mem", "aqmem", "mac", "mag", "mal"):
        with pytest.MonkeyPatch.context() as patch:
            torch.manual_seed(12)
            policy = (
                make_baseline(patch) if kind == "evo1_groot" else make_policy(kind, patch)
            ).to(accelerator.device)
            if args.device == "cuda":
                policy.config.dtype = "bfloat16"
            broadcast_model(policy)
            r = Recipe(
                steps=4,
                phase_length=1,
                warmup_steps=1,
                global_batch_size=4,
                horizon=3,
                execution=2,
                tbptt_events=3,
                training_mode="policy" if kind == "evo1_groot" else "alternating",
            )
            control = PhaseOptimizers(policy, r)
            history = []
            for step in range(r.steps):
                control.begin(step)
                loss = torch.zeros((), device=accelerator.device)
                for slot in range(accelerator.process_index, 4, accelerator.num_processes):
                    with episode_rng(100 + step * 4 + slot, accelerator.device):
                        events = episode_events(3 + slot * 2)
                        metrics, _ = backward_episode(
                            policy,
                            events,
                            decisions=2 + slot,
                            recipe=r,
                            backward=accelerator.backward,
                        )
                        loss += metrics["loss_sum"] / 4
                if dist.is_initialized():
                    dist.all_reduce(loss)
                synchronize_gradients(policy, bucket_bytes=1024)
                grads = {
                    n: p.grad.cpu().clone()
                    for n, p in policy.named_parameters()
                    if p.grad is not None
                }
                control.step()
                history.append({"loss": loss.cpu(), "grads": grads})
            result[kind] = {
                "state": {k: v.cpu() for k, v in policy.state_dict().items()},
                "history": history,
                "counts": control.counts,
            }
    # Exercise actual main-rank artifacts and per-rank resume, not just the reducer.
    with pytest.MonkeyPatch.context() as patch:
        torch.manual_seed(12)
        whole = make_policy("t2mem", patch).to(accelerator.device)
        expected = run(
            whole,
            source(),
            None,
            None,
            recipe=r,
            accelerator=accelerator,
            output=args.output / "whole",
            identity={"dataset": "probe"},
        )
        torch.manual_seed(12)
        split = make_policy("t2mem", patch).to(accelerator.device)
        run(
            split,
            source(),
            None,
            None,
            recipe=r,
            accelerator=accelerator,
            output=args.output / "split",
            identity={"dataset": "probe"},
            stop_after=2,
        )
        restored = make_policy("t2mem", patch).to(accelerator.device)
        actual = run(
            restored,
            source(),
            None,
            None,
            recipe=r,
            accelerator=accelerator,
            output=args.output / "resumed",
            identity={"dataset": "probe"},
            resume=args.output / "split/step-00000002",
        )
        torch.testing.assert_close(whole.state_dict(), restored.state_dict(), atol=0, rtol=0)
        assert actual == expected
        a = torch.load(
            args.output / "whole/step-00000004/training.pt", weights_only=False, map_location="cpu"
        )
        b = torch.load(
            args.output / "resumed/step-00000004/training.pt",
            weights_only=False,
            map_location="cpu",
        )
        assert_control_equal(a["control"], b["control"])
    # Joint AQMem through the production runner, with genuinely different tasks
    # and episode lengths. Capture synchronized gradients at optimizer boundaries.
    with pytest.MonkeyPatch.context() as patch:
        recipe = replace(r, training_mode="joint", sampling_mode="mixed_tasks")
        history = []
        original_step = PhaseOptimizers.step

        def capture_step(control):
            history.append(
                {
                    n: p.grad.detach().cpu().clone()
                    for n, p in control.policy.named_parameters()
                    if p.grad is not None
                }
            )
            return original_step(control)

        patch.setattr(PhaseOptimizers, "step", capture_step)
        torch.manual_seed(12)
        whole = make_policy("aqmem", patch).to(accelerator.device)
        expected = run(
            whole,
            mixed_source(),
            None,
            None,
            recipe=recipe,
            accelerator=accelerator,
            output=args.output / "joint-whole",
            identity={"dataset": "mixed-probe"},
        )
        whole_history = history[:]
        history.clear()
        torch.manual_seed(12)
        split = make_policy("aqmem", patch).to(accelerator.device)
        run(
            split,
            mixed_source(),
            None,
            None,
            recipe=recipe,
            accelerator=accelerator,
            output=args.output / "joint-split",
            identity={"dataset": "mixed-probe"},
            stop_after=2,
        )
        torch.manual_seed(999)
        restored = make_policy("aqmem", patch).to(accelerator.device)
        actual = run(
            restored,
            mixed_source(),
            None,
            None,
            recipe=recipe,
            accelerator=accelerator,
            output=args.output / "joint-resumed",
            identity={"dataset": "mixed-probe"},
            resume=args.output / "joint-split/step-00000002",
        )
        assert actual == expected
        torch.testing.assert_close(whole.state_dict(), restored.state_dict(), atol=0, rtol=0)
        torch.testing.assert_close(whole_history, history, atol=0, rtol=0)
        saved = torch.load(
            args.output / "joint-whole/step-00000004/training.pt",
            weights_only=False,
            map_location="cpu",
        )
        resumed = torch.load(
            args.output / "joint-resumed/step-00000004/training.pt",
            weights_only=False,
            map_location="cpu",
        )
        assert_control_equal(saved["control"], resumed["control"])
        rows = [
            json.loads(line)
            for line in (args.output / "joint-whole/metrics.jsonl").read_text().splitlines()
        ]
        assert all(row["global_task_counts"] == {"lift": 2, "move": 2} for row in rows)
        result["aqmem_joint_mixed"] = {
            "state": {k: v.cpu() for k, v in whole.state_dict().items()},
            "grads": whole_history,
            "losses": [row["loss"] for row in rows],
            "control": saved["control"],
            "totals": expected["totals"],
        }
    # Locally absent gradient, globally absent parameter, and phase-safe ordering.
    dummy = torch.nn.ParameterList([torch.nn.Parameter(torch.ones(2)) for _ in range(3)])
    dummy.to(accelerator.device)
    dummy[0].grad = torch.ones_like(dummy[0]) / accelerator.num_processes
    if accelerator.process_index == 0:
        dummy[1].grad = torch.ones_like(dummy[1]) * 3
    synchronize_gradients(dummy, bucket_bytes=4)
    assert torch.equal(dummy[0].grad, torch.ones_like(dummy[0]))
    assert torch.equal(dummy[1].grad, torch.ones_like(dummy[1]) * 3)
    assert dummy[2].grad is None
    dummy[0].grad.fill_(float("nan"))
    try:
        synchronize_gradients(dummy)
    except FloatingPointError:
        pass
    else:
        raise AssertionError("Nonfinite gradients were accepted")
    try:
        collective_error(
            "injected failure" if accelerator.process_index == 0 else None, accelerator.device
        )
    except RuntimeError as exc:
        assert "injected failure" in str(exc)
    else:
        raise AssertionError("Rank-local failure was not propagated")
    torch.save(result, args.output / f"rank-{accelerator.process_index}.pt")
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
