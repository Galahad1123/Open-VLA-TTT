"""Opt-in real InternVL smoke, without dataset/model downloads or simulator imports."""

import dataclasses
import os
from pathlib import Path

import pytest
import torch
from lerobot.configs.policies import PreTrainedConfig

from lerobot_policy_vla_ttt import Evo1GrootPolicy


@pytest.mark.integration
@pytest.mark.parametrize("stage", ["stage1", "stage2"])
def test_real_vlm_training_and_sampling(stage):
    vlm_path = os.getenv("VLA_TTT_GROOT_VLM_PATH")
    if not vlm_path:
        pytest.skip("Set VLA_TTT_GROOT_VLM_PATH to a pinned local InternVL3 snapshot")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA with bf16 required")
    assert Path(vlm_path).is_dir(), "VLM must already exist locally"
    root = Path(__file__).resolve().parents[2]
    config = PreTrainedConfig.from_pretrained(root / "configs/baselines/evo1_groot/libero")
    config = dataclasses.replace(
        config, device="cuda", vlm_model_name=vlm_path, training_stage=stage
    )
    torch.manual_seed(1000)
    policy = Evo1GrootPolicy(config, vlm_hub_kwargs={"local_files_only": True}).train()
    batch = {
        "observation.images.image": torch.rand(1, 3, 256, 256, device="cuda"),
        "observation.images.image2": torch.rand(1, 3, 256, 256, device="cuda"),
        "observation.state": torch.rand(1, 8, device="cuda"),
        "action": torch.rand(1, config.chunk_size, 7, device="cuda"),
        "task": ["pick up the red object"],
    }
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=1e-5)
    loss, _ = policy(batch)
    loss.backward()
    assert torch.isfinite(loss)
    grads = [p.grad for p in policy.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(grad).all() for grad in grads)
    vlm_grads = [p.grad for p in policy.model.embedder.parameters() if p.grad is not None]
    assert bool(vlm_grads) == (stage == "stage2")
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if stage == "stage2":
        # Characterize LeRobot 0.6.1's precision selection, then exercise its real
        # update boundary: a fp16 GradScaler cannot unscale the VLM's bf16 grads.
        from accelerate import Accelerator
        from lerobot.scripts.lerobot_train import update_policy
        from lerobot.utils.logging_utils import AverageMeter, MetricsTracker

        precision = {"bfloat16": "bf16", "float32": "no"}.get(getattr(config, "dtype", None))
        if not hasattr(config, "dtype") and config.use_amp:
            precision = "fp16"
        accelerator = Accelerator(mixed_precision=precision)
        policy, optimizer = accelerator.prepare(policy, optimizer)
        metrics = MetricsTracker(
            1,
            1,
            1,
            {
                name: AverageMeter(name)
                for name in ("loss", "grad_norm", "lr", "update_s", "gpu_mem_gb")
            },
        )
        update_policy(metrics, policy, batch, optimizer, 1.0, accelerator)
        assert torch.isfinite(torch.tensor(metrics.loss.avg))
    torch.manual_seed(5)
    first = policy.predict_action_chunk(batch)
    torch.manual_seed(5)
    second = policy.predict_action_chunk(batch)
    assert first.shape == (1, 50, 24) and torch.isfinite(first).all()
    assert torch.count_nonzero(first[:, :, 7:]) == 0
    torch.testing.assert_close(first, second)
    print(
        f"{stage}: loss={loss.item():.6f}, grad_tensors={len(grads)}, "
        f"head_params={sum(p.numel() for p in policy.model.action_head.parameters())}"
    )
