from copy import deepcopy
from dataclasses import fields

import pytest
import torch
from observation_memory.assets import patch_assets
from observation_memory.test_policy import policy_config
from torch.nn.attention import SDPBackend, sdpa_kernel

from lerobot_policy_vla_ttt.models.evo1_groot.configuration_evo1_groot import Evo1GrootConfig
from lerobot_policy_vla_ttt.models.evo1_groot.modeling_evo1_groot import Evo1GrootPolicy
from lerobot_policy_vla_ttt.models.t2mem.configuration_t2mem import T2MemConfig
from lerobot_policy_vla_ttt.training.episode_parallel.config import Recipe
from lerobot_policy_vla_ttt.training.episode_parallel.engine import backward_episode, episode_rng
from lerobot_policy_vla_ttt.training.train_episode_parallel import resolve_config

from .test_engine import episode_events


def make_baseline(monkeypatch):
    patch_assets(monkeypatch)
    tiny = policy_config(T2MemConfig)
    base = Evo1GrootConfig(**{f.name: getattr(tiny, f.name) for f in fields(Evo1GrootConfig)})
    base.dropout = 0.0
    config = resolve_config(
        base, "evo1_groot", Recipe(training_mode="policy", horizon=3, execution=2), device="cpu"
    )
    return Evo1GrootPolicy(config).float()


def test_baseline_sequence_matches_direct_chunk_gradients(monkeypatch):
    p = make_baseline(monkeypatch)
    q = deepcopy(p)
    events = episode_events()
    events[-1].frame["action_mask"] = torch.tensor([[[True] * 4, [False] * 4, [False] * 4]])
    r = Recipe(training_mode="policy", global_batch_size=2, horizon=3, execution=2)
    with episode_rng(25, torch.device("cpu")):
        metrics, memory = backward_episode(
            p, events, decisions=3, recipe=r, backward=lambda x: x.backward()
        )
    with episode_rng(25, torch.device("cpu")), sdpa_kernel(SDPBackend.MATH):
        losses = [q(e.frame)[0] for e in events if e.supervised]
        (sum(losses) / 6).backward()
    assert memory is None
    assert metrics["events"] == 5 and metrics["decisions"] == 3
    assert metrics["supervised_scalars"] == 28
    assert metrics["loss_sum"] == pytest.approx(float(sum(losses).detach()) / 3)
    for (n, a), (_, b) in zip(p.named_parameters(), q.named_parameters(), strict=True):
        assert (a.grad is None) == (b.grad is None), n
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad)


def test_baseline_optimizer_and_checkpoint_resume(monkeypatch, tmp_path):
    from accelerate import Accelerator

    from lerobot_policy_vla_ttt.training.episode_parallel.runner import run

    from .test_checkpoint import assert_control_equal, source

    r = Recipe(
        training_mode="policy",
        steps=2,
        warmup_steps=1,
        horizon=3,
        execution=2,
        global_batch_size=2,
        sampling_mode="mixed_tasks",
    )
    torch.manual_seed(12)
    whole = make_baseline(monkeypatch)
    initial = deepcopy(whole)
    acc = Accelerator(cpu=True, mixed_precision="no")
    common = dict(recipe=r, accelerator=acc, identity={"dataset": "fixture"})
    run(whole, source(), None, None, output=tmp_path / "whole", **common)
    run(initial, source(), None, None, output=tmp_path / "split", stop_after=1, **common)
    resumed = make_baseline(monkeypatch)
    run(
        resumed,
        source(),
        None,
        None,
        output=tmp_path / "resumed",
        resume=tmp_path / "split" / "step-00000001",
        **common,
    )
    torch.testing.assert_close(whole.state_dict(), resumed.state_dict(), rtol=0, atol=0)
    a = torch.load(tmp_path / "whole" / "step-00000002" / "training.pt", weights_only=False)
    b = torch.load(tmp_path / "resumed" / "step-00000002" / "training.pt", weights_only=False)
    assert_control_equal(a["control"], b["control"])
    assert a["control"]["counts"] == {"policy": 2}
    import json

    assert (
        json.loads(
            (tmp_path / "whole" / "step-00000002" / "pretrained_model" / "config.json").read_text()
        )["type"]
        == "evo1_groot"
    )


def test_baseline_rejects_memory_training(monkeypatch):
    p = make_baseline(monkeypatch)
    with pytest.raises(ValueError, match="policy"):
        resolve_config(p.config, "evo1_groot", Recipe(horizon=3, execution=2), device="cpu")
    with pytest.raises(ValueError, match="memory"):
        resolve_config(
            p.config,
            "evo1_groot",
            Recipe(training_mode="policy", horizon=3, execution=2),
            device="cpu",
            overrides={"vlm_layer_indices": [0]},
        )


def test_baseline_cli_dry_run(tmp_path):
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    checkpoint = tmp_path / "checkpoint"
    Evo1GrootConfig(device="cpu").save_pretrained(checkpoint)
    (checkpoint / "train_config.json").write_text(
        json.dumps({"dataset": {"image_transforms": {"enable": False}}})
    )
    dataset = tmp_path / "dataset"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta/info.json").write_text(
        json.dumps({"features": {"action": {"shape": [7]}}, "fps": 10})
    )
    output = tmp_path / "output"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "lerobot_policy_vla_ttt.training.train_episode_parallel",
            "--architecture",
            "evo1_groot",
            "--benchmark",
            "libero",
            "--checkpoint",
            str(checkpoint),
            "--dataset-root",
            str(dataset),
            "--output",
            str(output),
            "--recipe",
            "configs/training/libero-evo1-groot-episode.json",
            "--dry-run",
        ],
        cwd=root,
        env=dict(os.environ, PYTHONPATH=str(root / "src"), CUDA_VISIBLE_DEVICES=""),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    resolved = json.loads(result.stdout)
    assert resolved["recipe"]["training_mode"] == "policy"
    assert resolved["recipe"]["sampling_mode"] == "mixed_tasks"
    assert "memory_training" not in resolved["policy"]
    assert not output.exists()
