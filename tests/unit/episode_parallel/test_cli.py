import json
import os
import shutil
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

from observation_memory.test_policy import policy_config

from lerobot_policy_vla_ttt.models.evo1_groot.configuration_evo1_groot import Evo1GrootConfig
from lerobot_policy_vla_ttt.models.t2mem.configuration_t2mem import T2MemConfig
from lerobot_policy_vla_ttt.training.episode_parallel.config import Recipe
from lerobot_policy_vla_ttt.training.train_episode_parallel import resolve_config


def test_baseline_config_enables_full_policy_and_observation_protocol():
    tiny = policy_config(T2MemConfig)
    base = Evo1GrootConfig(**{f.name: getattr(tiny, f.name) for f in fields(Evo1GrootConfig)})
    r = Recipe(horizon=3, execution=2)
    for kind in ("t2mem", "aqmem", "mac", "mag", "mal"):
        overrides = {"memory_layers": [0, 1]} if kind in ("mac", "mag", "mal") else {}
        cfg = resolve_config(base, kind, r, device="cpu", overrides=overrides)
        assert (
            cfg.finetune_language_model and cfg.finetune_vision_model and cfg.finetune_action_head
        )
        assert cfg.memory_update_protocol == "half_execution"
        assert cfg.memory_update_rule == "t2mem_adaptive"
        assert cfg.vlm_dtype == "float32" and cfg.dtype == "bfloat16"
        assert not cfg.gradient_checkpointing_use_reentrant


def test_dry_run_does_not_load_weights_or_create_output(tmp_path):
    root = Path(__file__).resolve().parents[3]
    checkpoint = tmp_path / "checkpoint"
    cfg = Evo1GrootConfig(device="cpu")
    cfg.save_pretrained(checkpoint)
    (checkpoint / "train_config.json").write_text(
        json.dumps({"dataset": {"image_transforms": {"enable": True, "max_num_transforms": 3}}})
    )
    dataset = tmp_path / "dataset"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta/info.json").write_text(
        json.dumps({"features": {"action": {"shape": [7]}}, "fps": 10})
    )
    output = tmp_path / "run"
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        PYTHONPATH=str(root / "src"),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "lerobot_policy_vla_ttt.training.train_episode_parallel",
            "--checkpoint",
            str(checkpoint),
            "--dataset-root",
            str(dataset),
            "--output",
            str(output),
            "--architecture",
            "t2mem",
            "--dry-run",
        ],
        env=env,
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    resolved = json.loads(result.stdout)
    assert resolved["recipe"]["global_batch_size"] == 4
    assert resolved["gradient_reduction"] == "explicit_sum"
    assert resolved["augmentation"]["enable"] is True
    assert not output.exists()

    machine = tmp_path / "machine.toml"
    vlm = tmp_path / "vlm"
    vlm.mkdir()
    machine.write_text(
        '[assets]\ninitial_checkpoint="checkpoint"\n'
        'libero_dataset="dataset"\nvlm="vlm"\n'
        '[storage]\nruns_root="runs"\nhf_home="cache"\n'
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "lerobot_policy_vla_ttt.training.train_episode_parallel",
            "--machine",
            str(machine),
            "--run-name",
            "portable",
            "--dry-run",
        ],
        env=env,
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    resolved = json.loads(result.stdout)
    assert resolved["policy"]["vlm_model_name"] == str(vlm)
    assert resolved["paths"]["output"] == str(tmp_path / "runs/portable")
    assert resolved["hf_home"] == str(tmp_path / "cache")
    assert resolved["asset_validation"] == "metadata_only"
    assert not (tmp_path / "runs").exists()


def test_cli_real_checkpoint_processors_and_resume(monkeypatch, tmp_path):
    import torch
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.datasets import lerobot_dataset
    from observation_memory.assets import patch_assets

    from lerobot_policy_vla_ttt.cli import make_pre_post_processors
    from lerobot_policy_vla_ttt.models.evo1_groot.modeling_evo1_groot import Evo1GrootPolicy
    from lerobot_policy_vla_ttt.training.train_episode_parallel import main

    patch_assets(monkeypatch)
    vlm = tmp_path / "vlm"
    vlm.mkdir()
    (vlm / "config.json").write_text('{"fixture": "tiny"}')
    (vlm / "model.safetensors").write_bytes(b"tiny fixture asset")
    cfg = policy_config(
        T2MemConfig,
        vlm_model_name=str(vlm),
        max_state_dim=8,
        max_action_dim=8,
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (8,)),
            "observation.images.image": PolicyFeature(FeatureType.VISUAL, (3, 16, 16)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (7,))},
    )
    base_cfg = Evo1GrootConfig(**{f.name: getattr(cfg, f.name) for f in fields(Evo1GrootConfig)})
    baseline = Evo1GrootPolicy(base_cfg)
    checkpoint = tmp_path / "base"
    baseline.save_pretrained(checkpoint)
    stats = {
        k: {"min": -torch.ones(n), "max": torch.ones(n)}
        for k, n in [("action", 7), ("observation.state", 8)]
    }
    pre, post = make_pre_post_processors(base_cfg, dataset_stats=stats)
    pre.save_pretrained(checkpoint)
    post.save_pretrained(checkpoint)
    (checkpoint / "train_config.json").write_text(
        json.dumps({"dataset": {"image_transforms": {"enable": True, "max_num_transforms": 3}}})
    )
    dataset = tmp_path / "data"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta/info.json").write_text(
        json.dumps({"features": {"action": {"shape": [7]}}, "fps": 10})
    )
    from types import SimpleNamespace

    class TinyDataset:
        def __init__(self, *args, **kwargs):
            self.meta = SimpleNamespace(
                episodes=[
                    {
                        "episode_index": 0,
                        "tasks": ["move"],
                        "dataset_from_index": 0,
                        "dataset_to_index": 5,
                    }
                ]
            )

        def __len__(self):
            return 5

        def __getitem__(self, i):
            return {
                "observation.state": torch.full((8,), i / 10),
                "observation.images.image": torch.full((3, 16, 16), 0.5),
                "action": torch.full((3, 7), i / 10),
                "task": "move",
            }

    monkeypatch.setattr(lerobot_dataset, "LeRobotDataset", TinyDataset)
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(
        json.dumps(
            {
                "steps": 2,
                "phase_length": 1,
                "warmup_steps": 0,
                "horizon": 3,
                "execution": 2,
                "global_batch_size": 1,
                "tbptt_events": 3,
            }
        )
    )
    memory_path = tmp_path / "memory.json"
    memory_path.write_text(
        json.dumps(
            {
                "memory_heads": 2,
                "memory_head_dim": 4,
                "memory_hidden_dim": 6,
                "num_interface_tokens": 3,
            }
        )
    )
    main(
        [
            "--cpu",
            "--checkpoint",
            str(checkpoint),
            "--dataset-root",
            str(dataset),
            "--output",
            str(tmp_path / "first"),
            "--recipe",
            str(recipe_path),
            "--memory-config",
            str(memory_path),
            "--stop-after",
            "1",
        ]
    )
    # Same transform definitions in a different order change seeded sampling.
    # Resume identity must reject that change, even though dict equality would not.
    import pytest

    saved_first = torch.load(tmp_path / "first/step-00000001/training.pt", weights_only=False)
    changed_augmentation = saved_first["identity"]["augmentation"].copy()
    changed_augmentation["tfs"] = dict(reversed(list(changed_augmentation["tfs"].items())))
    augmentation_path = tmp_path / "changed-augmentation.json"
    augmentation_path.write_text(json.dumps(changed_augmentation))
    with pytest.raises(SystemExit) as rejected:
        main(
            [
                "--cpu",
                "--resume",
                str(tmp_path / "first/step-00000001"),
                "--dataset-root",
                str(dataset),
                "--output",
                str(tmp_path / "rejected"),
                "--augmentation-config",
                str(augmentation_path),
                "--dry-run",
            ]
        )
    assert rejected.value.code == 2
    main(
        [
            "--cpu",
            "--checkpoint",
            str(checkpoint),
            "--dataset-root",
            str(dataset),
            "--output",
            str(tmp_path / "whole"),
            "--recipe",
            str(recipe_path),
            "--memory-config",
            str(memory_path),
        ]
    )
    # Resource relocation must preserve exact resume, while changed content must not.
    moved_dataset = tmp_path / "relocated/data"
    shutil.move(dataset, moved_dataset)
    moved_vlm = tmp_path / "relocated/vlm"
    shutil.move(vlm, moved_vlm)
    moved_checkpoint = tmp_path / "relocated/checkpoint"
    shutil.move(tmp_path / "first/step-00000001", moved_checkpoint)
    resume_args = [
        "--cpu",
        "--resume",
        str(moved_checkpoint),
        "--dataset-root",
        str(moved_dataset),
        "--vlm-path",
        str(moved_vlm),
    ]
    (moved_vlm / "model.safetensors").write_bytes(b"wrong resource")
    with pytest.raises(SystemExit) as rejected:
        main(resume_args + ["--output", str(tmp_path / "wrong-vlm")])
    assert rejected.value.code == 2
    (moved_vlm / "model.safetensors").write_bytes(b"tiny fixture asset")
    (moved_dataset / "frames.parquet").write_bytes(b"unexpected payload")
    with pytest.raises(SystemExit) as rejected:
        main(resume_args + ["--output", str(tmp_path / "wrong-data")])
    assert rejected.value.code == 2
    (moved_dataset / "frames.parquet").unlink()
    main(
        resume_args
        + [
            "--output",
            str(tmp_path / "second"),
        ]
    )
    saved = torch.load(tmp_path / "second/step-00000002/training.pt", weights_only=False)
    assert saved["control"]["counts"] == {"memory": 1, "policy": 1}
    assert saved["totals"]["episodes"] == 2
    assert saved["identity"] == saved_first["identity"]
    from safetensors.torch import load_file

    from .test_checkpoint import assert_control_equal

    expected = torch.load(tmp_path / "whole/step-00000002/training.pt", weights_only=False)
    assert_control_equal(saved["control"], expected["control"])
    torch.testing.assert_close(
        load_file(tmp_path / "whole/step-00000002/pretrained_model/model.safetensors"),
        load_file(tmp_path / "second/step-00000002/pretrained_model/model.safetensors"),
        atol=0,
        rtol=0,
    )
    log = (tmp_path / "second/logs/rank-0.log").read_text()
    assert '"step": 2' in log
    manifest = json.loads((tmp_path / "second/run.json").read_text())
    assert manifest["launch"]["paths"]["dataset_root"] == str(moved_dataset)
    assert manifest["launch"]["environment"]["versions"]["torch"]
    assert manifest["launch"]["environment"]["git_commit"]


def test_indexed_cuda_device_resolves_to_lerobot_device_type(monkeypatch):
    # LeRobot's AMP capability check accepts 'cuda', not 'cuda:N'. Runtime
    # placement still uses the concrete Accelerate device/current CUDA rank.
    monkeypatch.setattr("lerobot.configs.policies.is_torch_device_available", lambda _: True)
    base = Evo1GrootConfig(device="cpu")
    cfg = resolve_config(base, "t2mem", Recipe(), device="cuda:3")
    assert cfg.device == "cuda"
    assert cfg.use_amp


def test_checkpoint_augmentation_roundtrip_preserves_seeded_pixels():
    from dataclasses import asdict

    import torch
    from lerobot.transforms import ImageTransforms, ImageTransformsConfig

    from lerobot_policy_vla_ttt.training.episode_parallel.checkpoint import canonical
    from lerobot_policy_vla_ttt.training.episode_parallel.data import temporal_transform
    from lerobot_policy_vla_ttt.training.train_episode_parallel import augmentation_config

    config = ImageTransformsConfig(enable=True, max_num_transforms=3)
    saved = canonical(asdict(config))
    restored = augmentation_config(saved)
    image = torch.linspace(0, 1, 3 * 16 * 16).reshape(3, 16, 16)
    actual = temporal_transform(image, ImageTransforms(restored), seed=7, camera="image")
    expected = temporal_transform(image, ImageTransforms(config), seed=7, camera="image")
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
