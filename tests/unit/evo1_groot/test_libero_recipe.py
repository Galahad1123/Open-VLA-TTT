"""The LIBERO preset must not silently reintroduce smoke-training defaults."""

from pathlib import Path

from lerobot.configs.policies import PreTrainedConfig


def test_libero_preset_produces_comparable_optimizer_and_schedule():
    root = Path(__file__).resolve().parents[3]
    config = PreTrainedConfig.from_pretrained(
        root / "configs/baselines/evo1_groot/libero", local_files_only=True
    )
    optimizer = config.get_optimizer_preset()
    assert optimizer.lr == 1e-5
    assert optimizer.weight_decay == 1e-3
    assert optimizer.grad_clip_norm == 1.0
    assert config.get_scheduler_preset().num_warmup_steps == 1000
    assert config.dropout == 0.2
    assert not config.use_flash_attn
    assert config.n_action_steps == 10
    assert config.num_inference_timesteps == 32
