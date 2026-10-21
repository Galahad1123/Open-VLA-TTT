import pytest
import torch

from lerobot_policy_vla_ttt.models.evo1_groot.configuration_evo1_groot import Evo1GrootConfig
from lerobot_policy_vla_ttt.training.episode_parallel.config import Recipe
from lerobot_policy_vla_ttt.training.train_episode_parallel import resolve_config


@pytest.mark.parametrize("kind", ["evo1_groot", "mac", "mag", "mal", "aqmem", "t2mem"])
@pytest.mark.parametrize(
    "precision,dtype", [("bf16", "bfloat16"), ("fp16", "float16"), ("fp32", "float32")]
)
def test_precision_resolves_without_changing_weight_dtype(kind, precision, dtype):
    base = Evo1GrootConfig(device="cpu")
    recipe = Recipe(training_mode="policy" if kind == "evo1_groot" else "alternating")
    cfg = resolve_config(base, kind, recipe, device="cpu", precision=precision)
    assert cfg.dtype == dtype
    assert cfg.vlm_dtype == "float32"
    assert cfg.use_amp == (precision != "fp32")
    assert not cfg.use_flash_attn


def test_outer_scale_accumulates_unscales_and_restores():
    from lerobot_policy_vla_ttt.training.episode_parallel.precision import OuterPrecision

    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(2)
    precision = OuterPrecision("float16", initial_scale=8, growth_interval=2)
    for x in (1.0, 3.0):
        precision.backward((model(torch.tensor([[x]])) ** 2).sum() / 2)
    assert model.weight.grad.item() == 160
    assert precision.prepare_gradients(model)
    assert model.weight.grad.item() == 20
    precision.success()
    restored = OuterPrecision("float16", initial_scale=8, growth_interval=2)
    restored.load_state_dict(precision.state_dict())
    restored.success()
    assert restored.scale == 16
    model.weight.grad.fill_(float("inf"))
    assert not restored.prepare_gradients(model)
    assert restored.scale == 8
    assert restored.skipped_updates == 1


def test_nonfinite_forward_is_not_retried():
    from lerobot_policy_vla_ttt.training.episode_parallel.precision import OuterPrecision

    with pytest.raises(FloatingPointError, match="forward loss"):
        OuterPrecision("float16").backward(torch.tensor(float("inf"), requires_grad=True))


def test_fp16_requires_scaler_state_but_old_fp32_is_supported():
    from lerobot_policy_vla_ttt.training.episode_parallel.precision import OuterPrecision

    OuterPrecision("float32").load_state_dict(None)
    with pytest.raises(ValueError, match="scaler state"):
        OuterPrecision("float16").load_state_dict(None)
    with pytest.raises(ValueError, match="precision"):
        OuterPrecision("float16").load_state_dict(OuterPrecision("float32").state_dict())


@pytest.mark.parametrize("chunk", [False, True])
def test_overflow_retries_same_batch_and_resume(monkeypatch, tmp_path, chunk):
    from accelerate import Accelerator

    from lerobot_policy_vla_ttt.training.episode_parallel.precision import OuterPrecision
    from lerobot_policy_vla_ttt.training.episode_parallel.runner import run

    from .test_baseline import make_baseline
    from .test_checkpoint import assert_control_equal, source
    from .test_engine import make_policy

    torch.manual_seed(12)
    model = make_baseline(monkeypatch) if chunk else make_policy("t2mem", monkeypatch)
    model.config.dtype = "float16"  # CPU exercises scaling/control, not CUDA autocast.
    reference = make_baseline(monkeypatch) if chunk else make_policy("t2mem", monkeypatch)
    reference.load_state_dict(model.state_dict())
    reference.config.dtype = "float32"
    recipe = Recipe(
        steps=2,
        phase_length=1,
        warmup_steps=1,
        global_batch_size=2,
        horizon=3,
        execution=2,
        tbptt_events=2,
        training_mode="policy" if chunk else "alternating",
        sampling_mode="mixed_tasks",
        update_unit="chunk" if chunk else "episode",
    )
    common = dict(
        recipe=recipe, accelerator=Accelerator(cpu=True, mixed_precision="no"), identity={}
    )
    expected = run(reference, source(), None, None, output=tmp_path / "reference", **common)
    original = OuterPrecision.prepare_gradients
    calls = []

    def inject_once(self, policy):
        calls.append(self.scale)
        if len(calls) == 1 and common["accelerator"].process_index == 0:
            next(p for p in policy.parameters() if p.grad is not None).grad.fill_(float("inf"))
        return original(self, policy)

    monkeypatch.setattr(OuterPrecision, "prepare_gradients", inject_once)
    run(model, source(), None, None, output=tmp_path / "split", stop_after=1, **common)
    assert calls == [65536, 32768]
    saved = torch.load(tmp_path / "split/step-00000001/training.pt", weights_only=False)
    assert saved["control"]["precision"]["scale"] == 32768
    assert saved["control"]["precision"]["skipped_updates"] == 1
    restored = make_baseline(monkeypatch) if chunk else make_policy("t2mem", monkeypatch)
    restored.config.dtype = "float16"
    actual = run(
        restored,
        source(),
        None,
        None,
        output=tmp_path / "resumed",
        resume=tmp_path / "split/step-00000001",
        **common,
    )
    assert calls[-1] == 32768
    assert actual == expected
    torch.testing.assert_close(reference.state_dict(), restored.state_dict(), atol=0, rtol=0)
    a = torch.load(tmp_path / "reference/step-00000002/training.pt", weights_only=False)
    b = torch.load(tmp_path / "resumed/step-00000002/training.pt", weights_only=False)
    assert_control_equal(a["control"], b["control"])


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
def test_policy_autocast_selection_without_gpu(dtype):
    from types import SimpleNamespace

    from lerobot_policy_vla_ttt.models.evo1_groot.modeling_evo1_groot import Evo1GrootPolicy

    # Constructing the context does not allocate CUDA tensors; inspect dispatch configuration.
    obj = SimpleNamespace(config=SimpleNamespace(dtype=dtype), _device=torch.device("cuda"))
    obj._amp_enabled = Evo1GrootPolicy._amp_enabled.fget(obj)
    assert obj._amp_enabled == (dtype != "float32")


def test_v100_rejects_bf16_before_model_load(monkeypatch):
    from lerobot_policy_vla_ttt.training.episode_parallel.precision import validate_device

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (7, 0))
    with pytest.raises(ValueError, match="V100"):
        validate_device("bfloat16", torch.device("cuda:0"))
    validate_device("float16", torch.device("cuda:0"))


@pytest.mark.parametrize("kind", ["mac", "mag", "mal", "aqmem", "t2mem"])
def test_scaled_outer_backward_preserves_inner_update_and_gradients(monkeypatch, kind):
    from lerobot_policy_vla_ttt.training.episode_parallel.engine import (
        backward_episode,
        episode_rng,
    )
    from lerobot_policy_vla_ttt.training.episode_parallel.precision import OuterPrecision

    from .test_engine import episode_events, make_policy

    first = make_policy(kind, monkeypatch)
    second = make_policy(kind, monkeypatch)
    second.load_state_dict(first.state_dict())
    recipe = Recipe(global_batch_size=1, horizon=3, execution=2, tbptt_events=3)
    for policy, dtype in [(first, "float32"), (second, "float16")]:
        scale = OuterPrecision(dtype, initial_scale=128)
        with episode_rng(57, torch.device("cpu")):
            backward_episode(
                policy, episode_events(), decisions=3, recipe=recipe, backward=scale.backward
            )
        assert scale.prepare_gradients(policy)
    for (name, a), (_, b) in zip(first.named_parameters(), second.named_parameters(), strict=True):
        assert (a.grad is None) == (b.grad is None), name
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, atol=0, rtol=0)


@pytest.mark.parametrize(
    "change",
    [{"scale": float("nan")}, {"scale": 0}, {"good_updates": 2000}, {"skipped_updates": -1}],
)
def test_corrupt_scale_checkpoint_rejected(change):
    from lerobot_policy_vla_ttt.training.episode_parallel.precision import OuterPrecision

    scaler = OuterPrecision("float16")
    with pytest.raises(ValueError, match="scaler state"):
        scaler.load_state_dict(scaler.state_dict() | change)
