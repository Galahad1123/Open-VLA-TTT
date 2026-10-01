import shutil
import subprocess
from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.evo1.configuration_evo1 import Evo1Config as NativeConfig
from lerobot.policies.evo1.modeling_evo1 import Evo1Policy as NativePolicy
from lerobot.policies.evo1.processor_evo1 import make_evo1_pre_post_processors as native_processors
from lerobot.processor import ProcessorStepRegistry

from lerobot_policy_vla_ttt.cli import local_processor_routing, make_pre_post_processors
from lerobot_policy_vla_ttt.configuration_vla_ttt import VLATTTConfig
from lerobot_policy_vla_ttt.evo1.configuration_evo1 import Evo1Config
from lerobot_policy_vla_ttt.evo1.modeling_evo1 import Evo1Policy
from lerobot_policy_vla_ttt.modeling_vla_ttt import VLATTTPolicy


def test_local_hierarchy_preserves_native_registration_and_config_fields():
    assert issubclass(VLATTTPolicy, Evo1Policy)
    assert not issubclass(VLATTTPolicy, NativePolicy)
    assert issubclass(VLATTTConfig, Evo1Config)
    assert not issubclass(VLATTTConfig, NativeConfig)
    assert PreTrainedConfig.get_choice_class("evo1") is NativeConfig
    assert PreTrainedConfig.get_choice_class("vla_ttt") is VLATTTConfig
    assert [(f.name, f.type, f.init) for f in fields(Evo1Config)] == [
        (f.name, f.type, f.init) for f in fields(NativeConfig)
    ]
    for name in ("pad_state", "pad_action", "action"):
        native = ProcessorStepRegistry.get(f"evo1_{name}_processor")
        local = ProcessorStepRegistry.get(f"vla_ttt_evo1_{name}_processor")
        assert native is not local
        assert native.__module__.startswith("lerobot.policies.evo1.")
        assert local.__module__.startswith("lerobot_policy_vla_ttt.evo1.")


def test_legacy_processor_conversion_and_resume_preserve_extras_and_overrides(tmp_path):
    from lerobot.configs.types import FeatureType, PolicyFeature

    config = VLATTTConfig(
        device="cpu",
        use_amp=False,
        task_field="custom_prompt",
        input_features={"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,))},
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )
    stats = {
        key: {"min": torch.zeros(7), "max": torch.ones(7)}
        for key in ("observation.state", "action")
    }
    native_pre, native_post = native_processors(config, stats)
    native_pre.save_pretrained(tmp_path)
    native_post.save_pretrained(tmp_path)
    pre, post = make_pre_post_processors(config, str(tmp_path), local_files_only=True)
    raw = {
        "observation.state": torch.ones(7),
        "action": torch.ones(4, 7),
        "embodiment_id": torch.tensor(0),
        "custom_prompt": "remember this",
        "state_mask": torch.ones(7, dtype=torch.bool),
    }
    expected, actual = native_pre(raw), pre(raw)
    assert expected.keys() == actual.keys()
    for key in expected:
        if isinstance(expected[key], torch.Tensor):
            torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
        else:
            assert actual[key] == expected[key]
    probe = torch.linspace(-1, 1, 24).reshape(1, 24)
    torch.testing.assert_close(post(probe), native_post(probe), atol=0, rtol=0)
    assert all(
        not type(step).__module__.startswith("lerobot.policies.evo1.")
        for pipeline in (pre, post)
        for step in pipeline.steps
    )
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)
    config.binarize_gripper = True
    resumed_pre, resumed_post = make_pre_post_processors(
        config,
        str(tmp_path),
        local_files_only=True,
        preprocessor_overrides={"normalizer_processor": {"stats": stats}},
    )
    assert "custom_prompt" in resumed_pre(raw)
    assert resumed_post(probe)[0, 6].item() in (-1, 1)


def test_runner_routing_is_scoped_and_restored_on_error():
    original = object()
    runner = SimpleNamespace(make_pre_post_processors=original)
    with pytest.raises(RuntimeError):
        with local_processor_routing(runner):
            assert runner.make_pre_post_processors is make_pre_post_processors
            raise RuntimeError("runner failed")
    assert runner.make_pre_post_processors is original


@pytest.mark.parametrize("command", ["vla-ttt-train", "vla-ttt-eval", "vla-ttt-rollout"])
def test_local_cli_preserves_lerobot_help(command):
    executable = shutil.which(command)
    assert executable is not None
    result = subprocess.run([executable, "--help"], capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stderr
    assert "--policy" in result.stdout


def test_adapter_delegates_native_config_unchanged(monkeypatch):
    from lerobot_policy_vla_ttt import cli

    sentinel = object()
    calls = []
    monkeypatch.setattr(
        cli, "upstream_processors", lambda *a, **k: calls.append((a, k)) or sentinel
    )
    config = NativeConfig(device="cpu", use_amp=False)
    assert cli.make_pre_post_processors(config, "/native", "revision") is sentinel
    assert calls == [((config, "/native", "revision"), {})]
