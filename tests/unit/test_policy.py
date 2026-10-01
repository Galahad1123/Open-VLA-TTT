from __future__ import annotations

import pytest

from lerobot_policy_vla_ttt import modeling_vla_ttt
from lerobot_policy_vla_ttt.configuration_vla_ttt import VLATTTConfig
from lerobot_policy_vla_ttt.evo1.modeling_evo1 import Evo1Policy
from lerobot_policy_vla_ttt.modeling_vla_ttt import VLATTTPolicy


def test_constructor_checks_version_then_initializes_parent_then_extensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []
    config = VLATTTConfig(device="cpu", use_amp=False)
    hub_kwargs = {"local_files_only": True}

    monkeypatch.setattr(
        modeling_vla_ttt,
        "ensure_supported_lerobot_version",
        lambda: calls.append(("compatibility",)),
    )

    def fake_parent_init(self, config_arg, *, vlm_hub_kwargs=None, **kwargs):
        calls.append(("parent", config_arg, vlm_hub_kwargs, kwargs))

    monkeypatch.setattr(Evo1Policy, "__init__", fake_parent_init)
    monkeypatch.setattr(
        VLATTTPolicy,
        "install_extensions",
        lambda self: calls.append(("extensions", self)),
    )

    policy = VLATTTPolicy(config, vlm_hub_kwargs=hub_kwargs, marker="value")

    assert calls == [
        ("compatibility",),
        ("parent", config, hub_kwargs, {"marker": "value"}),
        ("extensions", policy),
    ]


def test_default_extension_hook_keeps_parent_model_and_inherited_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel_model = object()
    original_config_class = Evo1Policy.config_class
    original_name = Evo1Policy.name

    monkeypatch.setattr(modeling_vla_ttt, "ensure_supported_lerobot_version", lambda: None)

    def fake_parent_init(self, config, *, vlm_hub_kwargs=None, **kwargs):
        self.config = config
        self.model = sentinel_model

    monkeypatch.setattr(Evo1Policy, "__init__", fake_parent_init)

    policy = VLATTTPolicy(VLATTTConfig(device="cpu", use_amp=False))

    assert policy.model is sentinel_model
    assert Evo1Policy.config_class is original_config_class
    assert Evo1Policy.name == original_name
    assert VLATTTPolicy.reset is Evo1Policy.reset
    assert VLATTTPolicy.select_action is Evo1Policy.select_action
    assert VLATTTPolicy.predict_action_chunk is Evo1Policy.predict_action_chunk
