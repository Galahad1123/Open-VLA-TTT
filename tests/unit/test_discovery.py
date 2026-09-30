from __future__ import annotations

from lerobot.policies.factory import get_policy_class, make_policy_config

import lerobot_policy_vla_ttt
from lerobot_policy_vla_ttt.configuration_vla_ttt import VLATTTConfig
from lerobot_policy_vla_ttt.modeling_vla_ttt import VLATTTPolicy


def test_public_factory_discovers_vla_ttt_config() -> None:
    config = make_policy_config("vla_ttt", device="cpu", use_amp=False)

    assert isinstance(config, VLATTTConfig)
    assert config.type == "vla_ttt"
    assert lerobot_policy_vla_ttt.VLATTTConfig is VLATTTConfig


def test_public_factory_discovers_vla_ttt_policy() -> None:
    assert get_policy_class("vla_ttt") is VLATTTPolicy
    assert lerobot_policy_vla_ttt.VLATTTPolicy is VLATTTPolicy
