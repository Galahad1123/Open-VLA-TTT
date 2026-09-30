from __future__ import annotations

import lerobot_policy_vla_ttt
from lerobot.policies.factory import make_policy_config
from lerobot_policy_vla_ttt.configuration_vla_ttt import VLATTTConfig


def test_public_factory_discovers_vla_ttt_config() -> None:
    config = make_policy_config("vla_ttt", device="cpu", use_amp=False)

    assert isinstance(config, VLATTTConfig)
    assert config.type == "vla_ttt"
    assert lerobot_policy_vla_ttt.VLATTTConfig is VLATTTConfig
