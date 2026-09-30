from __future__ import annotations

import subprocess
import sys

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


def test_lerobot_automatically_discovers_installed_plugin_in_clean_process() -> None:
    code = """
import sys
from lerobot.utils.import_utils import register_third_party_plugins

register_third_party_plugins()
assert "lerobot_policy_vla_ttt" in sys.modules

from lerobot.policies.factory import make_policy_config
config = make_policy_config("vla_ttt", device="cpu", use_amp=False)
assert type(config).__name__ == "VLATTTConfig"
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
