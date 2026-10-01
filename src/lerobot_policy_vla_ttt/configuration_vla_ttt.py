"""Configuration for the out-of-tree VLA-TTT policy."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields
from typing import Self

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.evo1.configuration_evo1 import Evo1Config as UpstreamEvo1Config

from .compatibility import SUPPORTED_LEROBOT_VERSION
from .evo1.configuration_evo1 import Evo1Config


@PreTrainedConfig.register_subclass("vla_ttt")
@dataclass
class VLATTTConfig(Evo1Config):
    """Evo-1 configuration tagged for the VLA-TTT plugin."""

    plugin_schema_version: int = 1
    tested_lerobot_version: str = SUPPORTED_LEROBOT_VERSION

    @classmethod
    def from_evo1_config(cls, config: Evo1Config) -> Self:
        """Create an independent plugin config from an Evo-1 config."""
        if not isinstance(config, (Evo1Config, UpstreamEvo1Config)):
            raise TypeError(f"Expected Evo1Config, got {type(config).__name__}")

        values = {
            field.name: deepcopy(getattr(config, field.name))
            for field in fields(Evo1Config)
            if field.init
        }
        return cls(**values)
