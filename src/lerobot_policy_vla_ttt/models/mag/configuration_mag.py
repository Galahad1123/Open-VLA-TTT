"""Independent MAG policy identity."""

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig

from ..titans.configuration import TitansConfig


@PreTrainedConfig.register_subclass("mag")
@dataclass
class MAGConfig(TitansConfig):
    pass
