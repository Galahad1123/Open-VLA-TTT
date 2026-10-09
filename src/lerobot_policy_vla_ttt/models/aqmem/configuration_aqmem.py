from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig

from ..observation_memory.configuration import ObservationMemoryConfig


@PreTrainedConfig.register_subclass("aqmem")
@dataclass
class AQMemConfig(ObservationMemoryConfig):
    pass
