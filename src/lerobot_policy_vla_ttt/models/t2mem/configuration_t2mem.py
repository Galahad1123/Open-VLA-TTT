from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig

from ..observation_memory.configuration import ObservationMemoryConfig


@PreTrainedConfig.register_subclass("t2mem")
@dataclass
class T2MemConfig(ObservationMemoryConfig):
    @property
    def interface_depth(self):
        # No unused final write module: its output would have no subsequent block.
        return self.num_layers - 1
