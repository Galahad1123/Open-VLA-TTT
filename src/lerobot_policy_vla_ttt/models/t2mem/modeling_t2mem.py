from ..observation_memory.policy import ObservationPolicy
from .action_head import T2MemActionHead
from .configuration_t2mem import T2MemConfig


class T2MemPolicy(ObservationPolicy):
    name = "t2mem"
    config_class = T2MemConfig
    head_class = T2MemActionHead
