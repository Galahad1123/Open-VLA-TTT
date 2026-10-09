from ..observation_memory.policy import ObservationPolicy
from .action_head import AQMemActionHead
from .configuration_aqmem import AQMemConfig


class AQMemPolicy(ObservationPolicy):
    name = "aqmem"
    config_class = AQMemConfig
    head_class = AQMemActionHead
