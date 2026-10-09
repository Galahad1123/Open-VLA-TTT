"""MAG policy with explicit VLM memory and unchanged GR00T-style DiT."""

from ..titans.policy import TitansPolicy
from .configuration_mag import MAGConfig
from .connection import MAGConnection


class MAGPolicy(TitansPolicy):
    config_class = MAGConfig
    connection_class = MAGConnection
    name = "mag"
