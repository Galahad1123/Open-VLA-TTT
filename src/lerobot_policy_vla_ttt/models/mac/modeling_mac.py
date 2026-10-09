"""MAC policy with explicit VLM memory and unchanged GR00T-style DiT."""

from ..titans.policy import TitansPolicy
from .configuration_mac import MACConfig
from .connection import MACConnection


class MACPolicy(TitansPolicy):
    config_class = MACConfig
    connection_class = MACConnection
    name = "mac"
