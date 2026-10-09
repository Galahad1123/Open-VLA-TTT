"""MAL policy with explicit VLM memory and unchanged GR00T-style DiT."""

from ..titans.policy import TitansPolicy
from .configuration_mal import MALConfig
from .connection import MALConnection


class MALPolicy(TitansPolicy):
    config_class = MALConfig
    connection_class = MALConnection
    name = "mal"
