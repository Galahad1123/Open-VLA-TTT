"""LeRobot policy plugin for VLA test-time training research."""

from .compatibility import (
    SUPPORTED_LEROBOT_VERSION,
    ensure_supported_lerobot_version,
    require_optional_module,
)
from .configuration_vla_ttt import VLATTTConfig
from .modeling_vla_ttt import VLATTTPolicy

__all__ = [
    "SUPPORTED_LEROBOT_VERSION",
    "VLATTTConfig",
    "VLATTTPolicy",
    "ensure_supported_lerobot_version",
    "require_optional_module",
]
