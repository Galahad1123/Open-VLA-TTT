"""LeRobot policy plugin for VLA test-time training research."""

from .checkpoint import convert_evo1_checkpoint
from .compatibility import (
    SUPPORTED_LEROBOT_VERSION,
    ensure_supported_lerobot_version,
    require_optional_module,
)
from .configuration_vla_ttt import VLATTTConfig
from .modeling_vla_ttt import VLATTTPolicy
from .models.evo1_groot.configuration_evo1_groot import Evo1GrootConfig
from .models.evo1_groot.modeling_evo1_groot import Evo1GrootPolicy
from .models.mac.configuration_mac import MACConfig
from .models.mac.modeling_mac import MACPolicy
from .models.mag.configuration_mag import MAGConfig
from .models.mag.modeling_mag import MAGPolicy
from .models.mal.configuration_mal import MALConfig
from .models.mal.modeling_mal import MALPolicy
from .models.robottt.configuration_robottt import RoboTTTConfig
from .models.robottt.modeling_robottt import RoboTTTPolicy

__all__ = [
    "MALConfig",
    "MALPolicy",
    "MAGConfig",
    "MAGPolicy",
    "MACConfig",
    "MACPolicy",
    "RoboTTTConfig",
    "RoboTTTPolicy",
    "Evo1GrootConfig",
    "Evo1GrootPolicy",
    "SUPPORTED_LEROBOT_VERSION",
    "VLATTTConfig",
    "VLATTTPolicy",
    "convert_evo1_checkpoint",
    "ensure_supported_lerobot_version",
    "require_optional_module",
]
