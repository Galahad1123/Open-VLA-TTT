"""Thin Evo-1 policy subclass for VLA-TTT research extensions."""

from __future__ import annotations

from typing import Any

from lerobot.policies.evo1.modeling_evo1 import Evo1Policy

from .compatibility import ensure_supported_lerobot_version
from .configuration_vla_ttt import VLATTTConfig


class VLATTTPolicy(Evo1Policy):
    """Evo-1 policy with an explicit, initially empty extension seam."""

    config_class = VLATTTConfig
    name = "vla_ttt"

    def __init__(
        self,
        config: VLATTTConfig,
        *,
        vlm_hub_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        ensure_supported_lerobot_version()
        super().__init__(config, vlm_hub_kwargs=vlm_hub_kwargs, **kwargs)
        self.install_extensions()

    def install_extensions(self) -> None:
        """Install project-owned model extensions; the baseline intentionally does nothing."""
