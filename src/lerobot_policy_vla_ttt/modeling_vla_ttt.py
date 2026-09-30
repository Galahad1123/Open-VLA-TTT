"""Thin Evo-1 policy subclass for VLA-TTT research extensions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lerobot.policies.evo1.modeling_evo1 import Evo1Policy

from .checkpoint import checkpoint_policy_type, load_state_dict_strict
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

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | Path,
        **kwargs: Any,
    ) -> VLATTTPolicy:
        """Load only a checkpoint already serialized as the plugin policy type."""
        if kwargs.get("strict", True) is not True:
            raise ValueError("VLATTTPolicy checkpoint loading requires strict=True.")
        kwargs["strict"] = True
        policy_type = checkpoint_policy_type(
            pretrained_name_or_path,
            revision=kwargs.get("revision"),
            local_files_only=kwargs.get("local_files_only", False),
            cache_dir=kwargs.get("cache_dir"),
            token=kwargs.get("token"),
        )
        if policy_type != "vla_ttt":
            if policy_type == "evo1":
                raise ValueError(
                    "Checkpoint type is 'evo1'; initialize it with from_evo1_pretrained()."
                )
            raise ValueError(
                f"Checkpoint type is {policy_type!r}; VLATTTPolicy.from_pretrained requires "
                "'vla_ttt'."
            )
        return super().from_pretrained(pretrained_name_or_path, **kwargs)

    @classmethod
    def from_evo1_pretrained(
        cls,
        pretrained_name_or_path: str | Path,
        *,
        revision: str | None = None,
        local_files_only: bool = False,
        **kwargs: Any,
    ) -> VLATTTPolicy:
        """Strictly initialize this plugin from an ordinary Evo-1 checkpoint."""
        if kwargs.get("strict", True) is not True:
            raise ValueError("VLATTTPolicy checkpoint loading requires strict=True.")
        kwargs["strict"] = True
        policy_type = checkpoint_policy_type(
            pretrained_name_or_path,
            revision=revision,
            local_files_only=local_files_only,
            cache_dir=kwargs.get("cache_dir"),
            token=kwargs.get("token"),
        )
        if policy_type != "evo1":
            raise ValueError(
                f"from_evo1_pretrained requires checkpoint type 'evo1', got {policy_type!r}."
            )

        source_policy = Evo1Policy.from_pretrained(
            pretrained_name_or_path,
            revision=revision,
            local_files_only=local_files_only,
            **kwargs,
        )
        config = VLATTTConfig.from_evo1_config(source_policy.config)
        constructor_kwargs = {}
        if "vlm_hub_kwargs" in kwargs:
            constructor_kwargs["vlm_hub_kwargs"] = kwargs["vlm_hub_kwargs"]
        policy = cls(config, **constructor_kwargs)
        load_state_dict_strict(policy, source_policy.state_dict())
        return policy
