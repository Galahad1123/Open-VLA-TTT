"""Runtime compatibility checks for supported upstream packages."""

from __future__ import annotations

import importlib.util
from importlib import metadata
from typing import Final

SUPPORTED_LEROBOT_VERSION: Final[str] = "0.6.1"


def ensure_supported_lerobot_version(installed_version: str | None = None) -> None:
    """Reject LeRobot versions not verified by this plugin."""
    current_version = installed_version or metadata.version("lerobot")
    if current_version != SUPPORTED_LEROBOT_VERSION:
        raise RuntimeError(
            "Unsupported LeRobot version "
            f"{current_version!r}; this plugin supports exactly {SUPPORTED_LEROBOT_VERSION!r}."
        )


def require_optional_module(module_name: str, *, install_command: str) -> None:
    """Raise an actionable error when an optional integration is unavailable."""
    if importlib.util.find_spec(module_name) is None:
        raise RuntimeError(
            f"Optional module {module_name!r} is not installed. Install it with: "
            f"{install_command}"
        )
