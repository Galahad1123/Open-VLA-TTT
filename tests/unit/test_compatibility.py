from __future__ import annotations

import importlib.util
from importlib import metadata

import pytest

from lerobot_policy_vla_ttt.compatibility import (
    SUPPORTED_LEROBOT_VERSION,
    ensure_supported_lerobot_version,
    require_optional_module,
)


def test_accepts_exact_supported_lerobot_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metadata, "version", lambda package: "0.6.1")

    ensure_supported_lerobot_version()

    assert SUPPORTED_LEROBOT_VERSION == "0.6.1"


def test_rejects_other_lerobot_version_with_both_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(metadata, "version", lambda package: "0.6.2")

    with pytest.raises(RuntimeError) as error:
        ensure_supported_lerobot_version()

    message = str(error.value)
    assert "0.6.2" in message
    assert "0.6.1" in message


def test_missing_optional_module_names_install_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda module_name: None)

    with pytest.raises(RuntimeError) as error:
        require_optional_module("libero", install_command="uv sync --extra libero")

    message = str(error.value)
    assert "libero" in message
    assert "uv sync --extra libero" in message
