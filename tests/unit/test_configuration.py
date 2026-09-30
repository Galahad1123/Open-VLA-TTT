from __future__ import annotations

from dataclasses import fields

import pytest

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.evo1.configuration_evo1 import Evo1Config
from lerobot_policy_vla_ttt.configuration_vla_ttt import VLATTTConfig


def test_config_round_trip_preserves_parent_fields_and_plugin_metadata(tmp_path) -> None:
    config = VLATTTConfig(
        device="cpu",
        use_amp=False,
        chunk_size=12,
        n_action_steps=7,
        tags=["baseline"],
    )

    config.save_pretrained(tmp_path)
    restored = PreTrainedConfig.from_pretrained(tmp_path)

    assert isinstance(restored, VLATTTConfig)
    assert restored.chunk_size == 12
    assert restored.n_action_steps == 7
    assert restored.tags == ["baseline"]
    assert restored.plugin_schema_version == 1
    assert restored.tested_lerobot_version == "0.6.1"
    assert restored.type == "vla_ttt"


def test_from_evo1_config_copies_every_init_field_without_aliasing() -> None:
    source = Evo1Config(device="cpu", use_amp=False, tags=["source"])

    converted = VLATTTConfig.from_evo1_config(source)

    for field in fields(Evo1Config):
        if field.init:
            assert getattr(converted, field.name) == getattr(source, field.name)
    assert converted.tags is not source.tags
    assert converted.normalization_mapping is not source.normalization_mapping
    assert converted.input_features is not source.input_features
    assert converted.output_features is not source.output_features


def test_from_evo1_config_rejects_other_types() -> None:
    with pytest.raises(TypeError, match="Evo1Config"):
        VLATTTConfig.from_evo1_config(object())  # type: ignore[arg-type]
