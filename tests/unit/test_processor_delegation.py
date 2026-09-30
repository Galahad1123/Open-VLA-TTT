from __future__ import annotations

from typing import Any

import pytest
import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.evo1.processor_evo1 import make_evo1_pre_post_processors
from lerobot.processor import NormalizerProcessorStep, UnnormalizerProcessorStep
from lerobot.utils.constants import ACTION, OBS_STATE

from lerobot_policy_vla_ttt import processor_vla_ttt
from lerobot_policy_vla_ttt.configuration_vla_ttt import VLATTTConfig


def _config() -> VLATTTConfig:
    return VLATTTConfig(
        device="cpu",
        use_amp=False,
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(7,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )


def _stats() -> dict[str, dict[str, torch.Tensor]]:
    return {
        OBS_STATE: {"min": torch.zeros(7), "max": torch.ones(7)},
        ACTION: {"min": torch.zeros(7), "max": torch.ones(7)},
    }


def _normalization_shapes(pipeline: Any) -> list[tuple[int, ...]]:
    shapes: list[tuple[int, ...]] = []
    for step in pipeline.steps:
        if isinstance(step, (NormalizerProcessorStep, UnnormalizerProcessorStep)):
            for feature_stats in step.stats.values():
                shapes.extend(tuple(value.shape) for value in feature_stats.values())
    return shapes


def test_fresh_factory_returns_exact_upstream_result(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config()
    stats = _stats()
    expected = (object(), object())
    received: list[tuple[object, object]] = []

    def fake_factory(config_arg, dataset_stats=None):
        received.append((config_arg, dataset_stats))
        return expected

    monkeypatch.setattr(processor_vla_ttt, "make_evo1_pre_post_processors", fake_factory)

    result = processor_vla_ttt.make_vla_ttt_pre_post_processors(config, dataset_stats=stats)

    assert result is expected
    assert received == [(config, stats)]


def test_fresh_factory_propagates_upstream_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args, **kwargs):
        raise ValueError("upstream feature mismatch")

    monkeypatch.setattr(processor_vla_ttt, "make_evo1_pre_post_processors", fail)

    with pytest.raises(ValueError, match="upstream feature mismatch"):
        processor_vla_ttt.make_vla_ttt_pre_post_processors(_config(), dataset_stats=_stats())


def test_pretrained_factory_forwards_every_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config()
    stats = _stats()
    expected = (object(), object())
    received: list[dict[str, object]] = []

    def fake_factory(**kwargs):
        received.append(kwargs)
        return expected

    monkeypatch.setattr(processor_vla_ttt, "make_pre_post_processors", fake_factory)

    result = processor_vla_ttt.make_vla_ttt_pre_post_processors_from_pretrained(
        config,
        "/checkpoint",
        revision="abc123",
        dataset_stats=stats,
        dataset_meta="meta",
        preprocessor_overrides={"device_processor": {"device": "cpu"}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
        preprocessor_config_filename="pre.json",
        postprocessor_config_filename="post.json",
    )

    assert result is expected
    assert received == [
        {
            "policy_cfg": config,
            "pretrained_path": "/checkpoint",
            "pretrained_revision": "abc123",
            "dataset_stats": stats,
            "dataset_meta": "meta",
            "preprocessor_overrides": {"device_processor": {"device": "cpu"}},
            "postprocessor_overrides": {"device_processor": {"device": "cpu"}},
            "preprocessor_config_filename": "pre.json",
            "postprocessor_config_filename": "post.json",
        }
    ]


def test_fresh_and_pretrained_paths_preserve_evo1_stat_padding(tmp_path) -> None:
    config = _config()
    stats = _stats()
    upstream_pre, upstream_post = make_evo1_pre_post_processors(config, dataset_stats=stats)
    plugin_pre, plugin_post = processor_vla_ttt.make_vla_ttt_pre_post_processors(
        config, dataset_stats=stats
    )

    assert [type(step) for step in plugin_pre.steps] == [type(step) for step in upstream_pre.steps]
    assert [type(step) for step in plugin_post.steps] == [
        type(step) for step in upstream_post.steps
    ]
    assert _normalization_shapes(plugin_pre) == [(24,), (24,), (24,), (24,)]
    assert _normalization_shapes(plugin_post) == [(24,), (24,), (24,), (24,)]

    upstream_pre.save_pretrained(tmp_path)
    upstream_post.save_pretrained(tmp_path)
    loaded_pre, loaded_post = processor_vla_ttt.make_vla_ttt_pre_post_processors_from_pretrained(
        config,
        tmp_path,
        dataset_stats=stats,
    )

    assert _normalization_shapes(loaded_pre) == [(24,), (24,), (24,), (24,)]
    assert _normalization_shapes(loaded_post) == [(24,), (24,), (24,), (24,)]
