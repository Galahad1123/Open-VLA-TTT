from __future__ import annotations

import gc

import pytest
import torch
from lerobot.policies.evo1.modeling_evo1 import Evo1Policy
from lerobot.policies.factory import make_pre_post_processors

from lerobot_policy_vla_ttt.checkpoint import convert_evo1_checkpoint
from lerobot_policy_vla_ttt.modeling_vla_ttt import VLATTTPolicy
from lerobot_policy_vla_ttt.processor_vla_ttt import (
    make_vla_ttt_pre_post_processors_from_pretrained,
)

ATOL = 5e-3
RTOL = 5e-3


def _rng_state() -> tuple[torch.Tensor, list[torch.Tensor]]:
    return torch.random.get_rng_state(), torch.cuda.get_rng_state_all()


def _restore_rng(state: tuple[torch.Tensor, list[torch.Tensor]]) -> None:
    cpu_state, cuda_state = state
    torch.random.set_rng_state(cpu_state)
    torch.cuda.set_rng_state_all(cuda_state)


@pytest.mark.integration
def test_native_evo1_and_plugin_are_numerically_equivalent(
    evo1_checkpoint, synthetic_batch_factory
) -> None:
    checkpoint, revision = evo1_checkpoint
    native = Evo1Policy.from_pretrained(checkpoint, revision=revision)
    plugin = VLATTTPolicy.from_evo1_pretrained(checkpoint, revision=revision)
    native_pre, _ = make_pre_post_processors(
        native.config,
        pretrained_path=checkpoint,
        pretrained_revision=revision,
    )
    plugin_pre, _ = make_vla_ttt_pre_post_processors_from_pretrained(
        plugin.config,
        checkpoint,
        revision=revision,
    )
    raw_batch = synthetic_batch_factory(native.config)
    native_batch = native_pre(raw_batch)
    plugin_batch = plugin_pre(raw_batch)

    native.reset()
    plugin.reset()
    torch.manual_seed(20260930)
    torch.cuda.manual_seed_all(20260930)
    state = _rng_state()
    native_actions = native.predict_action_chunk(native_batch)
    _restore_rng(state)
    plugin_actions = plugin.predict_action_chunk(plugin_batch)

    assert native_actions.shape == plugin_actions.shape
    torch.testing.assert_close(native_actions, plugin_actions, atol=ATOL, rtol=RTOL)


@pytest.mark.integration
def test_plugin_checkpoint_round_trip_preserves_output_and_reset(
    evo1_checkpoint, synthetic_batch_factory, tmp_path
) -> None:
    checkpoint, revision = evo1_checkpoint
    converted = convert_evo1_checkpoint(
        checkpoint,
        tmp_path / "vla-ttt-checkpoint",
        revision=revision,
    )
    original = VLATTTPolicy.from_evo1_pretrained(checkpoint, revision=revision)
    reloaded = VLATTTPolicy.from_pretrained(converted, local_files_only=True)
    original_pre, _ = make_vla_ttt_pre_post_processors_from_pretrained(
        original.config,
        checkpoint,
        revision=revision,
    )
    reloaded_pre, _ = make_pre_post_processors(
        reloaded.config,
        pretrained_path=str(converted),
    )
    raw_batch = synthetic_batch_factory(original.config)
    original_batch = original_pre(raw_batch)
    reloaded_batch = reloaded_pre(raw_batch)

    original.reset()
    reloaded.reset()
    torch.manual_seed(20260930)
    torch.cuda.manual_seed_all(20260930)
    state = _rng_state()
    original_actions = original.predict_action_chunk(original_batch)
    _restore_rng(state)
    reloaded_actions = reloaded.predict_action_chunk(reloaded_batch)
    torch.testing.assert_close(original_actions, reloaded_actions, atol=ATOL, rtol=RTOL)

    reloaded.select_action(reloaded_batch)
    assert len(reloaded._action_queue) > 0
    reloaded.reset()
    assert len(reloaded._action_queue) == 0

    del original, reloaded
    gc.collect()
    torch.cuda.empty_cache()
