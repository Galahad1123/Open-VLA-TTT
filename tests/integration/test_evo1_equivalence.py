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

    assert type(plugin.model).__module__.startswith("lerobot_policy_vla_ttt.evo1.")
    assert type(plugin.model.action_head).__module__.startswith("lerobot_policy_vla_ttt.evo1.")
    assert type(plugin.model.embedder).__module__.startswith("lerobot_policy_vla_ttt.evo1.")
    assert native.state_dict().keys() == plugin.state_dict().keys()
    assert native_actions.shape == plugin_actions.shape
    torch.testing.assert_close(native_actions, plugin_actions, atol=ATOL, rtol=RTOL)

    # Training-mode loss and action-head gradients, with the same stochastic state.
    action_dim = native.config.output_features["action"].shape[-1]
    raw_batch["action"] = torch.zeros(1, native.config.chunk_size, action_dim)
    native_batch = native_pre(raw_batch)
    plugin_batch = plugin_pre(raw_batch)
    native.train()
    plugin.train()
    state = _rng_state()
    native_loss, native_info = native(native_batch)
    native_loss.backward()
    _restore_rng(state)
    plugin_loss, plugin_info = plugin(plugin_batch)
    plugin_loss.backward()
    torch.testing.assert_close(native_loss, plugin_loss, atol=ATOL, rtol=RTOL)
    assert native_info["active_action_dims"] == plugin_info["active_action_dims"]
    native_params = dict(native.model.action_head.named_parameters())
    compared = 0
    for name, param in plugin.model.action_head.named_parameters():
        reference = native_params[name].grad
        assert (param.grad is None) == (reference is None)
        if reference is not None:
            torch.testing.assert_close(param.grad, reference, atol=ATOL, rtol=RTOL)
            compared += 1
    assert compared > 0


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
    reloaded_pre, _ = make_vla_ttt_pre_post_processors_from_pretrained(
        reloaded.config,
        converted,
        local_files_only=True,
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
