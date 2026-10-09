from dataclasses import replace

import pytest
import torch

from lerobot_policy_vla_ttt.models.titans.attention import attention_mask
from lerobot_policy_vla_ttt.models.titans.language import language_step


def inputs():
    x = torch.randn(2, 4, 16)
    valid = torch.tensor([[True, True, True, False], [True] * 4])
    visual = torch.tensor([[True, True, False, False]] * 2)
    return x, valid, visual


def test_attention_prefix_and_causality():
    mask = attention_mask(torch.tensor([[True, True, False]]), torch.tensor([[True, False]]))
    assert torch.isfinite(mask[0, 0]).tolist() == [
        [True, False, True, False, False],
        [True, False, True, True, False],
        [True, False, True, True, False],
    ]


@pytest.mark.parametrize("rule", ["gradient_descent", "momentum_decay"])
def test_two_decisions_outer_gradients_and_state_isolation(setup, rule):
    config = replace(setup.config, memory_update_rule=rule)
    policy = setup.policy_class(config)
    e = policy.model.embedder
    x, valid, visual = inputs()
    first, s = language_step(e.model.language_model, e.connections, config, x, valid, visual)
    saved = tuple(tuple(w.clone() for w in state.weights) for state in s)
    y, after = language_step(
        e.model.language_model, e.connections, config, x + 0.2, valid, visual, s
    )
    y.square().mean().backward()
    for connection in e.connections.values():
        assert connection.memory.fast.w1.grad.abs().sum() > 0
        assert connection.memory.update.raw_lr.grad.abs().sum() > 0
    assert all(state.update_count.tolist() == [2, 2] for state in after)
    for old, state in zip(saved, s, strict=True):
        torch.testing.assert_close(old, state.weights)
    changed = x.clone()
    changed[1] += 20
    changed[0, -1] = float("nan")  # Invalid padding cannot pollute any state.
    other, s2 = language_step(e.model.language_model, e.connections, config, changed, valid, visual)
    torch.testing.assert_close(first[0], other[0])
    for a, b in zip(s, s2, strict=True):
        for wa, wb in zip(a.weights, b.weights, strict=True):
            torch.testing.assert_close(wa[0], wb[0])


def test_disabled_path_matches_actual_qwen2(setup):
    cfg = replace(setup.config, memory_enabled=False)
    policy = setup.policy_class(cfg)
    e = policy.model.embedder
    x, valid, visual = inputs()
    y, state = language_step(e.model.language_model, e.connections, cfg, x, valid, visual)
    expected = e.model.language_model(
        inputs_embeds=x, attention_mask=valid, use_cache=False
    ).last_hidden_state
    torch.testing.assert_close(y[valid], expected[valid])
    assert state is None


@pytest.mark.parametrize("granularity,count", [("observation", 1), ("token", 2)])
def test_visual_only_masks_every_memory_input_and_clocks(setup, granularity, count):
    cfg = replace(setup.config, memory_input="visual", memory_granularity=granularity)
    policy = setup.policy_class(cfg)
    e = policy.model.embedder
    x, valid, visual = inputs()
    y, state = language_step(e.model.language_model, e.connections, cfg, x, valid, visual)
    changed = x.clone()
    changed[:, 2:] += 50  # Later text cannot change the earlier visual positions.
    _, other = language_step(e.model.language_model, e.connections, cfg, changed, valid, visual)
    for a, b in zip(state, other, strict=True):
        torch.testing.assert_close(a.weights, b.weights)
        assert a.update_count.tolist() == [count, count]
        assert a.decision_index.tolist() == [1, 1]
    assert y.isfinite().all()


def test_checkpoint_recomputation_preserves_states_outputs_and_gradients(setup):
    plain = setup.policy_class(setup.config)
    checked = setup.policy_class(replace(setup.config, enable_gradient_checkpointing=True))
    checked.load_state_dict(plain.state_dict())
    x, valid, visual = inputs()
    results = []
    for policy in (plain, checked):
        e = policy.model.embedder
        state = None
        for _ in range(2):
            y, state = language_step(
                e.model.language_model, e.connections, policy.config, x, valid, visual, state
            )
        y.square().sum().backward()
        results.append((y, state))
    torch.testing.assert_close(results[0][0], results[1][0])
    for a, b in zip(results[0][1], results[1][1], strict=True):
        torch.testing.assert_close(a.weights, b.weights)
        assert b.update_count.tolist() == [2, 2]
    for (n, a), (_, b) in zip(plain.named_parameters(), checked.named_parameters(), strict=True):
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, msg=n)
