"""Hand-calculated state transitions, exact ablations and masked token isolation."""

import pytest
import torch

from lerobot_policy_vla_ttt.ttt.memory import AssociativeMemory
from lerobot_policy_vla_ttt.ttt.updates.gradient_descent import GradientDescent
from lerobot_policy_vla_ttt.ttt.updates.momentum_decay import MomentumDecay


def test_momentum_decay_matches_two_hand_steps_and_freezes_masked_rows():
    update = MomentumDecay(0.1, 3, momentum_init=0.5, decay_init=0.2, data_dependent=False).double()
    weights = (torch.tensor([[2.0], [4.0]], dtype=torch.double),)
    grads = (torch.tensor([[3.0], [7.0]], dtype=torch.double),)
    mask = torch.tensor([True, False])
    features = torch.zeros(2, 3, dtype=torch.double)
    w, s = update.step(weights, grads, mask, features=features)
    torch.testing.assert_close(w[0], torch.tensor([[1.3], [4.0]], dtype=torch.double))
    torch.testing.assert_close(s[0], torch.tensor([[-0.3], [0.0]], dtype=torch.double))
    w, s = update.step(w, grads, mask, s, features)
    torch.testing.assert_close(w[0], torch.tensor([[0.59], [4.0]], dtype=torch.double))
    torch.testing.assert_close(s[0], torch.tensor([[-0.45], [0.0]], dtype=torch.double))
    w[0].sum().backward()
    assert update.raw_lr.grad.abs() > 0
    assert update.raw_momentum.grad.abs() > 0
    assert update.raw_decay.grad.abs() > 0


def test_disabled_momentum_and_decay_reduce_to_gd():
    gd = GradientDescent(0.1)
    update = MomentumDecay(0.1, 3, momentum=False, forgetting=False, data_dependent=False)
    weights, grads = (torch.randn(2, 3, 4),), (torch.randn(2, 3, 4),)
    mask = torch.tensor([True, False])
    for _ in range(3):
        expected = gd(weights, grads, mask)
        actual, state = update.step(weights, grads, mask, features=torch.zeros(2, 3))
        torch.testing.assert_close(actual, expected)
        weights = actual


@pytest.mark.parametrize("rule", ["gradient_descent", "momentum_decay"])
def test_token_masks_reset_detach_and_outer_gradients(rule):
    memory = AssociativeMemory(8, 4, 6, update_rule=rule).double()
    x = torch.randn(2, 4, 8, dtype=torch.double, requires_grad=True)
    mask = torch.tensor([[True, False, True, False], [False] * 4])
    initial = memory.initial_state(2)
    updated = memory.write(x, initial, token_mask=mask)
    changed = x.detach().clone().masked_fill(~mask[:, :, None], float("nan"))
    other = memory.write(changed, initial, token_mask=mask)
    torch.testing.assert_close(updated.weights, other.weights)
    for old, new in zip(initial.weights, updated.weights, strict=True):
        torch.testing.assert_close(old[1], new[1])
    assert updated.update_count.tolist() == [1, 0]
    y = memory.read(x, updated)
    y[0].square().mean().backward()
    assert memory.fast.w1.grad.abs().sum() > 0
    assert memory.qkv.weight.grad.abs().sum() > 0
    assert x.grad[0, mask[0]].abs().sum() > 0
    detached = updated.detach()
    if detached.auxiliary is not None:
        assert all(s.grad_fn is None for s in detached.auxiliary)
    reset = memory.reset_rows(detached, torch.tensor([True, False]))
    torch.testing.assert_close(reset.weights, initial.weights)
    if reset.auxiliary is not None:
        assert all(s.count_nonzero() == 0 for s in reset.auxiliary)


def test_read_is_pure_and_no_grad_writes_have_no_graph():
    m = AssociativeMemory(8, 4, 6, update_rule="momentum_decay")
    x = torch.randn(2, 3, 8)
    before = {k: v.clone() for k, v in m.state_dict().items()}
    with torch.no_grad():
        state = m.write(x)
        for _ in range(3):
            m.read(x, state)
    assert state.update_count.tolist() == [1, 1]
    assert all(v.grad_fn is None for v in (*state.weights, *state.auxiliary))
    for k, v in m.state_dict().items():
        torch.testing.assert_close(v, before[k])


def test_masked_rows_preserve_nonzero_momentum_and_coefficients_receive_gradients():
    m = AssociativeMemory(8, 4, 6, update_rule="momentum_decay")
    x = torch.randn(2, 4, 8)
    s = m.write(x)
    before = tuple(v.clone() for v in s.auxiliary)
    updated = m.write(x + 1, s, write_mask=torch.tensor([True, False]))
    for a, b in zip(before, updated.auxiliary, strict=True):
        torch.testing.assert_close(a[1], b[1])
    for a, b in zip(s.weights, updated.weights, strict=True):
        torch.testing.assert_close(a[1], b[1])
    m.read(x, updated).square().mean().backward()
    assert m.update.coefficients.weight.grad.abs().sum() > 0
