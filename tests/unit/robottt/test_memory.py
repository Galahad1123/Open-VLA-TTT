import pytest
import torch

from lerobot_policy_vla_ttt.ttt.functional import mlp
from lerobot_policy_vla_ttt.ttt.memory import AssociativeMemory


def test_update_matches_manual_gradient_and_outer_gradient():
    torch.manual_seed(1)
    memory = AssociativeMemory(8, 4, 6).double()
    x = torch.randn(2, 3, 8, dtype=torch.double, requires_grad=True)
    state = memory.initial_state(2)
    q, k, v = memory.project(x, state.position)
    loss = (mlp(state.weights, k) - v).square().mean((1, 2)).sum()
    grad = torch.autograd.grad(loss, state.weights, create_graph=True)
    expected = tuple(
        w - memory.update.learning_rate * g for w, g in zip(state.weights, grad, strict=True)
    )
    y, updated = memory(x, state)
    for actual, wanted in zip(updated.weights, expected, strict=True):
        torch.testing.assert_close(actual, wanted)
    torch.testing.assert_close(y, x + memory.gate.tanh() * memory.output(mlp(expected, q)))
    y.square().mean().backward()
    assert memory.fast.w1.grad.abs().sum() > 0
    assert memory.qkv.weight.grad.abs().sum() > 0
    assert memory.update.raw_lr.grad.abs().sum() > 0
    assert x.grad.abs().sum() > 0


def test_mask_batch_isolation_detach_and_reset():
    memory = AssociativeMemory(8, 4, 6)
    x = torch.randn(2, 3, 8)
    initial = memory.initial_state(2)
    _, state = memory(x, initial, write_mask=torch.tensor([True, False]))
    assert state.position.tolist() == [1, 0]
    for old, new in zip(initial.weights, state.weights, strict=True):
        torch.testing.assert_close(old[1], new[1])
    altered = x.clone()
    altered[1] += 10
    _, other = memory(altered, initial)
    for a, b in zip(state.weights, other.weights, strict=True):
        torch.testing.assert_close(a[0], b[0])
    detached = state.detach()
    assert all(w.grad_fn is None for w in detached.weights)
    reset = memory.reset_rows(detached, torch.tensor([True, False]))
    for a, b in zip(reset.weights, initial.weights, strict=True):
        torch.testing.assert_close(a[0], b[0])
    assert reset.position.tolist() == [0, 0]


def test_no_grad_updates_without_changing_parameters():
    memory = AssociativeMemory(8, 4, 6).eval()
    before = {k: v.clone() for k, v in memory.state_dict().items()}
    with torch.no_grad():
        _, state = memory(torch.randn(1, 3, 8))
    assert state.position.item() == 1
    assert all(w.grad_fn is None for w in state.weights)
    assert any(
        not torch.equal(a, b)
        for a, b in zip(state.weights, memory.initial_state(1).weights, strict=True)
    )
    for key, value in memory.state_dict().items():
        torch.testing.assert_close(value, before[key])
    with torch.inference_mode(), pytest.raises(RuntimeError, match="inference_mode"):
        memory(torch.randn(1, 3, 8))
