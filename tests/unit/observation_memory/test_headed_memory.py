import pytest
import torch

from lerobot_policy_vla_ttt.ttt.headed_memory import HeadedMemory


@pytest.mark.parametrize("rule", ["gradient_descent", "momentum_decay", "t2mem_adaptive"])
def test_update_is_independent_per_row_and_read_is_pure(rule):
    torch.manual_seed(3)
    memory = HeadedMemory(8, 2, 4, 6, update_rule=rule)
    x = torch.randn(2, 3, 8, requires_grad=True)
    initial = memory.initial_state(2)
    before = memory.read(x, initial)
    result = memory.write(x, initial, write_mask=torch.tensor([True, False]))
    assert result.update_count.tolist() == [1, 0]
    for a, b in zip(initial.weights, result.weights, strict=True):
        torch.testing.assert_close(a[1], b[1])
    torch.testing.assert_close(before, memory.read(x, initial))
    output = memory.read(x, result)
    output.square().mean().backward()
    assert memory.fast.w1.grad.abs().sum() > 0
    assert x.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in memory.parameters() if p.grad is not None)
    with torch.no_grad():
        inference = memory.write(x.detach(), result.detach())
    assert all(w.grad_fn is None for w in inference.weights)


def test_adaptive_step_matches_per_head_quadratic_reference():
    from lerobot_policy_vla_ttt.ttt.updates.t2mem_adaptive import T2MemAdaptive

    update = T2MemAdaptive().double()
    w = (torch.tensor([[[[0.2]], [[-0.3]]]], dtype=torch.double, requires_grad=True),)
    initial = (w[0].detach().clone(),)
    keys = torch.tensor([[[[1.0]], [[2.0]]]], dtype=torch.double)
    values = torch.tensor([[[[1.0]], [[1.0]]]], dtype=torch.double)

    def forward(weights, x):
        return x @ weights[0]

    # Zero displacement makes beta=1 because normalized residual=1 for head 2,
    # head 1 residual .8 gives smoothstep(.9); calculate both explicitly.
    per_head_loss = (forward(w, keys) - values).square().mean((-1, -2)) / 2
    grad = torch.autograd.grad(per_head_loss.sum(), w, create_graph=True)
    actual, _ = update.step(
        w, grad, torch.tensor([True]), initial=initial, keys=keys, values=values, forward=forward
    )
    residual = (forward(w, keys) - values).abs().flatten(2).mean(2).clamp(0, 1)
    # Zero displacement -> novelty=1.
    z = (1 + residual) / 2
    beta = 0.1 + 0.9 * (3 * z * z - 2 * z * z * z)
    # Hessian of per-head loss is k^2; calibrated eta=.5/k^2.
    expected = (
        w[0] - (0.5 * beta / keys.squeeze(-1).squeeze(-1).square())[:, :, None, None] * grad[0]
    )
    torch.testing.assert_close(actual[0], expected)
