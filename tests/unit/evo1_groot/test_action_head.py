import pytest
import torch
from lerobot.policies.factory import get_policy_class, make_policy_config

from lerobot_policy_vla_ttt.models.evo1_groot.action_head import GrootActionHead
from lerobot_policy_vla_ttt.models.evo1_groot.configuration_evo1_groot import Evo1GrootConfig


def tiny_config(**kwargs):
    values = dict(
        device="cpu",
        use_amp=False,
        embed_dim=16,
        hidden_dim=24,
        state_hidden_dim=24,
        num_heads=4,
        num_layers=2,
        chunk_size=3,
        n_action_steps=2,
        max_state_dim=4,
        max_action_dim=4,
        num_inference_timesteps=2,
        use_flash_attn=False,
    )
    return Evo1GrootConfig(**(values | kwargs))


def test_factory_keeps_three_baselines_distinct():
    config = make_policy_config("evo1_groot", device="cpu")
    assert type(config) is Evo1GrootConfig
    assert get_policy_class("evo1_groot").config_class is Evo1GrootConfig
    assert get_policy_class("evo1_groot") is not get_policy_class("vla_ttt")
    assert get_policy_class("evo1_groot") is not get_policy_class("evo1")


@pytest.mark.parametrize(
    "overrides",
    [
        dict(num_layers=1),
        dict(num_heads=3),
        dict(num_inference_timesteps=0),
        dict(num_register_tokens=-1),
        dict(flow_beta_alpha=0),
    ],
)
def test_invalid_architecture_fails_before_allocating_vlm(overrides):
    with pytest.raises(ValueError):
        tiny_config(**overrides)


def test_training_masks_padding_and_backpropagates_to_context_and_state():
    torch.manual_seed(0)
    head = GrootActionHead(tiny_config())
    context = torch.randn(2, 5, 16, requires_grad=True)
    state = torch.randn(2, 4, requires_grad=True)
    actions = torch.randn(2, 3, 4)
    mask = torch.ones_like(actions, dtype=torch.bool)
    mask[:, :, -1] = False
    mask[:, -1] = False
    velocity, noise = head(context, state, actions, action_mask=mask)
    assert velocity.shape == (2, 12)
    assert noise.shape == actions.shape
    assert torch.count_nonzero(velocity.reshape_as(actions)[~mask]) == 0
    assert torch.count_nonzero(noise[~mask]) == 0
    loss = ((velocity.reshape_as(actions) - (actions - noise)) ** 2)[mask].mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert context.grad.abs().sum() > 0
    assert state.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in head.parameters() if p.grad is not None)
    # Masked ground-truth values must not leak into valid velocity predictions.
    altered = actions.masked_fill(~mask, 10000)
    torch.manual_seed(23)
    first = head(context, state, actions, action_mask=mask)[0]
    torch.manual_seed(23)
    second = head(context, state, altered, action_mask=mask)[0]
    torch.testing.assert_close(first, second)


def test_sampling_respects_context_mask_batch_isolation_and_seed():
    torch.manual_seed(7)
    head = GrootActionHead(tiny_config(num_categories=2, num_register_tokens=2)).eval()
    context = torch.randn(2, 5, 16)
    state = torch.randn(2, 4)
    valid = torch.tensor([[True, True, False, False, False]] * 2)
    mask = torch.tensor([[True, True, False, False]] * 2)
    ids = torch.tensor([0, 1])

    def sample(c, s):
        torch.manual_seed(19)
        return head.get_action(c, s, action_mask=mask, context_mask=valid, embodiment_id=ids)

    first = sample(context, state)
    assert torch.isfinite(first).all()
    assert torch.count_nonzero(first.reshape(2, 3, 4)[:, :, 2:]) == 0
    altered = context.clone()
    altered[:, 2:] = 10000
    torch.testing.assert_close(first, sample(altered, state))
    altered[1] = -500
    torch.testing.assert_close(first[0], sample(altered, state)[0])
    assert not torch.allclose(first, sample(context, state + 2))
    assert not torch.allclose(first, sample(context + 2, state))
    with pytest.raises(ValueError, match="context"):
        head.get_action(context, state, context_mask=torch.zeros_like(valid))


def test_euler_integrates_velocity_in_noise_to_action_direction():
    head = GrootActionHead(tiny_config())
    # Constant vector field has the exact solution x(1)=x(0)+2.
    head.velocity = lambda context, state, actions, time, **kwargs: torch.full_like(actions, 2)
    context, state = torch.zeros(1, 5, 16), torch.zeros(1, 4)
    torch.manual_seed(12)
    expected = torch.randn(1, 3, 4) + 2
    torch.manual_seed(12)
    torch.testing.assert_close(head.get_action(context, state).reshape_as(expected), expected)
