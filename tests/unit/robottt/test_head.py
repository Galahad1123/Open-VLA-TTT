import pytest
import torch

from lerobot_policy_vla_ttt.models.evo1_groot.action_head import GrootActionHead
from lerobot_policy_vla_ttt.models.robottt.action_head import RoboTTTActionHead
from lerobot_policy_vla_ttt.models.robottt.configuration_robottt import RoboTTTConfig


def tiny_config(**kwargs):
    return RoboTTTConfig(
        **(
            dict(
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
                memory_dim=4,
                memory_hidden_dim=6,
                num_register_tokens=2,
                use_flash_attn=False,
            )
            | kwargs
        )
    )


def inputs():
    return torch.randn(2, 5, 16), torch.randn(2, 4), torch.randn(2, 3, 4)


def test_backbone_import_and_exact_bypass():
    cfg = tiny_config()
    baseline = GrootActionHead(tiny_config(num_register_tokens=0)).eval()
    head = RoboTTTActionHead(cfg).eval()
    report = head.initialize_from_baseline(baseline)
    assert "tokens.registers" in report["new"]
    context, state, actions = inputs()
    time = torch.rand(2)
    kwargs = dict(
        context_mask=torch.ones(2, 5, dtype=torch.bool),
        embodiment_id=torch.zeros(2, dtype=torch.long),
    )
    expected = baseline.velocity(context, state, actions, time, **kwargs)
    actual, memory = head.velocity_step(
        context, state, actions, time, memory_enabled=False, **kwargs
    )
    torch.testing.assert_close(actual, expected)
    assert memory is None


def test_temporal_gradients_detach_and_causal_prefix():
    head = RoboTTTActionHead(tiny_config())
    context, state, actions = inputs()
    context.requires_grad_()
    p1, _, memory = head.training_step(context, state, actions)
    p2, _, next_memory = head.training_step(context + 1, state, actions, memory=memory)
    p2.square().mean().backward()
    assert context.grad.abs().sum() > 0
    assert head.memories[0].fast.w1.grad.abs().sum() > 0
    assert all(m.position.tolist() == [2, 2] for m in next_memory)
    before = p1.detach().clone()
    head.training_step(context + 100, state, actions, memory=memory)
    torch.testing.assert_close(p1.detach(), before)


def test_cumulative_sampling_reset_and_state_not_checkpointed():
    head = RoboTTTActionHead(tiny_config()).eval()
    context, state, _ = inputs()
    keys = set(head.state_dict())
    torch.manual_seed(1)
    a = head.get_action(context, state)
    assert all(s.position.tolist() == [2, 2] for s in head.inference_state)
    head.get_action(context, state)
    assert all(s.position.tolist() == [4, 4] for s in head.inference_state)
    assert set(head.state_dict()) == keys
    head.reset()
    torch.manual_seed(1)
    torch.testing.assert_close(a, head.get_action(context, state))
    with pytest.raises(ValueError, match="batch"):
        head.get_action(context[:1], state[:1])


def test_later_positions_cannot_change_earlier_outputs_but_history_matters():
    torch.manual_seed(4)
    head = RoboTTTActionHead(tiny_config()).eval()
    for module in head.memories:
        module.gate.data.fill_(0.3)
    context, state, actions = inputs()
    kwargs = dict(
        context_mask=torch.ones(2, 5, dtype=torch.bool),
        embodiment_id=torch.zeros(2, dtype=torch.long),
    )
    time = torch.full((2,), 0.5)

    def rollout(contexts, chunks):
        memory, outputs = None, []
        for c, chunk in zip(contexts, chunks, strict=True):
            pred, memory = head.velocity_step(c, state, chunk, time, memory=memory, **kwargs)
            outputs.append(pred.detach())
        return outputs

    reference = rollout([context, context + 1, context + 2], [actions] * 3)
    changed_future = rollout([context, context + 1, context + 100], [actions] * 3)
    torch.testing.assert_close(reference[0], changed_future[0])
    torch.testing.assert_close(reference[1], changed_future[1])
    changed_history = rollout([context + 10, context + 1, context + 2], [actions] * 3)
    assert (reference[1] - changed_history[1]).abs().max() > 1e-6
    # Diagnostic, NOT an expected invariance: earlier true chunks can affect memory.
    changed_labels = rollout([context, context + 1, context + 2], [actions + 10, actions, actions])
    assert (reference[1] - changed_labels[1]).abs().max() > 1e-6


def test_detach_cuts_w0_gradient_without_resetting_values():
    head = RoboTTTActionHead(tiny_config())
    context, state, actions = inputs()
    _, _, memory = head.training_step(context, state, actions)
    detached = tuple(s.detach() for s in memory)
    pred, _, updated = head.training_step(context, state, actions, memory=detached)
    pred.square().mean().backward()
    assert head.memories[0].fast.w1.grad is None
    assert head.memories[0].qkv.weight.grad.abs().sum() > 0
    assert updated[0].position.tolist() == [2, 2]
