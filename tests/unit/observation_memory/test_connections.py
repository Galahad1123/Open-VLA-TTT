import pytest
import torch

from lerobot_policy_vla_ttt.models.aqmem.action_head import AQMemActionHead
from lerobot_policy_vla_ttt.models.aqmem.configuration_aqmem import AQMemConfig
from lerobot_policy_vla_ttt.models.evo1_groot.action_head import GrootActionHead
from lerobot_policy_vla_ttt.models.t2mem.action_head import T2MemActionHead
from lerobot_policy_vla_ttt.models.t2mem.configuration_t2mem import T2MemConfig

KINDS = [(AQMemConfig, AQMemActionHead), (T2MemConfig, T2MemActionHead)]


def config(cls, **kwargs):
    return cls(
        **(
            dict(
                device="cpu",
                dtype="float32",
                embed_dim=16,
                hidden_dim=24,
                state_hidden_dim=24,
                num_heads=4,
                num_layers=4,
                vlm_num_layers=4,
                chunk_size=3,
                n_action_steps=2,
                max_state_dim=4,
                max_action_dim=4,
                memory_heads=2,
                memory_head_dim=4,
                memory_hidden_dim=6,
                num_interface_tokens=3,
                num_inference_timesteps=2,
                enable_gradient_checkpointing=False,
                use_flash_attn=False,
            )
            | kwargs
        )
    )


@pytest.mark.parametrize("cls,head_cls", KINDS)
def test_read_write_separation_and_full_baseline_bypass(cls, head_cls):
    torch.manual_seed(2)
    cfg = config(cls)
    head = head_cls(cfg).eval()
    base = GrootActionHead(cfg).eval()
    head.initialize_from_baseline(base)
    context, state = torch.randn(2, 5, 16), torch.randn(2, 4)
    features = tuple(torch.randn_like(context) for _ in cfg.vlm_layer_indices)
    mask = torch.tensor([[1, 1, 1, 0, 0]] * 2).bool()
    cache = head.prepare_observation(features, mask)
    assert all(s.update_count.tolist() == [0, 0] for s in cache.memory)
    torch.manual_seed(8)
    expected = base.get_action(context, state, context_mask=mask)
    head.config.memory_enabled = False
    torch.manual_seed(8)
    actual = head.sample(context, state, cache, context_mask=mask)
    torch.testing.assert_close(actual.flatten(1), expected)
    head.config.memory_enabled = True
    for _ in range(3):
        head.sample(context, state, cache, context_mask=mask)
    assert all(s.update_count.tolist() == [0, 0] for s in cache.memory)
    updated = head.commit(cache)
    assert all(s.update_count.tolist() == [1, 1] for s in updated)
    altered = list(features)
    altered[0] = altered[0].clone()
    altered[0][:, 3:] = 10000
    same = head.prepare_observation(altered, mask)
    for x, y in zip(cache.summaries, same.summaries, strict=True):
        torch.testing.assert_close(x, y)


@pytest.mark.parametrize("cls,head_cls", KINDS)
def test_labels_never_enter_writes_and_history_gradient_reaches_w0(cls, head_cls):
    head = head_cls(config(cls))
    context = torch.randn(2, 5, 16)
    features = [context + i for i in range(len(head.config.vlm_layer_indices))]
    valid = torch.ones(2, 5, dtype=torch.bool)
    cache = head.prepare_observation(features, valid)
    memory = head.commit(cache)
    cache2 = head.prepare_observation(features, valid, memory)
    actions = torch.randn(2, 3, 4)
    pred, noise = head.training_prediction(
        context, torch.randn(2, 4), actions, cache2, context_mask=valid
    )
    (pred - (actions - noise)).square().mean().backward()
    assert head.memories[0].fast.w1.grad.abs().sum() > 0
    assert head.memories[0].k.weight.grad.abs().sum() > 0


def test_aqmem_queries_attention_output_before_residual(monkeypatch):
    head = AQMemActionHead(config(AQMemConfig)).eval()
    block = head.blocks[0]
    tokens = torch.ones(2, 4, 16)
    query = torch.randn_like(tokens)
    context = torch.randn(2, 5, 16)
    valid = torch.ones(2, 5, dtype=torch.bool)
    cache = head.prepare_observation(tuple(context for _ in head.interfaces), valid)
    captured = {}

    def record_attention(module, args, output):
        captured["attention"] = output[0]

    handle = block.attention.register_forward_hook(record_attention)

    def read(x, state):
        captured["query"] = x
        return torch.ones_like(x) * 2

    monkeypatch.setattr(head.memories[0], "read", read)
    output = head.connect_attention(0, block, tokens, query, context, valid, cache)
    handle.remove()
    torch.testing.assert_close(captured["query"], captured["attention"])
    torch.testing.assert_close(
        output, tokens + captured["attention"] + head.action_gates[0].tanh() * 2
    )


def test_aqmem_zero_gate_preserves_baseline_after_adaptive_writes():
    from torch.nn.attention import SDPBackend, sdpa_kernel

    torch.manual_seed(2)
    cfg = config(AQMemConfig, memory_gate_init=0.0, memory_update_rule="t2mem_adaptive")
    head, base = AQMemActionHead(cfg).eval(), GrootActionHead(cfg).eval()
    head.initialize_from_baseline(base)
    context, state = torch.randn(2, 5, 16), torch.randn(2, 4)
    mask = torch.tensor([[1, 1, 1, 0, 0]] * 2).bool()
    features = tuple(torch.randn_like(context) for _ in head.interfaces)
    memory = None
    for _ in range(3):
        memory = head.commit(head.prepare_observation(features, mask, memory))
    cache = head.prepare_observation(features, mask, memory)
    assert head.config.memory_enabled
    assert all(s.update_count.tolist() == [3, 3] for s in cache.memory)
    actions, time = torch.randn(2, 3, 4), torch.rand(2)
    ids = torch.zeros(2, dtype=torch.long)
    with sdpa_kernel(SDPBackend.MATH):
        expected = base.velocity(
            context, state, actions, time, context_mask=mask, embodiment_id=ids
        )
        actual = head.velocity_with_memory(
            context, state, actions, time, cache, context_mask=mask, embodiment_id=ids
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.square().mean().backward()
        assert any(g.grad is not None and g.grad.abs().sum() > 0 for g in head.action_gates)
        torch.manual_seed(8)
        expected_actions = base.get_action(context, state, context_mask=mask)
        torch.manual_seed(8)
        actual_actions = head.sample(context, state, cache, context_mask=mask)
        torch.testing.assert_close(actual_actions.flatten(1), expected_actions, rtol=0, atol=0)
