"""Architecture contracts: pre-attention memory vs parallel gating vs context/writeback."""

from dataclasses import replace
from importlib import import_module

import torch

from lerobot_policy_vla_ttt.models.titans.attention import attend as real_attend
from lerobot_policy_vla_ttt.models.titans.language import language_step


def test_attention_inputs_and_write_sources_distinguish_architectures(setup, monkeypatch):
    p = setup.policy_class(setup.config)
    e = p.model.embedder
    layer = e.model.language_model.layers[0]
    connection = e.connections["0"]
    x = torch.randn(2, 4, 16)
    valid = torch.ones(2, 4, dtype=torch.bool)
    visual = valid.clone()
    rotary = e.model.language_model.rotary_emb(x, torch.arange(4)[None])
    module = import_module(f"lerobot_policy_vla_ttt.models.{p.name}.connection")
    observed = {}
    write = connection.memory.write

    def record_write(value, *args, **kwargs):
        observed["write"] = value.detach().clone()
        return write(value, *args, **kwargs)

    def record_attend(block, value, *args, **kwargs):
        observed["attention_input"] = value.detach().clone()
        observed["prefix"] = kwargs["prefix"].detach().clone()
        output = real_attend(block, value, *args, **kwargs)
        observed["attention_output"] = output.detach().clone()
        return output

    monkeypatch.setattr(connection.memory, "write", record_write)
    monkeypatch.setattr(module, "attend", record_attend)
    _, state = connection(layer, x, valid, visual, rotary, None, None, None)
    normalized = layer.input_layernorm(x)
    if p.name == "mac":
        torch.testing.assert_close(observed["write"], observed["attention_output"])
        torch.testing.assert_close(observed["attention_input"], normalized)
        assert observed["prefix"].shape[1] == p.config.num_persistent_tokens + x.shape[1]
    elif p.name == "mag":
        torch.testing.assert_close(observed["write"], normalized)
        torch.testing.assert_close(observed["attention_input"], normalized)
        assert observed["prefix"].shape[1] == p.config.num_persistent_tokens
    else:
        torch.testing.assert_close(observed["write"], normalized)
        retrieved = connection.memory.read(
            normalized, state, position=torch.zeros(2, dtype=torch.long)
        )
        torch.testing.assert_close(observed["attention_input"], retrieved)
        assert not torch.allclose(observed["attention_input"], normalized + retrieved)


def test_joint_gradients_through_unmodified_lower_layer_and_leading_padding(setup):
    cfg = replace(
        setup.config,
        memory_layers=(1,),
        memory_training="joint",
        training_stage="stage2",
        enable_gradient_checkpointing=True,
        memory_update_rule="momentum_decay",
    )
    p = setup.policy_class(cfg)
    e = p.model.embedder
    x = torch.randn(2, 4, 16, requires_grad=True)
    valid = torch.tensor([[False, False, True, True], [True] * 4])
    y, state = language_step(e.model.language_model, e.connections, cfg, x, valid, valid)
    y.square().mean().backward()
    assert y.isfinite().all() and x.grad.isfinite().all()
    assert x.grad[~valid].count_nonzero() == 0
    weight = e.model.language_model.layers[0].self_attn.q_proj.weight
    assert weight.grad.isfinite().all() and weight.grad.abs().sum() > 0
    assert state[0].update_count.tolist() == [1, 1]
