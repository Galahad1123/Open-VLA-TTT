from dataclasses import fields

import pytest
import torch
from observation_memory.assets import patch_assets
from observation_memory.test_policy import policy_config
from torch import nn

from lerobot_policy_vla_ttt.training.episode_parallel.config import Recipe
from lerobot_policy_vla_ttt.training.episode_parallel.data import Event
from lerobot_policy_vla_ttt.training.episode_parallel.engine import backward_episode, episode_rng
from lerobot_policy_vla_ttt.training.episode_parallel.optim import PhaseOptimizers
from lerobot_policy_vla_ttt.training.train_robottt import ARCHITECTURES


class ScalarPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward_step(self, frame, memory, **kwargs):
        x = frame["action"]
        return (self.weight * x).square().sum(), torch.tensor(x.numel()), None


def test_equal_episode_and_chunk_weight_despite_different_lengths_and_counts():
    p = ScalarPolicy()
    r = Recipe(global_batch_size=2, tbptt_events=1)
    a = [Event({"action": torch.tensor([1.0])}, True)]
    b = [
        Event({"action": torch.tensor([2.0, 2.0, 2.0])}, True),
        Event({"action": torch.tensor([4.0])}, True),
    ]
    loss_a, _ = backward_episode(p, a, decisions=1, recipe=r, backward=lambda x: x.backward())
    loss_b, _ = backward_episode(p, b, decisions=2, recipe=r, backward=lambda x: x.backward())
    # mean(1, mean(4,16)) = 5.5; d/dw at w=1 equals 11.
    assert loss_a["loss_sum"] + loss_b["loss_sum"] == 11.0
    assert p.weight.grad.item() == 11.0


def make_policy(kind, monkeypatch):
    patch_assets(monkeypatch)
    cls, policy_cls = ARCHITECTURES[kind]
    options = dict(
        memory_training="joint",
        training_stage="stage2",
        dropout=0.0,
        memory_update_rule="t2mem_adaptive",
        enable_gradient_checkpointing=True,
    )
    if kind in ("mac", "mag", "mal"):
        options.update(
            memory_layers=(0, 1),
            memory_dim=4,
            memory_update_protocol="half_execution",
            num_persistent_tokens=2,
        )
    base = policy_config(ARCHITECTURES["t2mem"][0])
    if kind in ("aqmem", "t2mem"):
        options["vlm_layer_indices"] = None
    cfg = cls(
        **({f.name: getattr(base, f.name) for f in fields(cls) if hasattr(base, f.name)} | options)
    )
    return policy_cls(cfg).float()


def episode_events(length=5):
    # Data construction must not consume the episode's flow/dropout RNG.
    g = torch.Generator().manual_seed(7)
    return [
        Event(
            {
                "observation.state": torch.rand(1, 4, generator=g),
                "observation.images.image": torch.rand(1, 3, 16, 16, generator=g),
                **({"action": torch.rand(1, 3, 4, generator=g)} if i % 2 == 0 else {}),
                "task": ["move"],
            },
            i % 2 == 0,
        )
        for i in range(length)
    ]


@pytest.mark.parametrize("kind", ["t2mem", "aqmem", "mac", "mag", "mal"])
def test_real_policy_memory_gradients_phase_switch_and_detached_state(kind, monkeypatch):
    torch.manual_seed(12)
    p = make_policy(kind, monkeypatch)
    r = Recipe(
        steps=4,
        phase_length=1,
        warmup_steps=1,
        global_batch_size=1,
        horizon=3,
        execution=2,
        tbptt_events=4,
    )
    control = PhaseOptimizers(p, r)
    for step in range(4):
        phase = control.begin(step)
        before = {n: x.detach().clone() for n, x in p.named_parameters()}
        with episode_rng(25, torch.device("cpu")):
            metrics, memory = backward_episode(
                p, episode_events(), decisions=3, recipe=r, backward=lambda x: x.backward()
            )
        active = {n: x for n, x in p.named_parameters() if x.requires_grad}
        if phase == "memory":
            assert any(
                ".fast." in n and x.grad is not None and x.grad.abs().sum() > 0
                for n, x in active.items()
            )
            assert any(
                ".update.rho" in n and x.grad is not None and x.grad.abs().sum() > 0
                for n, x in active.items()
            )
        assert metrics["decisions"] == 3 and metrics["events"] == 5
        assert all(w.grad_fn is None for s in memory for w in s.weights)
        control.step()
        assert all(torch.equal(before[n], x) for n, x in p.named_parameters() if n not in active)
        assert any(not torch.equal(before[n], x) for n, x in active.items())
    assert control.counts == {"memory": 2, "policy": 2}
    assert len(control.optimizers["memory"].state) and len(control.optimizers["policy"].state)
    groups = control.optimizers["memory"].param_groups
    for group in groups:
        assert group["lr"] == pytest.approx(group["peak_lr"] * 0.1)
        if group["category"] in ("w0", "inner_step", "t2mem_gate"):
            assert group["weight_decay"] == 0
        if group["category"] == "t2mem_gate":
            assert kind == "t2mem" and group["peak_lr"] == 5e-4


def test_episode_rng_isolated_and_repeatable():
    state = torch.get_rng_state().clone()
    with episode_rng(10, torch.device("cpu")):
        a = torch.rand(4)
    with episode_rng(10, torch.device("cpu")):
        b = torch.rand(4)
    assert torch.equal(a, b) and torch.equal(state, torch.get_rng_state())


def test_aqmem_joint_updates_all_components_once(monkeypatch):
    torch.manual_seed(12)
    policy = make_policy("aqmem", monkeypatch)
    recipe = Recipe(
        steps=3,
        warmup_steps=2,
        global_batch_size=1,
        horizon=3,
        execution=2,
        tbptt_events=4,
        training_mode="joint",
    )
    control = PhaseOptimizers(policy, recipe)
    assert set(control.optimizers) == {"joint"}
    groups = control.optimizers["joint"].param_groups
    ids = [id(p) for group in groups for p in group["params"]]
    assert len(ids) == len(set(ids))
    for step, factor in enumerate((0.5, 1.0, 0.1)):
        assert control.begin(step) == "joint"
        assert set(ids) == {id(p) for p in policy.parameters() if p.requires_grad}
        before = {n: p.detach().clone() for n, p in policy.named_parameters()}
        with episode_rng(25, torch.device("cpu")):
            backward_episode(
                policy,
                episode_events(),
                decisions=3,
                recipe=recipe,
                backward=lambda loss: loss.backward(),
            )
        for fragment in (
            "vision_tower",
            "language_model",
            "multi_modal_projector",
            "blocks.",
            ".fast.",
            ".update.rho",
            "action_gates",
        ):
            candidates = [(n, p) for n, p in policy.named_parameters() if fragment in n]
            assert candidates, fragment
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for _, p in candidates), (
                fragment
            )
        control.step()
        for fragment in (
            "vision_tower",
            "language_model",
            "multi_modal_projector",
            "blocks.",
            ".fast.",
            ".update.rho",
            "action_gates",
        ):
            assert any(
                not torch.equal(before[n], p) for n, p in policy.named_parameters() if fragment in n
            ), fragment
        assert control.counts == {"joint": step + 1}
        for group in groups:
            peak = (
                recipe.policy_lr
                if group["category"] in ("vlm", "action_expert")
                else recipe.memory_lr
            )
            assert group["peak_lr"] == peak
            assert group["lr"] == pytest.approx(peak * factor)
            if group["category"] in ("w0", "inner_step"):
                assert group["weight_decay"] == 0
        assert all(
            state["step"] == step + 1 for state in control.optimizers["joint"].state.values()
        )
