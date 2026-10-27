import pytest
import torch

from lerobot_policy_vla_ttt.training.episode_parallel.config import Recipe
from lerobot_policy_vla_ttt.training.episode_parallel.microbatch import backward_microbatch

from .test_engine import episode_events, make_policy


def frozen(policy):
    policy.config.finetune_language_model = False
    policy.config.finetune_vision_model = False
    policy.config.finetune_vlm = False
    if policy.config.type == "aqmem":
        policy.model.set_phase("joint")
    else:
        policy.model.set_finetune_flags()
        policy.train()
    return policy


@pytest.mark.parametrize("preprocessing", [False, True])
@pytest.mark.parametrize("kind", ["aqmem", "evo1_groot"])
@pytest.mark.parametrize("size", [2, 3])
def test_static_microbatch_matches_serial_gradients_and_memory(
    monkeypatch, size, kind, preprocessing
):
    from .test_baseline import make_baseline

    def make():
        if preprocessing:
            from dataclasses import fields

            from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
            from observation_memory.assets import patch_assets
            from observation_memory.test_policy import policy_config

            from lerobot_policy_vla_ttt.models.aqmem.configuration_aqmem import AQMemConfig
            from lerobot_policy_vla_ttt.models.aqmem.modeling_aqmem import AQMemPolicy
            from lerobot_policy_vla_ttt.models.evo1_groot.configuration_evo1_groot import (
                Evo1GrootConfig,
            )
            from lerobot_policy_vla_ttt.models.evo1_groot.modeling_evo1_groot import Evo1GrootPolicy

            patch_assets(monkeypatch)
            cfg = policy_config(
                AQMemConfig,
                max_action_dim=16,
                max_state_dim=16,
                robomme_action_representation="delta_joint",
                postprocess_action_dim=8,
                binarize_gripper=False,
                memory_update_rule="t2mem_adaptive",
                input_features={
                    "observation.state": PolicyFeature(FeatureType.STATE, (8,)),
                    "observation.images.image": PolicyFeature(FeatureType.VISUAL, (3, 16, 16)),
                },
                output_features={"action": PolicyFeature(FeatureType.ACTION, (8,))},
                normalization_mapping={
                    "STATE": NormalizationMode.QUANTILES,
                    "ACTION": NormalizationMode.QUANTILES,
                },
            )
            if kind == "evo1_groot":
                cfg = Evo1GrootConfig(
                    **{f.name: getattr(cfg, f.name) for f in fields(Evo1GrootConfig)}
                )
            return frozen((AQMemPolicy if kind == "aqmem" else Evo1GrootPolicy)(cfg).float())
        return frozen(
            make_policy(kind, monkeypatch) if kind == "aqmem" else make_baseline(monkeypatch)
        )

    torch.manual_seed(91)
    serial = make()
    batch = make()
    batch.load_state_dict(serial.state_dict())
    pre = None
    if preprocessing:
        from lerobot_policy_vla_ttt.models.evo1_groot.processor_evo1_groot import (
            make_evo1_groot_pre_post_processors,
        )

        stats = {
            k: {"q01": torch.full((8,), -2.0), "q99": torch.full((8,), 4.0)}
            for k in ("action", "observation.state")
        }
        pre, _ = make_evo1_groot_pre_post_processors(serial.config, stats)
    streams = [episode_events(3 + 2 * i) for i in range(size)]
    if preprocessing:
        for stream in streams:
            for event in stream:
                event.frame["observation.state"] = event.frame["observation.state"].repeat(1, 2)
                if "action" in event.frame:
                    event.frame["action"] = event.frame["action"].repeat(1, 1, 2)
    # Exercise different supervision starts as well as trailing padding.
    from lerobot_policy_vla_ttt.training.episode_parallel.data import Event

    streams[1][0] = Event({k: v for k, v in streams[1][0].frame.items() if k != "action"}, False)
    streams[0][-1].frame["action_is_pad"] = torch.tensor([[False, True, True]])
    counts = [sum(e.supervised for e in x) for x in streams]
    recipe = Recipe(global_batch_size=6, tbptt_events=2)
    states = []
    metrics = []
    for i in range(size):
        m, s = backward_microbatch(
            serial,
            [iter(streams[i])],
            decisions=[counts[i]],
            seeds=[100 + i],
            recipe=recipe,
            preprocess=pre,
            backward=lambda x: x.backward(),
        )
        states.append(s)
        metrics.append(m)
    m, s = backward_microbatch(
        batch,
        [iter(x) for x in streams],
        decisions=counts,
        seeds=list(range(100, 100 + size)),
        recipe=recipe,
        preprocess=pre,
        backward=lambda x: x.backward(),
    )
    assert m["loss_sum"] == pytest.approx(sum(x["loss_sum"] for x in metrics), rel=2e-4)
    assert m["decisions"] == sum(counts)
    assert m["events"] == sum(map(len, streams))
    for (name, a), (name2, b) in zip(
        serial.named_parameters(), batch.named_parameters(), strict=True
    ):
        assert name == name2
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, rtol=4e-3, atol=2e-5, msg=name)
    if s is None:
        assert all(x is None for x in states)
        return
    for layer, state in enumerate(s):
        for i in range(size):
            assert state.update_count[i] == states[i][layer].update_count[0]
            assert state.decision_index[i] == states[i][layer].decision_index[0]
            assert state.position[i] == states[i][layer].position[0]
            for weight, reference in zip(state.weights, states[i][layer].weights, strict=True):
                torch.testing.assert_close(weight[i], reference[0], rtol=4e-3, atol=2e-5)


def test_flow_draws_are_independent_of_batch_and_do_not_consume_outer_rng():
    from types import SimpleNamespace

    from lerobot_policy_vla_ttt.training.flow_matching import flow_row_seeds, sample_flow

    config = SimpleNamespace(flow_beta_alpha=1.5, flow_beta_beta=1.0, flow_time_scale=0.999)
    actions = torch.zeros(3, 4, 8)
    mask = torch.ones_like(actions, dtype=torch.bool)
    rng = torch.random.get_rng_state()
    with flow_row_seeds([2, 3, 4]):
        joint = sample_flow(actions, mask, config)
    for row, seed in enumerate([2, 3, 4]):
        with flow_row_seeds([seed]):
            separate = sample_flow(actions[row : row + 1], mask[row : row + 1], config)
        for a, b in zip(joint, separate, strict=True):
            torch.testing.assert_close(a[row : row + 1], b, rtol=0, atol=0)
    assert len(torch.unique(joint[2])) == 3
    assert torch.equal(torch.random.get_rng_state(), rng)


@pytest.mark.parametrize("kind", ["aqmem", "evo1_groot"])
def test_original_nonzero_dropout_batch_is_finite(monkeypatch, kind):
    from .test_baseline import make_baseline

    policy = frozen(
        make_policy("aqmem", monkeypatch) if kind == "aqmem" else make_baseline(monkeypatch)
    )
    policy.config.dropout = 0.2
    for module in policy.model.action_head.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.2
        if isinstance(module, torch.nn.MultiheadAttention):
            module.dropout = 0.2
    streams = [episode_events(5), episode_events(3)]
    metrics, _ = backward_microbatch(
        policy,
        [iter(x) for x in streams],
        decisions=[3, 2],
        seeds=[10, 11],
        recipe=Recipe(global_batch_size=6, tbptt_events=2),
        backward=lambda x: x.backward(),
    )
    assert torch.isfinite(torch.tensor(metrics["loss_sum"]))
    assert all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None)
    assert not any(p.grad is not None for p in policy.model.embedder.parameters())
