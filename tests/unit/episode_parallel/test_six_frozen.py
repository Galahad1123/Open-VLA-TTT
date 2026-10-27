import pytest

from lerobot_policy_vla_ttt.models.evo1_groot.configuration_evo1_groot import Evo1GrootConfig
from lerobot_policy_vla_ttt.training.episode_parallel.config import Recipe
from lerobot_policy_vla_ttt.training.episode_parallel.data import Episode, MixedTaskEpisodeSampler
from lerobot_policy_vla_ttt.training.episode_parallel.robomme import select_task_subset
from lerobot_policy_vla_ttt.training.train_episode_parallel import resolve_config


def test_frozen_resolve_both_architectures():
    source = Evo1GrootConfig(chunk_size=20, n_action_steps=10)
    for arch, mode in [("evo1_groot", "policy"), ("aqmem", "joint")]:
        recipe = Recipe(horizon=20, freeze_vlm=True, training_mode=mode)
        config = resolve_config(source, arch, recipe, device="cpu")
        assert not config.finetune_language_model
        assert not config.finetune_vision_model
        assert config.finetune_action_head


def test_subset_uses_task_family_not_language_instruction():
    episodes = [Episode(i, f"color instruction {i}", i * 10, (i + 1) * 10) for i in range(12)]
    groups = {f"task{t}": [2 * t, 2 * t + 1] for t in range(6)}
    selected = select_task_subset(episodes, groups)
    sampler = MixedTaskEpisodeSampler(selected, seed=1000, global_batch_size=6)
    for step in range(5):
        sample = sampler.batch(step)
        assert len({selected[s.episode_id].task for s in sample}) == 6
    with pytest.raises(ValueError):
        select_task_subset(episodes, {"bad": [99]})
    with pytest.raises(ValueError):
        Recipe(task_groups={"a": [1], "b": [1]})


def test_official_augmentation_replays_without_consuming_flow_rng():
    import torch

    from lerobot_policy_vla_ttt.training.episode_parallel.data import EpisodeSource

    class Dataset:
        def __len__(self):
            return 20

        def __getitem__(self, index):
            return {
                "observation.images.image": torch.arange(3 * 256 * 256)
                .reshape(3, 256, 256)
                .to(torch.uint8),
                "action": torch.zeros(20, 8),
            }

    source = EpisodeSource(
        Dataset(),
        [Episode(0, "t", 0, 20)],
        execution=10,
        horizon=20,
        robomme_augmentation="official_v1",
    )
    state = torch.random.get_rng_state()
    a = source.decision(0, 0, seed=12)
    b = source.decision(0, 1, seed=12)
    assert torch.equal(state, torch.random.get_rng_state())
    assert torch.equal(a["observation.images.image"], b["observation.images.image"])


def test_delta_inference_delivers_every_execution_observation():
    from collections import deque
    from types import SimpleNamespace

    import numpy as np
    import torch

    from lerobot_policy_vla_ttt.evaluation.policy import Evo1Inference

    calls = []

    class Policy:
        config = SimpleNamespace(robomme_action_representation="delta_joint", n_action_steps=10)

        def predict_action_chunk(self, batch, *, env_step):
            calls.append(("decision", env_step))
            return torch.ones(1, 20, 8)

        def observe(self, batch, *, env_step):
            calls.append(("observe", env_step))

        def reset(self):
            calls.append(("reset", 0))

    class Identity:
        def __call__(self, x):
            return x

        def reset(self):
            pass

    engine = Evo1Inference.__new__(Evo1Inference)
    engine.policy = Policy()
    engine.pre = engine.post = Identity()
    engine._absolute_action_queue = deque()
    engine._delta_env_step = 0
    for step in range(21):
        obs = {
            "state": np.full(8, step),
            "task": "t",
            "images": {k: np.zeros((256, 256, 3), dtype=np.uint8) for k in ("image", "image2")},
        }
        result = engine.infer(obs)
        np.testing.assert_array_equal(result, [step // 10 * 10 + 1] * 7 + [1])
    assert calls == [("decision" if i % 10 == 0 else "observe", i) for i in range(21)]
    engine.reset(1)
    assert engine._delta_env_step == 0
    assert not engine._absolute_action_queue


def test_joint_optimizer_never_unfreezes_vlm(monkeypatch):
    import torch

    from lerobot_policy_vla_ttt.training.episode_parallel.engine import backward_episode
    from lerobot_policy_vla_ttt.training.episode_parallel.optim import PhaseOptimizers

    from .test_engine import episode_events, make_policy

    policy = make_policy("aqmem", monkeypatch)
    policy.config.finetune_vlm = False
    policy.config.finetune_language_model = False
    policy.config.finetune_vision_model = False
    recipe = Recipe(
        training_mode="joint", freeze_vlm=True, steps=2, global_batch_size=1, tbptt_events=2
    )
    control = PhaseOptimizers(policy, recipe)
    before = {n: p.detach().clone() for n, p in policy.model.embedder.named_parameters()}
    for step in range(2):
        control.begin(step)
        assert not any(p.requires_grad for p in policy.model.embedder.parameters())
        backward_episode(
            policy, episode_events(), decisions=3, recipe=recipe, backward=lambda x: x.backward()
        )
        assert any(p.grad is not None for p in policy.model.action_head.memory_parameters())
        control.step()
    for n, p in policy.model.embedder.named_parameters():
        assert torch.equal(p, before[n])


def test_paired_recipes_have_identical_budget_and_samples():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[3] / "configs/training/robomme-screen"
    a = json.loads((root / "multitask-six-frozen-baseline.json").read_text())
    b = json.loads((root / "multitask-six-frozen-aqmem.json").read_text())
    a.pop("training_mode")
    b.pop("training_mode")
    assert a == b
    assert sum(map(len, a["task_groups"].values())) == 266
    Recipe(**a, training_mode="policy")


def test_actual_half_execution_clock_with_delta_queue():
    from collections import deque
    from contextlib import nullcontext
    from types import SimpleNamespace

    import numpy as np
    import torch

    from lerobot_policy_vla_ttt.evaluation.policy import Evo1Inference
    from lerobot_policy_vla_ttt.models.observation_memory.lifecycle import HalfExecutionLifecycle

    writes = []

    class Policy(HalfExecutionLifecycle):
        config = SimpleNamespace(
            robomme_action_representation="delta_joint", n_action_steps=10, write_stride=5
        )

        def eval(self):
            return self

        def _maybe_autocast(self):
            return nullcontext()

        def _sample_candidate(self, batch):
            writes.append(("decision", int(batch["observation.state"][0, 0])))
            return torch.ones(1, 20, 8), None

        def _write_candidate(self, batch):
            writes.append(("write", int(batch["observation.state"][0, 0])))
            return None

        def reset(self):
            self._reset_observation_clock()

    class Identity:
        def __call__(self, value):
            return value

        def reset(self):
            pass

    engine = Evo1Inference.__new__(Evo1Inference)
    engine.policy = Policy()
    engine.pre = engine.post = Identity()
    engine._absolute_action_queue = deque()
    engine.metadata = {"demonstration_mode": "write_only_prefix"}
    engine.reset(1)

    def obs(i):
        return {
            "state": np.full(8, i),
            "task": "move",
            "images": {k: np.zeros((256, 256, 3), dtype=np.uint8) for k in ("image", "image2")},
        }

    engine.consume_demo(obs(99))
    for i in range(21):
        engine.infer(obs(i))
    assert writes == [
        ("write", 99),
        ("decision", 0),
        ("write", 5),
        ("decision", 10),
        ("write", 15),
        ("decision", 20),
    ]
    engine.reset(1)
    engine.consume_demo(obs(98))
    engine.infer(obs(0))
    assert writes[-2:] == [("write", 98), ("decision", 0)]
