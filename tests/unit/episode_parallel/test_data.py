import pytest
import torch

from lerobot_policy_vla_ttt.training.episode_parallel.config import Recipe
from lerobot_policy_vla_ttt.training.episode_parallel.data import (
    Episode,
    EpisodeSampler,
    EpisodeSource,
    temporal_transform,
)


def test_global_batch_is_same_task_replayable_and_sharded_by_slot():
    episodes = [Episode(0, "a", 0, 4), Episode(1, "a", 4, 104), Episode(2, "b", 104, 108)]
    sampler = EpisodeSampler(episodes, seed=10, global_batch_size=4)
    for step in range(8):
        slots = sampler.batch(step)
        assert len(slots) == 4
        assert len({episodes[s.episode_id].task for s in slots}) == 1
        assert slots == EpisodeSampler(episodes, seed=10, global_batch_size=4).batch(step)
        assert len({s.seed for s in slots}) == 4
        assert (
            sorted(
                sampler.local_batch(step, 0, 2) + sampler.local_batch(step, 1, 2),
                key=lambda s: s.slot,
            )
            == slots
        )
    # Sampling cannot acquire a length bias by depending on frame counts.
    longer = [Episode(e.index, e.task, e.start, e.end + 1000) for e in episodes]
    assert sampler.batch(3) == EpisodeSampler(longer, seed=10, global_batch_size=4).batch(3)
    with pytest.raises(ValueError, match="divisible"):
        sampler.local_batch(0, 0, 3)


def test_full_episode_events_mask_tail_and_omit_midpoint_actions():
    frames = [
        {"observation.state": torch.tensor([float(i)]), "action": torch.ones(5, 2), "task": "a"}
        for i in range(9)
    ]
    source = EpisodeSource(frames, [Episode(0, "a", 1, 8)], execution=4, horizon=5)
    events = list(source.events(0, seed=1))
    assert [e.frame["observation.state"].item() for e in events] == [1, 3, 5, 7]
    assert [e.supervised for e in events] == [True, False, True, False]
    assert source.decision_count(0) == 2
    assert events[0].frame["action_mask"].sum() == 10
    assert events[2].frame["action_mask"].sum() == 6
    assert "action" not in events[1].frame


def test_temporal_augmentation_replays_and_preserves_model_rng():
    def transform(x):
        return x * torch.rand(())

    image = torch.ones(3, 4, 4)
    state = torch.get_rng_state().clone()
    a = temporal_transform(image, transform, seed=11, camera="left")
    b = temporal_transform(image, transform, seed=11, camera="left")
    c = temporal_transform(image, transform, seed=11, camera="right")
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    assert torch.equal(torch.get_rng_state(), state)


def test_recipe_rejects_invalid_sizes_and_counts_phase_updates():
    with pytest.raises(ValueError):
        Recipe(global_batch_size=0)
    with pytest.raises(ValueError):
        Recipe(execution=3)
    r = Recipe(steps=7, phase_length=2, warmup_steps=1)
    assert [r.phase(i) for i in range(7)] == [
        "memory",
        "memory",
        "policy",
        "policy",
        "memory",
        "memory",
        "policy",
    ]
    assert r.phase_budget("memory") == 4
    assert r.phase_budget("policy") == 3
