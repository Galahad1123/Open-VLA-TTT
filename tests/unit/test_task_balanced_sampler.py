from collections import Counter

import numpy as np
import pytest
from torch.utils.data import DataLoader

from lerobot_policy_vla_ttt.training.task_balanced_sampler import TaskBalancedSampler


@pytest.mark.parametrize("batch_size", [6, 8, 12])
def test_each_batch_covers_tasks_and_every_prefix_is_balanced(batch_size):
    groups = {str(i): range(i * 100, i * 100 + i + 1) for i in range(6)}
    sampler = TaskBalancedSampler(groups, batch_size=batch_size, steps=101, seed=17)
    indices = list(sampler)
    assert len(indices) == len(sampler) == 101 * batch_size
    cumulative = Counter({i: 0 for i in range(6)})
    for start in range(0, len(indices), batch_size):
        batch = indices[start : start + batch_size]
        counts = Counter(index // 100 for index in batch)
        assert set(counts) == set(range(6))
        assert set(counts.values()) <= {batch_size // 6, batch_size // 6 + 1}
        cumulative.update(counts)
        assert max(cumulative.values()) - min(cumulative.values()) <= 1
        assert all(type(index) is int and index in groups[str(index // 100)] for index in batch)


def test_seed_epoch_and_reiteration_are_reproducible():
    groups = {str(i): range(i * 10, i * 10 + 10) for i in range(6)}
    sampler = TaskBalancedSampler(groups, 8, 20, 42)
    first = list(sampler)
    assert list(sampler) == first == list(TaskBalancedSampler(groups, 8, 20, 42))
    assert first != list(TaskBalancedSampler(groups, 8, 20, 43))
    sampler.set_epoch(1)
    assert list(sampler) != first
    sampler.set_epoch(0)
    assert list(sampler) == first


def test_sampling_is_uniform_with_replacement_and_batches_are_shuffled():
    sampler = TaskBalancedSampler({"a": [0, 1, 2, 3], "b": np.array([10, 11])}, 8, 2000, 7)
    batches = np.array(list(sampler)).reshape(-1, 8)
    counts = Counter(batches.ravel())
    assert set(counts) == {0, 1, 2, 3, 10, 11}
    assert all(1800 < counts[index] < 2200 for index in range(4))
    assert all(3800 < counts[index] < 4200 for index in (10, 11))
    assert any(len(set(batch)) < len(batch) for batch in batches)
    assert 700 < sum(batches[:, 0] < 10) < 1300


def test_flat_sampler_works_with_dataloader():
    sampler = TaskBalancedSampler({"a": [0], "b": range(1, 3)}, 2, 4, 0)
    batches = list(DataLoader([10, 20, 30], batch_size=2, sampler=sampler))
    assert len(batches) == 4
    assert all(10 in batch and (20 in batch or 30 in batch) for batch in batches)


@pytest.mark.parametrize(
    "groups",
    [
        {},
        {"a": []},
        {"a": range(0)},
        {"a": [0, 0]},
        {"a": [0], "b": [0]},
        {"a": [-1]},
        {"a": [1.5]},
        {"a": [True]},
        {"a": np.array([False])},
        {"a": np.array([[1]])},
        {"a": ["1"]},
        {"a": np.array([np.nan])},
        {"a": [2**64]},
        {1: [0]},
        {"": [0]},
        {"a": {0, 1}},
    ],
)
def test_rejects_invalid_groups(groups):
    with pytest.raises((TypeError, ValueError)):
        TaskBalancedSampler(groups, 8, 2, 0)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("batch_size", 0),
        ("batch_size", -1),
        ("batch_size", True),
        ("batch_size", 2.5),
        ("batch_size", 1),
        ("steps", 0),
        ("steps", -1),
        ("steps", True),
        ("steps", 2.5),
        ("seed", -1),
        ("seed", True),
        ("seed", 1.5),
    ],
)
def test_rejects_invalid_scalar_arguments(name, value):
    arguments = dict(batch_size=2, steps=2, seed=0)
    arguments[name] = value
    with pytest.raises((TypeError, ValueError)):
        TaskBalancedSampler({"a": [0], "b": [1]}, **arguments)


@pytest.mark.parametrize("epoch", [-1, True, 1.5])
def test_rejects_invalid_epoch(epoch):
    sampler = TaskBalancedSampler({"a": [0]}, 1, 1, 0)
    with pytest.raises((TypeError, ValueError)):
        sampler.set_epoch(epoch)
