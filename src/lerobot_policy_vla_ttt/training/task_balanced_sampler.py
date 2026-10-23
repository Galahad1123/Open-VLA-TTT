"""Finite, reproducible frame sampling balanced over dataset task categories."""

from collections.abc import Iterator
from numbers import Integral

import numpy as np
from torch.utils.data import Sampler


def _integer(value: int, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return int(value)


class TaskBalancedSampler(Sampler[int]):
    """Yield exactly ``steps * batch_size`` dataset indices.

    Each consecutive batch draws uniformly with replacement within each task.
    Every task gets ``batch_size // number_of_tasks`` draws; remaining slots
    cycle through a seeded task permutation, keeping all prefix totals within
    one sample of each other. The complete batch is then shuffled.

    Groups are dataset task categories, not textual instruction variants.
    Iteration is repeatable until ``set_epoch`` changes the random stream.
    Pass the same batch size to the consuming DataLoader.
    """

    def __init__(
        self,
        frame_groups: dict[str, list[int] | range | np.ndarray],
        batch_size: int,
        steps: int,
        seed: int,
    ) -> None:
        self.batch_size = _integer(batch_size, "batch_size", 1)
        self.steps = _integer(steps, "steps", 1)
        self.seed = _integer(seed, "seed", 0)
        self.epoch = 0
        if not isinstance(frame_groups, dict) or not frame_groups:
            raise ValueError("frame_groups must be a nonempty dictionary")
        if self.batch_size < len(frame_groups):
            raise ValueError("batch_size must include at least one sample per task")

        self._groups: list[range | np.ndarray] = []
        for name, indices in frame_groups.items():
            if not isinstance(name, str) or not name:
                raise ValueError("task names must be nonempty strings")
            if not isinstance(indices, (list, range, np.ndarray)):
                raise TypeError("frame groups must contain lists, ranges, or NumPy arrays")
            if isinstance(indices, np.ndarray) and indices.ndim != 1:
                raise ValueError("frame groups must be one-dimensional")
            if not len(indices):
                raise ValueError(f"task {name!r} has no frames")
            if isinstance(indices, range):
                low, high = sorted((indices[0], indices[-1]))
                if low < 0 or high > np.iinfo(np.int64).max:
                    raise ValueError("frame indices must be nonnegative int64 values")
                self._groups.append(indices)
                continue
            if isinstance(indices, list):
                # Check before conversion: NumPy otherwise silently coerces bools/floats.
                for index in indices:
                    _integer(index, "frame index", 0)
            values = np.asarray(indices)
            if values.dtype.kind not in "iu":
                raise TypeError("frame indices must be integers")
            if np.any(values < 0) or np.any(values > np.iinfo(np.int64).max):
                raise ValueError("frame indices must be nonnegative int64 values")
            values = values.astype(np.int64, copy=True)
            values.flags.writeable = False
            self._groups.append(values)

        # Temporary compact arrays validate both duplicates and cross-task overlap.
        all_indices = np.concatenate([np.asarray(group) for group in self._groups])
        if len(np.unique(all_indices)) != len(all_indices):
            raise ValueError("frame groups must contain unique, nonoverlapping indices")

    def __len__(self) -> int:
        return self.steps * self.batch_size

    def set_epoch(self, epoch: int) -> None:
        """Select a new deterministic stream; does not advance during iteration."""
        self.epoch = _integer(epoch, "epoch", 0)

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch]))
        task_count = len(self._groups)
        order = rng.permutation(task_count)
        base, extra = divmod(self.batch_size, task_count)
        cursor = 0
        for _ in range(self.steps):
            counts = np.full(task_count, base, dtype=np.int64)
            for offset in range(extra):
                counts[order[(cursor + offset) % task_count]] += 1
            cursor = (cursor + extra) % task_count
            batch = []
            for group, count in zip(self._groups, counts, strict=True):
                positions = rng.integers(len(group), size=int(count))
                batch.extend(int(group[int(position)]) for position in positions)
            rng.shuffle(batch)
            yield from batch
