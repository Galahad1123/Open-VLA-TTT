"""Global hierarchical sampling and lazy, complete, chronological episodes."""

import hashlib
import random
from dataclasses import dataclass

import torch
from torch.utils.data import default_collate


def derive_seed(*parts):
    digest = hashlib.sha256(repr(parts).encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


@dataclass(frozen=True)
class Episode:
    index: int
    task: str
    start: int
    end: int


@dataclass(frozen=True)
class Sample:
    slot: int
    episode_id: int
    seed: int


@dataclass(frozen=True)
class Event:
    frame: dict
    supervised: bool


class EpisodeSampler:
    def __init__(self, episodes, *, seed, global_batch_size):
        if global_batch_size < 1:
            raise ValueError("global_batch_size must be positive")
        self.seed, self.global_batch_size = seed, global_batch_size
        self.tasks = {}
        seen = set()
        for ep in episodes:
            if ep.index in seen or not ep.task or ep.end <= ep.start:
                raise ValueError("Invalid episode metadata")
            seen.add(ep.index)
            self.tasks.setdefault(ep.task, []).append(ep.index)
        if not self.tasks:
            raise ValueError("No episodes")
        self.tasks = {k: sorted(v) for k, v in sorted(self.tasks.items())}

    def batch(self, step):
        if step < 0:
            raise ValueError("step must be nonnegative")
        rng = random.Random(derive_seed(self.seed, "sampling", step))
        task = rng.choice(list(self.tasks))
        return [
            Sample(
                slot, rng.choice(self.tasks[task]), derive_seed(self.seed, "episode", step, slot)
            )
            for slot in range(self.global_batch_size)
        ]

    def local_batch(self, step, rank, world_size):
        if world_size < 1 or self.global_batch_size % world_size:
            raise ValueError("global batch must be divisible by world size")
        if not 0 <= rank < world_size:
            raise ValueError("Invalid rank")
        return self.batch(step)[rank::world_size]


def temporal_transform(image, transform, *, seed, camera):
    """Replay CPU transform RNG per camera without consuming the model RNG stream."""
    if image.device.type != "cpu":
        raise ValueError("Episode augmentation expects CPU images")
    py_state = random.getstate()
    try:
        with torch.random.fork_rng(devices=[]):
            camera_seed = derive_seed(seed, "augmentation", camera)
            torch.random.default_generator.manual_seed(camera_seed)
            random.seed(camera_seed)
            return transform(image)
    finally:
        random.setstate(py_state)


class EpisodeSource:
    def __init__(self, dataset, episodes, *, execution, horizon, transform=None):
        if execution < 2 or execution % 2 or horizon < 1:
            raise ValueError("Invalid execution/horizon")
        self.dataset, self.execution, self.horizon = dataset, execution, horizon
        self.transform = transform
        episodes = list(episodes)
        self.episodes = {e.index: e for e in episodes}
        if len(episodes) != len(self.episodes) or not episodes:
            raise ValueError("Empty or duplicate episodes")
        previous = 0
        for ep in sorted(episodes, key=lambda e: e.start):
            if not 0 <= previous <= ep.start < ep.end <= len(dataset):
                raise ValueError("Episode ranges must be disjoint and within dataset")
            previous = ep.end

    def decision_count(self, episode_id):
        ep = self.episodes[episode_id]
        return (ep.end - ep.start + self.execution - 1) // self.execution

    def events(self, episode_id, *, seed):
        ep = self.episodes[episode_id]
        for index in range(ep.start, ep.end, self.execution // 2):
            supervised = (index - ep.start) % self.execution == 0
            frame = dict(self.dataset[index])
            if self.transform is not None:
                for key, value in frame.items():
                    if key.startswith("observation.images."):
                        frame[key] = temporal_transform(
                            value, self.transform, seed=seed, camera=key
                        )
            if supervised:
                action = frame["action"]
                if action.ndim != 2 or action.shape[0] != self.horizon:
                    raise ValueError("Dataset must return a full action horizon")
                valid = torch.arange(self.horizon) < (ep.end - index)
                frame["action_mask"] = (
                    frame.get("action_mask", torch.ones_like(action, dtype=torch.bool))
                    & valid[:, None]
                )
            else:
                for key in ("action", "action_mask", "action_is_pad"):
                    frame.pop(key, None)
            yield Event(default_collate([frame]), supervised)
