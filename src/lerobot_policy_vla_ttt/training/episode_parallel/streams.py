"""Persistent episode slots with a fixed number of supervised chunks per update."""

import random
from collections import Counter
from dataclasses import dataclass

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from .data import EpisodeSampler, derive_seed
from .engine import episode_rng


@dataclass(frozen=True)
class ChunkSample:
    slot: int
    episode_id: int
    decision: int
    seed: int
    flow_seed: int


class ChunkStreamSampler:
    """One chronological chunk per slot. Refill exhausted slots independently.

    Selection depends only on metadata, seed and step. Replay to a checkpoint
    boundary reconstructs every cursor without reading frames or replaying models.
    Active tasks are kept distinct when enough tasks are available.
    """

    def __init__(self, source, *, seed, global_batch_size):
        self.source, self.seed = source, seed
        self.tasks = EpisodeSampler(
            source.episodes.values(), seed=seed, global_batch_size=global_batch_size
        ).tasks
        self.slots = [None] * global_batch_size
        self.generations = [0] * global_batch_size
        self.step = 0

    def seek(self, step):
        if step < self.step:
            raise ValueError("Cannot seek backwards")
        while self.step < step:
            self.batch(self.step)

    def batch(self, step):
        if step != self.step:
            raise ValueError("Chunk stream expects consecutive steps")
        # Remove every exhausted slot before assigning replacement tasks.
        for i, slot in enumerate(self.slots):
            if slot is not None and slot[1] == self.source.decision_count(slot[0]):
                self.slots[i] = None
        occupied = Counter(self.source.episodes[s[0]].task for s in self.slots if s is not None)
        for i, slot in enumerate(self.slots):
            if slot is not None:
                continue
            seed = derive_seed(self.seed, "stream_episode", i, self.generations[i])
            rng = random.Random(seed)
            fewest = min(occupied[t] for t in self.tasks)
            task = rng.choice([t for t in self.tasks if occupied[t] == fewest])
            self.slots[i] = (rng.choice(self.tasks[task]), 0, seed)
            self.generations[i] += 1
            occupied[task] += 1
        samples = []
        for i, (episode, decision, seed) in enumerate(self.slots):
            samples.append(
                ChunkSample(
                    i, episode, decision, seed, derive_seed(self.seed, "stream_flow", step, i)
                )
            )
            self.slots[i] = (episode, decision + 1, seed)
        self.step += 1
        return samples


def prepare_chunks(policy, source, samples, preprocess):
    """Count valid elements after preprocessing, before global normalization."""
    if policy.config.type != "evo1_groot":
        raise ValueError("Chunk streams require stateless evo1_groot")
    prepared = []
    for sample in samples:
        frame = source.decision(sample.episode_id, sample.decision, seed=sample.seed)
        if preprocess:
            frame = preprocess(frame)
        # This is the exact mask used by Evo1Policy.forward (including padding).
        with torch.no_grad():
            _, mask = policy._prepare_actions(frame)
            count = int(mask.sum())
        if count < 1:
            raise ValueError("Supervised decision has no valid action elements")
        prepared.append((sample, frame, count))
    return prepared


def backward_chunks(policy, prepared, *, global_valid_count, backward):
    """Logical batch mean over all valid action elements, across every rank.

    Microbatch one chunk at a time for bounded GPU graph memory. Independent
    slot RNG makes noise and dropout invariant to rank assignment.
    """
    if global_valid_count < 1:
        raise ValueError("Global batch has no valid action elements")
    metrics = dict.fromkeys(
        ("loss_sum", "episodes", "events", "decisions", "supervised_scalars"), 0
    )
    device = next(policy.parameters()).device
    for sample, frame, count in prepared:
        with episode_rng(sample.flow_seed, device), sdpa_kernel(SDPBackend.MATH):
            loss, info = policy(frame)
            if int(info["active_action_dims"]) != count:
                raise ValueError("Prepared action mask differs from forward mask")
            numerator = loss * count
            backward(numerator / global_valid_count)
        metrics["loss_sum"] += float(numerator.detach())
        metrics["supervised_scalars"] += count
        metrics["decisions"] += 1
        metrics["events"] += 1
        metrics["episodes"] += int(sample.decision == 0)
    return metrics
