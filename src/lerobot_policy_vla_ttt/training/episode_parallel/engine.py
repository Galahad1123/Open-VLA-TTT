"""Local TBPTT: one episode per microbatch, one update per global batch."""

import random
from contextlib import contextmanager

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


@contextmanager
def episode_rng(seed, device):
    """Flow noise/dropout follow a global sample slot, not its assigned rank."""
    py_state, np_state = random.getstate(), np.random.get_state()
    devices = [device.index] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            if devices:
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed % (2**32))
            yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


def backward_episode(policy, events, *, decisions, recipe, backward, preprocess=None):
    """Accumulate SUM(episode means)/global_batch; never step or clear gradients.

    Single-episode events make per-chunk and per-episode normalization explicit,
    without changing the legacy forward_step interface. No collectives occur here.
    """
    if decisions < 1:
        raise ValueError("An episode needs supervised decisions")
    memory, segment_loss = None, None
    loss_sum, count, seen, event_count = 0.0, 0, 0, 0
    with sdpa_kernel(SDPBackend.MATH):
        for event in events:
            frame = preprocess(event.frame) if preprocess else event.frame
            device = next(policy.parameters()).device
            if "observation.state" in frame and len(frame["observation.state"]) != 1:
                raise ValueError("Microbatch must contain exactly one episode")
            supervised = torch.tensor([event.supervised], device=device)
            numerator, valid_count, memory = policy.forward_step(
                frame,
                memory,
                supervision_mask=supervised,
                write_mask=torch.ones(1, device=device, dtype=torch.bool),
            )
            if event.supervised:
                n = int(valid_count)
                if n < 1:
                    raise ValueError("Supervised decision has no valid action elements")
                chunk_mean = numerator / valid_count
                contribution = chunk_mean / (decisions * recipe.global_batch_size)
                segment_loss = contribution if segment_loss is None else segment_loss + contribution
                loss_sum += float(chunk_mean.detach()) / decisions
                count += n
                seen += 1
            event_count += 1
            if event_count % recipe.tbptt_events == 0:
                if segment_loss is not None and segment_loss.requires_grad:
                    backward(segment_loss)
                memory = None if memory is None else tuple(s.detach() for s in memory)
                segment_loss = None
            del frame, numerator
        if seen != decisions:
            raise ValueError(f"Expected {decisions} decisions, received {seen}")
        if segment_loss is not None and segment_loss.requires_grad:
            backward(segment_loss)
    memory = None if memory is None else tuple(s.detach() for s in memory)
    return {
        "loss_sum": loss_sum,
        "decisions": seen,
        "events": event_count,
        "supervised_scalars": count,
        "episodes": 1,
    }, memory
