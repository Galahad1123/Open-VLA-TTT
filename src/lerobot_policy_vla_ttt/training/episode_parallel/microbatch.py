"""Experimental static episode batches; padded lanes never write memory or loss.

Event clocks and TBPTT boundaries are relative to each episode's start. Flow
randomness is keyed by episode seed and event index, independent of batch size.
"""

from itertools import zip_longest

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from ..flow_matching import flow_row_seeds
from .data import derive_seed
from .engine import episode_rng


def join_frames(frames):
    keys = set(frames[0])
    if any(set(f) != keys for f in frames):
        raise ValueError("Preprocessed frame keys differ")
    result = {}
    for key in keys:
        values = [f[key] for f in frames]
        if isinstance(values[0], torch.Tensor):
            result[key] = torch.cat(values)
        elif isinstance(values[0], list):
            result[key] = [x for v in values for x in v]
        elif all(v == values[0] for v in values):
            result[key] = values[0]
        else:
            raise ValueError(f"Cannot batch frame field {key}")
    return result


def backward_microbatch(policy, streams, *, decisions, seeds, recipe, backward, preprocess=None):
    if not streams or len(streams) != len(decisions) or len(streams) != len(seeds):
        raise ValueError("Invalid episode microbatch")
    if min(decisions) < 1:
        raise ValueError("Each episode needs supervision")
    if policy.config.type not in {"evo1_groot", "aqmem"}:
        raise ValueError("Experimental batching supports evo1_groot and aqmem only")
    if any(p.requires_grad for p in policy.model.embedder.parameters()):
        raise ValueError("Experimental batching requires frozen VLM")
    device = next(policy.parameters()).device
    seen = [0] * len(streams)
    metrics = dict(loss_sum=0.0, episodes=len(streams), events=0, decisions=0, supervised_scalars=0)
    memory, segment = None, None
    with (
        episode_rng(derive_seed(seeds[0], "microbatch_dropout"), device),
        sdpa_kernel(SDPBackend.MATH),
    ):
        for index, events in enumerate(zip_longest(*streams)):
            live = [e for e in events if e is not None]
            template = live[0].frame
            frames = []
            for event in events:
                raw = dict(template if event is None else event.frame)
                supervised = event is not None and event.supervised
                if not supervised:
                    raw["action"] = torch.zeros(
                        1,
                        policy.config.chunk_size,
                        policy.config.output_features["action"].shape[0],
                    )
                    raw["action_mask"] = torch.zeros_like(raw["action"], dtype=torch.bool)
                # A single explicit action_mask encodes supervision and tail padding.
                if "action_mask" not in raw:
                    raw["action_mask"] = torch.ones_like(raw["action"], dtype=torch.bool)
                if "action_is_pad" in raw:
                    raw["action_mask"] = raw["action_mask"] & ~raw["action_is_pad"][..., None]
                raw.pop("action_is_pad", None)
                frame = preprocess(raw) if preprocess else raw
                frames.append(frame)
            batch = join_frames(frames)
            active = torch.tensor([e is not None for e in events], device=device)
            supervised = torch.tensor(
                [e is not None and e.supervised for e in events], device=device
            )
            with flow_row_seeds([derive_seed(seed, "flow_event", index) for seed in seeds]):
                if policy.config.type == "aqmem":
                    num, count, memory = policy.forward_step(
                        batch,
                        memory,
                        write_mask=active,
                        supervision_mask=supervised,
                        loss_reduction="none",
                    )
                    means = num / count.clamp_min(1)
                elif supervised.any():
                    means, _ = policy(batch, reduction="none")
                    _, mask = policy._prepare_actions(batch)
                    count = mask.sum((1, 2))
                else:
                    means = torch.zeros(len(streams), device=device)
                    count = torch.zeros(len(streams), device=device, dtype=torch.long)
            if supervised.any():
                if (count[supervised] < 1).any():
                    raise ValueError("Empty supervised action mask")
                weights = torch.tensor(decisions, device=device)
                loss = (means * supervised / weights).sum()
                contribution = loss / recipe.global_batch_size
                segment = contribution if segment is None else segment + contribution
                metrics["loss_sum"] += float(loss.detach())
            metrics["events"] += int(active.sum())
            metrics["decisions"] += int(supervised.sum())
            metrics["supervised_scalars"] += int(count[supervised].sum())
            for i, e in enumerate(events):
                seen[i] += int(e is not None and e.supervised)
            if (index + 1) % recipe.tbptt_events == 0:
                if segment is not None and segment.requires_grad:
                    backward(segment)
                memory = None if memory is None else tuple(s.detach() for s in memory)
                segment = None
        if seen != list(decisions):
            raise ValueError(f"Expected decisions {decisions}, got {seen}")
        if segment is not None and segment.requires_grad:
            backward(segment)
    return metrics, None if memory is None else tuple(s.detach() for s in memory)
