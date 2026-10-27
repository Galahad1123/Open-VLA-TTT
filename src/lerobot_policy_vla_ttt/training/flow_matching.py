"""Independent noise levels for each batch row at each trajectory position."""

from contextlib import contextmanager
from contextvars import ContextVar

import torch

_ROW_SEEDS = ContextVar("flow_row_seeds", default=None)


@contextmanager
def flow_row_seeds(seeds):
    token = _ROW_SEEDS.set(seeds)
    try:
        yield
    finally:
        _ROW_SEEDS.reset(token)


def has_flow_row_seeds():
    return _ROW_SEEDS.get() is not None


def sample_flow(actions, mask, config):
    seeds = _ROW_SEEDS.get()
    if seeds is not None:
        if len(seeds) != len(actions):
            raise ValueError("Flow seeds must match batch")
        devices = [actions.device.index] if actions.is_cuda else []
        outputs = []
        with flow_row_seeds(None):
            for i, seed in enumerate(seeds):
                with torch.random.fork_rng(devices=devices):
                    torch.random.default_generator.manual_seed(seed)
                    if actions.is_cuda:
                        with torch.cuda.device(actions.device):
                            torch.cuda.manual_seed(seed)
                    outputs.append(sample_flow(actions[i : i + 1], mask[i : i + 1], config))
        return tuple(torch.cat(parts) for parts in zip(*outputs, strict=True))
    actions = actions.masked_fill(~mask, 0)
    noise = torch.randn_like(actions).masked_fill(~mask, 0)
    alpha = actions.new_tensor(config.flow_beta_alpha, dtype=torch.float32)
    beta = actions.new_tensor(config.flow_beta_beta, dtype=torch.float32)
    time = config.flow_time_scale * (
        1 - torch.distributions.Beta(alpha, beta).sample((len(actions),))
    )
    weight = time[:, None, None].to(actions.dtype)
    return (1 - weight) * noise + weight * actions, noise, time
