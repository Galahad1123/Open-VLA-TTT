"""Independent noise levels for each batch row at each trajectory position."""

import torch


def sample_flow(actions, mask, config):
    actions = actions.masked_fill(~mask, 0)
    noise = torch.randn_like(actions).masked_fill(~mask, 0)
    alpha = actions.new_tensor(config.flow_beta_alpha, dtype=torch.float32)
    beta = actions.new_tensor(config.flow_beta_beta, dtype=torch.float32)
    time = config.flow_time_scale * (
        1 - torch.distributions.Beta(alpha, beta).sample((len(actions),))
    )
    weight = time[:, None, None].to(actions.dtype)
    return (1 - weight) * noise + weight * actions, noise, time
