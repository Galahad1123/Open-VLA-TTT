"""Detached, RNG-free observations; local rank-0 features and synchronized gradients."""

from contextvars import ContextVar

import torch

CURRENT = ContextVar("episode_diagnostics", default=None)


def rms(x):
    return x.detach().float().square().mean().sqrt()


class Diagnostics:
    def __init__(self, interval, enabled=True):
        self.interval = interval if enabled else 0
        self.active = False
        self.mask = None

    def __enter__(self):
        self.token = CURRENT.set(self)
        return self

    def __exit__(self, *args):
        CURRENT.reset(self.token)

    def begin(self, step):
        self.active = bool(self.interval and (step == 1 or step % self.interval == 0))
        self.reads, self.states, self.groups, self.samples, self.gates = {}, [], [], [], {}
        self.mask = None

    @torch.no_grad()
    def read(self, layer, main, residual):
        if not self.active:
            return
        # Exclude padded and unsupervised rows from action-read diagnostics.
        if self.mask is not None:
            main, residual = main[self.mask], residual[self.mask]
        if main.numel() == 0:
            return
        values = torch.stack((main.float().square().mean(), residual.float().square().mean()))
        if layer not in self.reads:
            self.reads[layer] = [values, 1]
        else:
            self.reads[layer][0].add_(values)
            self.reads[layer][1] += 1

    @torch.no_grad()
    def state(self, states):
        if self.active and states is not None:
            self.states.append([torch.stack([rms(w) for w in s.weights]) for s in states])

    @staticmethod
    def sample(group):
        # Uniform positions across the logical concatenation, at most 4096 scalars.
        params = group["params"]
        total = sum(p.numel() for p in params)
        stride = max(1, (total + 4095) // 4096)
        pieces, offset = [], 0
        for p in params:
            start = (-offset) % stride
            if start < p.numel():
                pieces.append(p.detach().reshape(-1)[start::stride].float())
            offset += p.numel()
        return torch.cat(pieces).clone()

    @torch.no_grad()
    def before_step(self, optimizer, policy):
        if not self.active:
            return
        self.optimizer = optimizer
        for i, g in enumerate(optimizer.param_groups):
            squares = [
                p.grad.detach().float().square().sum() for p in g["params"] if p.grad is not None
            ]
            norm = float(torch.stack(squares).sum().sqrt()) if squares else 0.0
            self.groups.append(
                dict(index=i, category=g["category"], lr=g["lr"], grad_norm_before_clip=norm)
            )
            self.samples.append(self.sample(g))
        head = policy.model.action_head
        for i, gate in enumerate(getattr(head, "action_gates", [])):
            v = gate.detach().float().tanh()
            self.gates[str(i)] = dict(
                tanh_mean=float(v.mean()),
                tanh_rms=float(rms(v)),
                tanh_min=float(v.min()),
                tanh_max=float(v.max()),
            )

    @torch.no_grad()
    def after_step(self):
        if not self.active:
            return None
        for result, before, group in zip(
            self.groups, self.samples, self.optimizer.param_groups, strict=True
        ):
            delta = self.sample(group) - before
            result.update(
                sample_count=before.numel(),
                sampled_parameter_rms=float(rms(before)),
                sampled_update_rms=float(rms(delta)),
                sampled_relative_update=float(rms(delta) / rms(before).clamp_min(1e-12)),
            )
        reads = {}
        for layer, (total, n) in self.reads.items():
            values = (total / n).sqrt().cpu().tolist()
            reads[str(layer)] = dict(
                main_rms=values[0],
                residual_rms=values[1],
                residual_to_main=values[1] / max(values[0], 1e-12),
                calls=n,
            )
        return dict(
            scope="rank0_features_synchronized_gradients",
            sample_protocol="uniform_flat_positions_at_most_4096_per_optimizer_group",
            optimizer_groups=self.groups,
            gates=self.gates,
            memory_reads=reads,
            fast_weights=[[v.cpu().tolist() for v in batch] for batch in self.states],
        )


def record_read(layer, main, residual):
    diag = CURRENT.get()
    if diag is not None and diag.active:
        diag.read(layer, main, residual)
