"""T²Mem Appendix A.1: per-head novelty/residual scaling and curvature step.

Calibration is stop-gradient; the actual update preserves higher-order gradients.
The objective callable is necessary: gradients alone cannot determine curvature.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class T2MemAdaptive(nn.Module):
    def __init__(self):
        super().__init__()
        self.rho = nn.Parameter(torch.tensor(math.log(math.expm1(1.0))))

    def step(self, weights, gradients, write_mask, *, initial, keys, values, forward):
        def dot(xs, ys):
            return sum((x * y).flatten(2).sum(-1) for x, y in zip(xs, ys, strict=True))

        with torch.no_grad():
            w = tuple(x.detach() for x in weights)
            g = tuple(x.detach() for x in gradients)
            delta = tuple(x - z.detach() for x, z in zip(w, initial, strict=True))
            G, D = dot(g, g), dot(delta, delta)
            alignment = -dot(g, delta) / (G.sqrt() * D.sqrt()).clamp_min(1e-12)
            novelty = 1 - alignment.clamp(0, 1)
            novelty = torch.where(D <= 1e-24, 1.0, novelty)
            novelty = torch.where(G <= 1e-24, 0.0, novelty)
            k, v = keys.detach(), values.detach()
            prediction, jg = torch.func.jvp(lambda *p: forward(p, k), w, g)
            residual = (
                (prediction - v).square().mean((-1, -2)).sqrt()
                / v.square().mean((-1, -2)).sqrt().clamp_min(1e-8)
            ).clamp(0, 1)
            z = novelty * (1 + residual) / 2
            beta = 0.1 + 0.9 * (3 * z.square() - 2 * z.pow(3))
            heads = k.shape[1]
            curvature = 2 * jg.square().mean((-1, -2)) / heads
            q = (G / curvature.clamp_min(1e-30)).clamp(1e-12, 1e3)
            candidate = tuple(
                x - q.reshape(*q.shape, *([1] * (x.ndim - 2))) * d
                for x, d in zip(w, g, strict=True)
            )
            loss = (prediction - v).square().mean((-1, -2)) / heads
            trial = (forward(candidate, k) - v).square().mean((-1, -2)) / heads
            b = (trial - loss + G * q) / q.square()
            length = torch.where(b > 0, G / (2 * b).clamp_min(1e-30), q).clamp_max(1e3)
        eta = 0.5 * F.softplus(self.rho) * length * beta
        updated = tuple(
            torch.where(
                write_mask.reshape(-1, *([1] * (x.ndim - 1))),
                x - eta.reshape(*eta.shape, *([1] * (x.ndim - 2))) * g,
                x,
            )
            for x, g in zip(weights, gradients, strict=True)
        )
        return updated, None
