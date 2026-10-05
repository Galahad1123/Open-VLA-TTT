"""Episode state deliberately lives outside module state_dicts."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FastWeightState:
    weights: tuple[torch.Tensor, ...]
    position: torch.Tensor

    def detach(self):
        return FastWeightState(tuple(w.detach() for w in self.weights), self.position.detach())
