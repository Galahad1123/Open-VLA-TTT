"""Episode state deliberately lives outside module state_dicts."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FastWeightState:
    weights: tuple[torch.Tensor, ...]
    position: torch.Tensor
    decision_index: torch.Tensor
    update_count: torch.Tensor
    auxiliary: tuple[torch.Tensor, ...] | None = None

    def detach(self):
        return FastWeightState(
            tuple(w.detach() for w in self.weights),
            self.position.detach(),
            self.decision_index.detach(),
            self.update_count.detach(),
            None if self.auxiliary is None else tuple(w.detach() for w in self.auxiliary),
        )
