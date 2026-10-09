"""Compose stateful VLM and unchanged Evo1-GR00T action expert."""

from torch import nn

from ..evo1_groot.action_head import GrootActionHead
from .embedder import TitansEmbedder


class TitansModel(nn.Module):
    def __init__(self, config, connection_class, *, vlm_hub_kwargs=None):
        super().__init__()
        self.embedder = TitansEmbedder(config, connection_class, vlm_hub_kwargs=vlm_hub_kwargs)
        self.action_head = GrootActionHead(config).to(config.device)
        self.rtc_processor = None
        self.requires_grad_(False)
        if config.memory_training == "joint":
            self.embedder.model.language_model.requires_grad_(config.finetune_language_model)
            self.embedder.model.vision_tower.requires_grad_(config.finetune_vision_model)
            self.embedder.model.multi_modal_projector.requires_grad_(config.finetune_vision_model)
            self.action_head.requires_grad_(config.finetune_action_head)
        self.embedder.connections.requires_grad_(config.memory_enabled)
        for connection in self.embedder.connections.values():
            connection.memory.gate.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        # Frozen pretrained modules stay deterministic but still propagate input gradients.
        for module in (
            self.embedder.model.language_model,
            self.embedder.model.vision_tower,
            self.embedder.model.multi_modal_projector,
            self.action_head,
        ):
            if not any(p.requires_grad for p in module.parameters()):
                module.eval()
        return self
