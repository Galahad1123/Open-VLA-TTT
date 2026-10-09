"""Unchanged VLM/DiT backbones plus explicit observation-memory branches."""

from torch import nn

from .embedder import MultiLayerEmbedder


class ObservationModel(nn.Module):
    def __init__(self, config, head_class, *, vlm_hub_kwargs=None):
        super().__init__()
        self.config = config
        self.embedder = MultiLayerEmbedder(config, vlm_hub_kwargs=vlm_hub_kwargs)
        self.action_head = head_class(config).to(config.device)
        self.rtc_processor = None
        self.set_phase(config.memory_training)

    def set_phase(self, phase):
        if phase not in {"memory_only", "policy_only", "joint"}:
            raise ValueError("Invalid phase")
        self.config.memory_training = phase
        self.requires_grad_(False)
        if phase != "memory_only":
            self.embedder.model.language_model.requires_grad_(
                bool(self.config.finetune_language_model)
            )
            self.embedder.model.vision_tower.requires_grad_(bool(self.config.finetune_vision_model))
            self.embedder.model.multi_modal_projector.requires_grad_(
                bool(self.config.finetune_vision_model)
            )
            self.action_head.requires_grad_(bool(self.config.finetune_action_head))
        for p in self.action_head.memory_parameters():
            p.requires_grad_(phase != "policy_only" and self.config.memory_enabled)
        if not self.action_head.interface_reads:
            for interface in self.action_head.interfaces:
                interface.gate.requires_grad_(False)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        for module in (
            self.embedder.model.language_model,
            self.embedder.model.vision_tower,
            self.embedder.model.multi_modal_projector,
        ):
            if not any(p.requires_grad for p in module.parameters()):
                module.eval()
        # Frozen backbone dropout must not change the conditioning during memory-only training.
        if self.config.memory_training == "memory_only":
            for block in self.action_head.blocks:
                block.eval()
        return self
