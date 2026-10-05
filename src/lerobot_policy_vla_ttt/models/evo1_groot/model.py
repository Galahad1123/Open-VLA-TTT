"""Reuse Evo-1 VL fusion and freeze controls with an independent action expert."""

from torch import nn

from ...evo1.evo1_model import Evo1Model
from ...evo1.internvl3_embedder import InternVL3Embedder
from .action_head import GrootActionHead


class Evo1GrootModel(Evo1Model):
    def __init__(self, config, vlm_hub_kwargs=None):
        nn.Module.__init__(self)
        self.config = config
        self._device = config.device
        self.return_cls_only = config.return_cls_only
        self.rtc_processor = None
        train_vlm = bool(config.finetune_language_model or config.finetune_vision_model)
        self.embedder = InternVL3Embedder(
            model_name=config.vlm_model_name,
            image_size=int(config.image_resolution[0]),
            device=config.device,
            num_language_layers=config.vlm_num_layers,
            model_dtype=config.vlm_dtype,
            use_flash_attn=config.use_flash_attn,
            max_text_length=config.max_text_length,
            enable_gradient_checkpointing=config.enable_gradient_checkpointing and train_vlm,
            gradient_checkpointing_use_reentrant=config.gradient_checkpointing_use_reentrant,
            hub_kwargs=vlm_hub_kwargs,
        )
        self.horizon = config.chunk_size
        self.per_action_dim = config.max_action_dim
        self.action_head = GrootActionHead(config).to(config.device)
