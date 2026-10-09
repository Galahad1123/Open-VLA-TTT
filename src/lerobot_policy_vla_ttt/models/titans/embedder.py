"""Reuse InternVL preprocessing/weights, explicitly execute its Qwen2 language stack.

Image-feature replacement follows installed Transformers InternVLModel.forward.
The Evo-1 snapshot and installed Transformers files are never modified.
"""

import torch
from torch import nn

from ...evo1.internvl3_embedder import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    InternVL3Embedder,
    _batched_pixel_values,
)
from .language import language_step


class TitansEmbedder(InternVL3Embedder):
    def __init__(self, config, connection_class, *, vlm_hub_kwargs=None):
        super().__init__(
            model_name=config.vlm_model_name,
            image_size=int(config.image_resolution[0]),
            device=config.device,
            num_language_layers=config.vlm_num_layers,
            model_dtype=config.vlm_dtype,
            use_flash_attn=False,
            max_text_length=config.max_text_length,
            enable_gradient_checkpointing=config.enable_gradient_checkpointing,
            gradient_checkpointing_use_reentrant=False,
            hub_kwargs=vlm_hub_kwargs,
        )
        if self.model.config.text_config.model_type != "qwen2":
            raise ValueError("Titans VLM integration currently requires InternVL with Qwen2")
        if self.model.config.text_config.hidden_size != config.embed_dim:
            raise ValueError("VLM hidden size must match action conditioning width")
        if len(self.model.language_model.layers) != config.vlm_num_layers:
            raise ValueError("Requested language layer count exceeds the pretrained model")
        if any(
            layer.self_attn.sliding_window is not None for layer in self.model.language_model.layers
        ):
            raise ValueError("Expected the baseline full-causal Qwen2 layers, not sliding layers")
        self.memory_config = config
        self.connections = nn.ModuleDict(
            {str(i): connection_class(config) for i in config.memory_layers}
        ).to(config.device)
        # Language checkpointing is owned by the explicit pure-state loop.
        self.model.language_model.gradient_checkpointing_disable()

    def encode(
        self,
        camera_images,
        image_masks,
        text_prompts,
        memory=None,
        write_mask=None,
        reset_mask=None,
    ):
        batch, views = image_masks.shape
        tiles = [[1] * views for _ in range(batch)]
        prompts = self._build_multimodal_prompts(tiles, text_prompts)
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
        ).to(self.device)
        ids = encoded["input_ids"]
        if not ((ids == self.img_context_token_id).sum(1) == views * self.num_image_token).all():
            raise ValueError("Prompt truncation removed image placeholders")
        valid = self._mask_absent_image_tokens(
            ids, encoded["attention_mask"], image_masks, tiles
        ).bool()
        visual = (ids == self.img_context_token_id) & valid
        pixels = _batched_pixel_values(
            camera_images,
            views,
            self.image_size,
            torch.tensor(IMAGENET_MEAN),
            torch.tensor(IMAGENET_STD),
            self.model_dtype,
            self.device,
        )
        embeddings = self.model.get_input_embeddings()(ids)
        images = self.model.get_image_features(pixels, return_dict=True).pooler_output
        images = images.to(embeddings)
        placeholders = self.model.get_placeholder_mask(
            ids, inputs_embeds=embeddings, image_features=images
        )
        embeddings = embeddings.masked_scatter(placeholders, images)
        hidden, state = language_step(
            self.model.language_model,
            self.connections,
            self.memory_config,
            embeddings,
            valid,
            visual,
            memory,
            write_mask,
            reset_mask,
        )
        if self.memory_config.return_cls_only:
            positions = torch.arange(valid.shape[1], device=valid.device)
            last = (valid.long() * positions).argmax(1)
            hidden = hidden[torch.arange(batch, device=hidden.device), last][:, None]
            valid = torch.ones((batch, 1), device=hidden.device, dtype=torch.bool)
        return hidden.float(), valid, state
