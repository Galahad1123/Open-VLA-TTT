"""Read selected block outputs without replacing or modifying the VLM stack."""

from ...evo1.internvl3_embedder import InternVL3Embedder


class MultiLayerEmbedder(InternVL3Embedder):
    def __init__(self, config, *, vlm_hub_kwargs=None):
        super().__init__(
            model_name=config.vlm_model_name,
            image_size=int(config.image_resolution[0]),
            device=config.device,
            num_language_layers=config.vlm_num_layers,
            model_dtype=config.vlm_dtype,
            use_flash_attn=config.use_flash_attn,
            max_text_length=config.max_text_length,
            enable_gradient_checkpointing=config.enable_gradient_checkpointing,
            gradient_checkpointing_use_reentrant=False,
            hub_kwargs=vlm_hub_kwargs,
        )
        layers = self.model.language_model.layers
        self.layer_indices = config.vlm_layer_indices
        if max(self.layer_indices) >= len(layers):
            raise ValueError("Selected VL layer exceeds loaded language stack")
        self._capture = None
        self._handles = [
            layers[i].register_forward_hook(self._hook(i)) for i in set(self.layer_indices)
        ]

    def _hook(self, index):
        def capture(module, args, output):
            if self._capture is not None:
                self._capture[index] = output[0] if isinstance(output, tuple) else output

        return capture

    def encode(self, images, masks, prompts):
        if self._capture is not None:
            raise RuntimeError("Concurrent VLM calls on one policy are unsupported")
        captured = {}
        self._capture = captured
        try:
            context, valid = self.get_fused_image_text_embedding_batched(
                images, masks, prompts, False
            )
            return context, valid, tuple(captured[i] for i in self.layer_indices)
        finally:
            # Checkpoint backward recomputation must never overwrite/retain capture buffers.
            self._capture = None
