import pytest
import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from torch import nn

from lerobot_policy_vla_ttt.models.evo1_groot import model
from lerobot_policy_vla_ttt.models.evo1_groot.configuration_evo1_groot import Evo1GrootConfig


class TinyEmbedder(nn.Module):
    """Replace only the external pretrained VLM; keep policy/model/head/processors real."""

    def __init__(self, **kwargs):
        super().__init__()
        self.model = nn.ModuleDict()  # Expose the native HF InternVL branch layout.
        self.model.language_model = nn.Linear(16, 16)
        self.model.vision_tower = nn.Linear(3, 16)
        self.model.multi_modal_projector = nn.Linear(16, 16)

    def get_fused_image_text_embedding_batched(
        self, camera_images, image_masks, text_prompts, return_cls_only
    ):
        image = torch.stack([x.mean(dim=(-2, -1)) for x in camera_images], dim=1)
        vision = self.model.vision_tower(image)
        tokens = self.model.language_model(self.model.multi_modal_projector(vision))
        return tokens, image_masks[:, : tokens.shape[1]]


@pytest.fixture
def config(monkeypatch):
    monkeypatch.setattr(model, "InternVL3Embedder", TinyEmbedder)
    return Evo1GrootConfig(
        device="cpu",
        use_amp=False,
        embed_dim=16,
        hidden_dim=24,
        state_hidden_dim=24,
        num_heads=4,
        num_layers=2,
        chunk_size=3,
        n_action_steps=2,
        max_state_dim=10,
        max_action_dim=10,
        num_inference_timesteps=2,
        image_resolution=(16, 16),
        use_flash_attn=False,
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (8,)),
            "observation.images.image": PolicyFeature(FeatureType.VISUAL, (3, 16, 16)),
            "observation.images.image2": PolicyFeature(FeatureType.VISUAL, (3, 16, 16)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (7,))},
    )
