"""CPU fixtures use real tiny HF Qwen2 layers, replacing only image/text asset loading."""

from types import SimpleNamespace

import pytest
import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from torch import nn
from transformers import Qwen2Config, Qwen2Model

from lerobot_policy_vla_ttt.models.mac.configuration_mac import MACConfig
from lerobot_policy_vla_ttt.models.mac.modeling_mac import MACPolicy
from lerobot_policy_vla_ttt.models.mag.configuration_mag import MAGConfig
from lerobot_policy_vla_ttt.models.mag.modeling_mag import MAGPolicy
from lerobot_policy_vla_ttt.models.mal.configuration_mal import MALConfig
from lerobot_policy_vla_ttt.models.mal.modeling_mal import MALPolicy
from lerobot_policy_vla_ttt.models.titans import model
from lerobot_policy_vla_ttt.models.titans.language import language_step

KINDS = {
    "mac": (MACConfig, MACPolicy),
    "mag": (MAGConfig, MAGPolicy),
    "mal": (MALConfig, MALPolicy),
}


def tiny_language():
    cfg = Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        attention_dropout=0.0,
    )
    cfg._attn_implementation = "eager"
    return Qwen2Model(cfg)


class TinyEmbedder(nn.Module):
    def __init__(self, config, connection_class, **kwargs):
        super().__init__()
        self.memory_config = config
        self.model = nn.Module()
        self.model.language_model = tiny_language()
        self.model.vision_tower = nn.Linear(3, 16)
        self.model.multi_modal_projector = nn.Linear(16, 16)
        self.connections = nn.ModuleDict(
            {str(i): connection_class(config) for i in config.memory_layers}
        )

    def encode(
        self,
        camera_images,
        image_masks,
        text_prompts,
        memory=None,
        write_mask=None,
        reset_mask=None,
    ):
        image = torch.stack([x.mean((-1, -2)) for x in camera_images], dim=1)
        image = self.model.multi_modal_projector(self.model.vision_tower(image))
        ids = torch.tensor([[len(p) % 32, 1] for p in text_prompts], device=image.device)
        text = self.model.language_model.embed_tokens(ids)
        x = torch.cat((image, text), dim=1)
        valid = torch.cat(
            (image_masks[:, : image.shape[1]], torch.ones_like(ids, dtype=torch.bool)), dim=1
        )
        visual = valid.clone()
        visual[:, image.shape[1] :] = False
        y, state = language_step(
            self.model.language_model,
            self.connections,
            self.memory_config,
            x,
            valid,
            visual,
            memory,
            write_mask,
            reset_mask,
        )
        return y, valid, state


@pytest.fixture(params=list(KINDS))
def setup(request, monkeypatch):
    monkeypatch.setattr(model, "TitansEmbedder", TinyEmbedder)
    config_class, policy_class = KINDS[request.param]
    config = config_class(
        device="cpu",
        dtype="float32",
        vlm_dtype="float32",
        use_flash_attn=False,
        embed_dim=16,
        hidden_dim=24,
        state_hidden_dim=24,
        num_heads=4,
        num_layers=2,
        vlm_num_layers=2,
        memory_layers=(0, 1),
        memory_dim=4,
        memory_hidden_dim=6,
        chunk_size=3,
        n_action_steps=2,
        num_inference_timesteps=2,
        max_state_dim=10,
        max_action_dim=10,
        image_resolution=(16, 16),
        num_persistent_tokens=2,
        enable_gradient_checkpointing=False,
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (8,)),
            "observation.images.image": PolicyFeature(FeatureType.VISUAL, (3, 16, 16)),
            "observation.images.image2": PolicyFeature(FeatureType.VISUAL, (3, 16, 16)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (7,))},
    )
    return SimpleNamespace(config=config, policy_class=policy_class)


def batch():
    return {
        "observation.state": torch.rand(2, 8),
        "observation.images.image": torch.rand(2, 3, 16, 16),
        "observation.images.image2": torch.rand(2, 3, 16, 16),
        "action": torch.rand(2, 3, 7),
        "task": ["move", "pick"],
    }


def stats():
    return {
        key: {"min": -torch.ones(n), "max": torch.ones(n)}
        for key, n in [("action", 7), ("observation.state", 8)]
    }
