"""Actual tiny InternVL vision/projector/Qwen2, no pretrained assets or GPU."""

from copy import deepcopy

import torch
from transformers import BatchEncoding, InternVLConfig, InternVLModel, Qwen2Config

from lerobot_policy_vla_ttt.evo1 import internvl3_embedder
from lerobot_policy_vla_ttt.models.titans import model
from lerobot_policy_vla_ttt.models.titans.embedder import TitansEmbedder


class TinyTokenizer:
    def convert_tokens_to_ids(self, token):
        return 5

    def __call__(self, prompts, **kwargs):
        # Parent prompt builder creates four image positions per view.
        counts = [p.count("<IMG_CONTEXT>") for p in prompts]
        ids = torch.tensor([[1] + [5] * n + [2, 3] for n in counts])
        return BatchEncoding({"input_ids": ids, "attention_mask": torch.ones_like(ids)})


def patch_assets(monkeypatch):
    text = Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    cfg = InternVLConfig(
        vision_config=dict(
            image_size=[16, 16],
            patch_size=[4, 4],
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=1,
            num_attention_heads=4,
        ),
        text_config=text.to_dict(),
        image_seq_length=4,
        image_token_id=5,
        downsample_ratio=0.5,
    )
    cfg._attn_implementation = "eager"
    native = InternVLModel(cfg)
    monkeypatch.setattr(
        internvl3_embedder.AutoModel,
        "from_pretrained",
        lambda *a, **k: deepcopy(native).to(dtype=k.get("torch_dtype", torch.float32)),
    )
    monkeypatch.setattr(
        internvl3_embedder.AutoTokenizer, "from_pretrained", lambda *a, **k: TinyTokenizer()
    )
    monkeypatch.setattr(model, "TitansEmbedder", TitansEmbedder)
