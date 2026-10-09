"""Actual tiny InternVL vision/projector/Qwen2, no pretrained assets or GPU."""

from copy import deepcopy
from dataclasses import fields, replace

import pytest
import torch
from transformers import BatchEncoding, InternVLConfig, InternVLModel, Qwen2Config

from lerobot_policy_vla_ttt.evo1 import internvl3_embedder
from lerobot_policy_vla_ttt.models.evo1_groot.configuration_evo1_groot import Evo1GrootConfig
from lerobot_policy_vla_ttt.models.evo1_groot.modeling_evo1_groot import Evo1GrootPolicy
from lerobot_policy_vla_ttt.models.titans import model
from lerobot_policy_vla_ttt.models.titans.embedder import TitansEmbedder

from .conftest import batch


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


@pytest.mark.parametrize("checkpointing", [False, True])
def test_actual_internvl_training_bypass_and_baseline_import(setup, monkeypatch, checkpointing):
    patch_assets(monkeypatch)
    cfg = replace(
        setup.config,
        enable_gradient_checkpointing=checkpointing,
        memory_training="joint",
        training_stage="stage2",
        memory_update_rule="momentum_decay",
    )
    baseline_values = {f.name: getattr(cfg, f.name) for f in fields(Evo1GrootConfig)}
    baseline = Evo1GrootPolicy(Evo1GrootConfig(**baseline_values)).eval()
    p = setup.policy_class(cfg)
    report = p.initialize_from_baseline(baseline)
    assert report["copied"] and report["new"]
    assert all(k.startswith("embedder.connections.") for k in report["new"])
    p.config.memory_enabled = False
    b = batch()
    images, valid_images = p._collect_image_batches(b)
    prompts = p._normalize_task_batch(b)
    original, mask = baseline.model.embedder.get_fused_image_text_embedding_batched(
        images, valid_images, prompts, return_cls_only=False
    )
    actual, actual_mask, state = p._encode(b, None)
    torch.testing.assert_close(actual[mask], original[mask], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(actual_mask, mask)
    assert state is None
    torch.manual_seed(10)
    expected_actions = baseline.predict_action_chunk(b)
    torch.manual_seed(10)
    torch.testing.assert_close(p.predict_action_chunk(b), expected_actions, atol=2e-6, rtol=2e-5)
    p.config.memory_enabled = True
    p.train()
    numerator, count, state = p.forward_step(b)
    (numerator / count).backward()
    assert state[0].update_count.tolist() == [1, 1]
    assert p.model.embedder.connections["0"].memory.fast.w1.grad.abs().sum() > 0
    assert any(
        v.grad is not None and v.grad.abs().sum() > 0
        for v in p.model.embedder.model.vision_tower.parameters()
    )
    # Missing first view must neither produce NaNs nor contribute memory data.
    image_masks = torch.tensor([[False, True], [True, True]])
    hidden, mask, state = p.model.embedder.encode(images, image_masks, prompts)
    assert hidden.isfinite().all()
    assert mask[0].sum() < mask[1].sum()


def test_bfloat16_vlm_keeps_fast_weights_float32(setup, monkeypatch):
    patch_assets(monkeypatch)
    cfg = replace(
        setup.config,
        vlm_dtype="bfloat16",
        enable_gradient_checkpointing=True,
        memory_training="joint",
        training_stage="stage2",
        memory_update_rule="momentum_decay",
    )
    p = setup.policy_class(cfg).train()
    numerator, count, state = p.forward_step(batch())
    (numerator / count).backward()
    assert numerator.isfinite()
    assert all(w.dtype == torch.float32 for s in state for w in s.weights + s.auxiliary)
    assert all(v.grad is None or v.grad.isfinite().all() for v in p.parameters())
    assert p.model.embedder.connections["0"].memory.fast.w1.grad.abs().sum() > 0
    p.eval()
    assert p.predict_action_chunk(batch()).isfinite().all()
