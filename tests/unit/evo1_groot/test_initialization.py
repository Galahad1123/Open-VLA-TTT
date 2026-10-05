from pathlib import Path
from types import SimpleNamespace

import pytest
from lerobot.configs.policies import PreTrainedConfig
from test_policy_integration import stats

from lerobot_policy_vla_ttt.baselines import initialize_groot


@pytest.mark.parametrize("benchmark, dimension", [("libero", 7), ("robomme", 8)])
def test_presets_load_as_independent_factory_configs(benchmark, dimension):
    root = Path(__file__).resolve().parents[3]
    cfg = PreTrainedConfig.from_pretrained(root / "configs/baselines/evo1_groot" / benchmark)
    assert cfg.type == "evo1_groot"
    assert cfg.output_features["action"].shape == (dimension,)
    assert cfg.finetune_action_head and not cfg.finetune_vlm


def test_initialize_saves_fresh_policy_stats_and_provenance(config, tmp_path, monkeypatch):
    import json

    source = tmp_path / "preset"
    config.save_pretrained(source)
    vlm = tmp_path / "vlm"
    vlm.mkdir()
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    metadata = SimpleNamespace(
        features={
            "action": {"shape": (7,)},
            "observation.state": {"shape": (8,)},
            "observation.images.image": {"shape": (3, 16, 16)},
            "observation.images.image2": {"shape": (3, 16, 16)},
        },
        stats=stats(),
    )
    monkeypatch.setattr(initialize_groot, "LeRobotDatasetMetadata", lambda *a, **kw: metadata)
    result = initialize_groot.initialize(
        source, dataset, "test/local", "libero", vlm, tmp_path / "new", device="cpu"
    )
    restored = PreTrainedConfig.from_pretrained(result)
    assert restored.type == "evo1_groot"
    assert restored.vlm_model_name == str(vlm.resolve())
    manifest = json.loads((result / "baseline_provenance.json").read_text())
    assert manifest["action_head_initialization"] == "random"
    assert manifest["trained_on_target"] is False
    assert (result / "policy_preprocessor.json").exists()
    with pytest.raises(ValueError, match="destination"):
        initialize_groot.initialize(
            source, dataset, "test/local", "libero", vlm, result, device="cpu"
        )


def test_dataset_schema_rejects_wrong_control_space():
    with pytest.raises(ValueError, match="action"):
        initialize_groot.validate_dataset_schema({"action": {"shape": [7]}}, "robomme")
