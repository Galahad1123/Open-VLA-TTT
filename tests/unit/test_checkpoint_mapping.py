from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from lerobot.policies.evo1.configuration_evo1 import Evo1Config
from lerobot.policies.evo1.modeling_evo1 import Evo1Policy
from torch import nn

from lerobot_policy_vla_ttt.checkpoint import compare_state_dicts, load_state_dict_strict
from lerobot_policy_vla_ttt.configuration_vla_ttt import VLATTTConfig
from lerobot_policy_vla_ttt.modeling_vla_ttt import VLATTTPolicy


class TinyVLATTTPolicy(VLATTTPolicy):
    def __init__(self, config: VLATTTConfig, **kwargs) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.weight = nn.Parameter(torch.zeros(2, 2))


class TinyEvo1Policy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = Evo1Config(device="cpu", use_amp=False)
        self.weight = nn.Parameter(torch.full((2, 2), 3.0))


def _checkpoint_dir(tmp_path: Path, policy_type: str) -> Path:
    path = tmp_path / policy_type
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"type": policy_type}))
    return path


def test_compatible_state_dict_loads_exact_values() -> None:
    target = nn.Linear(2, 2)
    source = {key: torch.full_like(value, 4.0) for key, value in target.state_dict().items()}

    report = compare_state_dicts(source, target.state_dict())
    load_state_dict_strict(target, source)

    assert report.is_compatible
    assert report.missing_keys == ()
    assert report.unexpected_keys == ()
    assert report.shape_mismatches == ()
    assert all(
        torch.equal(value, torch.full_like(value, 4.0)) for value in target.state_dict().values()
    )


def test_incompatible_state_dict_reports_every_problem_without_partial_load() -> None:
    target = nn.Linear(2, 2)
    before = {key: value.clone() for key, value in target.state_dict().items()}
    source = {
        "weight": torch.zeros(3, 2),
        "unexpected": torch.zeros(1),
    }

    report = compare_state_dicts(source, target.state_dict())
    with pytest.raises(ValueError) as error:
        load_state_dict_strict(target, source)

    message = str(error.value)
    assert report.missing_keys == ("bias",)
    assert report.unexpected_keys == ("unexpected",)
    assert report.shape_mismatches[0].key == "weight"
    assert "bias" in message
    assert "unexpected" in message
    assert "weight" in message
    assert "(3, 2)" in message
    assert "(2, 2)" in message
    assert all(torch.equal(value, before[key]) for key, value in target.state_dict().items())


def test_from_evo1_pretrained_strictly_warm_starts_plugin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = _checkpoint_dir(tmp_path, "evo1")
    source_policy = TinyEvo1Policy()
    calls: list[dict[str, object]] = []

    def fake_load(cls, pretrained_name_or_path, **kwargs):
        calls.append({"path": pretrained_name_or_path, **kwargs})
        return source_policy

    monkeypatch.setattr(Evo1Policy, "from_pretrained", classmethod(fake_load))

    policy = TinyVLATTTPolicy.from_evo1_pretrained(
        source_path,
        revision="abc123",
        local_files_only=True,
    )

    assert isinstance(policy, TinyVLATTTPolicy)
    assert policy.config.type == "vla_ttt"
    assert torch.equal(policy.weight, source_policy.weight)
    assert calls == [
        {
            "path": source_path,
            "revision": "abc123",
            "local_files_only": True,
            "strict": True,
        }
    ]


@pytest.mark.parametrize("loader", ["from_pretrained", "from_evo1_pretrained"])
def test_checkpoint_entry_points_reject_non_strict_loading(
    loader: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_type = "vla_ttt" if loader == "from_pretrained" else "evo1"
    source_path = _checkpoint_dir(tmp_path, checkpoint_type)
    load_calls: list[str] = []

    def fake_load(cls, pretrained_name_or_path, **kwargs):
        load_calls.append(str(pretrained_name_or_path))
        return object()

    monkeypatch.setattr(Evo1Policy, "from_pretrained", classmethod(fake_load))

    with pytest.raises(ValueError, match="strict=True"):
        getattr(TinyVLATTTPolicy, loader)(source_path, strict=False)

    assert load_calls == []


def test_checkpoint_entry_points_reject_wrong_types_before_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plugin_path = _checkpoint_dir(tmp_path, "vla_ttt")
    evo1_path = _checkpoint_dir(tmp_path, "evo1")
    other_path = _checkpoint_dir(tmp_path, "other")
    load_calls: list[str] = []

    def fake_load(cls, pretrained_name_or_path, **kwargs):
        load_calls.append(str(pretrained_name_or_path))
        return object()

    monkeypatch.setattr(Evo1Policy, "from_pretrained", classmethod(fake_load))

    assert TinyVLATTTPolicy.from_pretrained(plugin_path) is not None
    assert load_calls == [str(plugin_path)]

    with pytest.raises(ValueError, match="from_evo1_pretrained"):
        TinyVLATTTPolicy.from_pretrained(evo1_path)
    with pytest.raises(ValueError, match="other"):
        TinyVLATTTPolicy.from_pretrained(other_path)
    with pytest.raises(ValueError, match="vla_ttt"):
        TinyVLATTTPolicy.from_evo1_pretrained(plugin_path)
    with pytest.raises(ValueError, match="other"):
        TinyVLATTTPolicy.from_evo1_pretrained(other_path)

    assert load_calls == [str(plugin_path)]
