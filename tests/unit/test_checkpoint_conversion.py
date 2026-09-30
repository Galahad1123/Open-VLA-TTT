from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from lerobot_policy_vla_ttt import checkpoint, processor_vla_ttt
from lerobot_policy_vla_ttt.checkpoint import convert_evo1_checkpoint
from lerobot_policy_vla_ttt.configuration_vla_ttt import VLATTTConfig
from lerobot_policy_vla_ttt.modeling_vla_ttt import VLATTTPolicy


class FakeArtifact:
    def __init__(self, filename: str, content: str) -> None:
        self.filename = filename
        self.content = content
        self.save_calls: list[Path] = []

    def save_pretrained(self, destination: str | Path) -> None:
        path = Path(destination)
        self.save_calls.append(path)
        (path / self.filename).write_text(self.content)


class FakePolicy(FakeArtifact):
    def __init__(self) -> None:
        super().__init__("model.safetensors", "fake-weights")
        self.config = VLATTTConfig(device="cpu", use_amp=False)

    def save_pretrained(self, destination: str | Path) -> None:
        super().save_pretrained(destination)
        self.config.save_pretrained(destination)


def _source(tmp_path: Path, policy_type: str = "evo1") -> Path:
    source = tmp_path / f"source-{policy_type}"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"type": policy_type}))
    return source


def test_conversion_saves_self_contained_policy_processors_and_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path)
    destination = tmp_path / "converted"
    policy = FakePolicy()
    preprocessor = FakeArtifact("policy_preprocessor.json", "pre")
    postprocessor = FakeArtifact("policy_postprocessor.json", "post")
    processor_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        VLATTTPolicy,
        "from_evo1_pretrained",
        classmethod(lambda cls, *args, **kwargs: policy),
    )
    monkeypatch.setattr(
        processor_vla_ttt,
        "make_vla_ttt_pre_post_processors_from_pretrained",
        lambda *args, **kwargs: processor_calls.append(kwargs) or (preprocessor, postprocessor),
    )

    result = convert_evo1_checkpoint(
        source,
        destination,
        revision="abc123",
        local_files_only=True,
    )

    assert result == destination
    assert len(policy.save_calls) == 1
    assert len(preprocessor.save_calls) == 1
    assert len(postprocessor.save_calls) == 1
    assert processor_calls == [{"revision": "abc123", "local_files_only": True}]
    assert json.loads((destination / "config.json").read_text())["type"] == "vla_ttt"
    metadata = json.loads((destination / "vla_ttt_conversion.json").read_text())
    assert metadata == {
        "format_version": 1,
        "source": str(source),
        "source_policy_type": "evo1",
        "source_revision": "abc123",
        "target_policy_type": "vla_ttt",
    }

    shutil.rmtree(source)
    assert (destination / "model.safetensors").read_text() == "fake-weights"
    assert (destination / "policy_preprocessor.json").read_text() == "pre"
    assert (destination / "policy_postprocessor.json").read_text() == "post"


def test_conversion_rejects_nonempty_destination_without_overwriting(tmp_path: Path) -> None:
    source = _source(tmp_path)
    destination = tmp_path / "converted"
    destination.mkdir()
    marker = destination / "keep.txt"
    marker.write_text("keep")

    with pytest.raises(FileExistsError, match="not empty"):
        convert_evo1_checkpoint(source, destination, local_files_only=True)

    assert marker.read_text() == "keep"


def test_conversion_rejects_wrong_source_before_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path, "other")
    construction_calls: list[str] = []
    monkeypatch.setattr(
        VLATTTPolicy,
        "from_evo1_pretrained",
        classmethod(lambda cls, *args, **kwargs: construction_calls.append("policy")),
    )
    monkeypatch.setattr(
        processor_vla_ttt,
        "make_vla_ttt_pre_post_processors_from_pretrained",
        lambda *args, **kwargs: construction_calls.append("processors"),
    )

    with pytest.raises(ValueError) as error:
        checkpoint.convert_evo1_checkpoint(source, tmp_path / "converted")

    assert "evo1" in str(error.value)
    assert "other" in str(error.value)
    assert construction_calls == []
