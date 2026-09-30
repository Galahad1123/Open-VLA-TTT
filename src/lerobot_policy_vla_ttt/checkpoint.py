"""Strict checkpoint inspection and state-dict compatibility helpers."""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.constants import CONFIG_NAME
from torch import Tensor, nn


@dataclass(frozen=True)
class TensorShapeMismatch:
    """One tensor whose source and target shapes differ."""

    key: str
    source_shape: tuple[int, ...]
    target_shape: tuple[int, ...]


@dataclass(frozen=True)
class StateDictCompatibility:
    """Complete compatibility report for two state dictionaries."""

    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    shape_mismatches: tuple[TensorShapeMismatch, ...]

    @property
    def is_compatible(self) -> bool:
        return not (self.missing_keys or self.unexpected_keys or self.shape_mismatches)


def compare_state_dicts(
    source: Mapping[str, Tensor], target: Mapping[str, Tensor]
) -> StateDictCompatibility:
    """Compare all keys and tensor shapes without mutating the target."""
    source_keys = set(source)
    target_keys = set(target)
    shared_keys = sorted(source_keys & target_keys)
    mismatches = tuple(
        TensorShapeMismatch(
            key=key,
            source_shape=tuple(source[key].shape),
            target_shape=tuple(target[key].shape),
        )
        for key in shared_keys
        if source[key].shape != target[key].shape
    )
    return StateDictCompatibility(
        missing_keys=tuple(sorted(target_keys - source_keys)),
        unexpected_keys=tuple(sorted(source_keys - target_keys)),
        shape_mismatches=mismatches,
    )


def _format_incompatibility(report: StateDictCompatibility) -> str:
    lines = ["Checkpoint state dict is incompatible with the target policy:"]
    if report.missing_keys:
        lines.append(f"  missing keys: {', '.join(report.missing_keys)}")
    if report.unexpected_keys:
        lines.append(f"  unexpected keys: {', '.join(report.unexpected_keys)}")
    if report.shape_mismatches:
        lines.append("  shape mismatches:")
        lines.extend(
            f"    {item.key}: source {item.source_shape}, target {item.target_shape}"
            for item in report.shape_mismatches
        )
    return "\n".join(lines)


def load_state_dict_strict(target: nn.Module, source: Mapping[str, Tensor]) -> None:
    """Validate completely, then load atomically with PyTorch strict mode."""
    report = compare_state_dicts(source, target.state_dict())
    if not report.is_compatible:
        raise ValueError(_format_incompatibility(report))
    target.load_state_dict(source, strict=True)


def checkpoint_policy_type(
    pretrained_name_or_path: str | Path,
    *,
    revision: str | None = None,
    local_files_only: bool = False,
    cache_dir: str | Path | None = None,
    token: str | bool | None = None,
) -> str:
    """Read only a checkpoint's serialized policy type."""
    source = Path(pretrained_name_or_path)
    if source.is_dir():
        config_path = source / CONFIG_NAME
    else:
        config_path = Path(
            hf_hub_download(
                repo_id=str(pretrained_name_or_path),
                filename=CONFIG_NAME,
                revision=revision,
                local_files_only=local_files_only,
                cache_dir=cache_dir,
                token=token,
            )
        )
    if not config_path.is_file():
        raise FileNotFoundError(f"{CONFIG_NAME} not found in {pretrained_name_or_path}")
    with config_path.open() as config_file:
        config = json.load(config_file)
    policy_type = config.get("type")
    if not isinstance(policy_type, str) or not policy_type:
        raise ValueError(f"Missing or invalid 'type' in {config_path}")
    return policy_type


def convert_evo1_checkpoint(
    source: str | Path,
    destination: Path,
    *,
    revision: str | None = None,
    local_files_only: bool = False,
) -> Path:
    """Convert a complete Evo-1 checkpoint into a self-contained plugin checkpoint."""
    from .modeling_vla_ttt import VLATTTPolicy
    from .processor_vla_ttt import make_vla_ttt_pre_post_processors_from_pretrained

    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Destination {destination} is not empty")

    source_type = checkpoint_policy_type(
        source,
        revision=revision,
        local_files_only=local_files_only,
    )
    if source_type != "evo1":
        raise ValueError(f"Checkpoint conversion requires source type 'evo1', got {source_type!r}.")

    policy = VLATTTPolicy.from_evo1_pretrained(
        source,
        revision=revision,
        local_files_only=local_files_only,
    )
    preprocessor, postprocessor = make_vla_ttt_pre_post_processors_from_pretrained(
        policy.config,
        source,
        revision=revision,
        local_files_only=local_files_only,
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        policy.save_pretrained(staging)
        preprocessor.save_pretrained(staging)
        postprocessor.save_pretrained(staging)
        metadata = {
            "format_version": 1,
            "source": str(source),
            "source_policy_type": source_type,
            "source_revision": revision,
            "target_policy_type": "vla_ttt",
        }
        (staging / "vla_ttt_conversion.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        if destination.exists():
            destination.rmdir()
        staging.replace(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination
