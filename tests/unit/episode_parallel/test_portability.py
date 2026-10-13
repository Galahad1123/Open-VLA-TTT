import argparse
import json
import shutil
from pathlib import Path

import pytest

from lerobot_policy_vla_ttt import portability


def arguments(**overrides):
    return argparse.Namespace(
        **(
            dict(
                machine=None,
                checkpoint=None,
                resume=None,
                dataset_root=None,
                vlm_path=None,
                output=None,
                run_name=None,
            )
            | overrides
        )
    )


def test_machine_paths_follow_config_directory_and_cli_wins(tmp_path, monkeypatch):
    config = tmp_path / "machine.toml"
    config.write_text(
        '[assets]\ninitial_checkpoint="models/base"\n'
        'vlm="models/vlm"\nlibero_dataset="data"\n'
        '[storage]\nruns_root="runs"\nhf_home="cache"\n'
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    args = arguments(machine=config, run_name="trial", dataset_root=Path("override"))
    resolved = portability.resolve_machine(args)
    assert args.checkpoint == tmp_path / "models/base"
    assert args.vlm_path == tmp_path / "models/vlm"
    assert args.dataset_root == elsewhere / "override"
    assert args.output == tmp_path / "runs/trial"
    assert resolved["storage"]["hf_home"] == str(tmp_path / "cache")
    resumed = arguments(machine=config, resume=Path("saved"), run_name="continued")
    portability.resolve_machine(resumed)
    assert resumed.checkpoint is None  # machine defaults cannot conflict with --resume


@pytest.mark.parametrize(
    "text", ['[asset]\nvlm="/model"', '[assets]\nvml="/model"', "[storage]\nruns_root=42"]
)
def test_machine_typos_fail_early(tmp_path, text):
    config = tmp_path / "machine.toml"
    config.write_text(text)
    with pytest.raises(ValueError):
        portability.resolve_machine(arguments(machine=config))


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a/b", ".", ".."])
def test_run_name_cannot_escape_root(tmp_path, name):
    config = tmp_path / "machine.toml"
    config.write_text('[storage]\nruns_root="runs"')
    with pytest.raises(ValueError, match="run.name"):
        portability.resolve_machine(arguments(machine=config, run_name=name))


def test_content_identity_survives_copy_but_detects_payload_changes(tmp_path):
    root = tmp_path / "first"
    (root / "data").mkdir(parents=True)
    (root / "meta").mkdir()
    (root / "meta/info.json").write_text('{"fps":10}')
    (root / "data/frames.parquet").write_bytes(b"frames")
    moved = tmp_path / "second"
    shutil.copytree(root, moved)
    before = portability.content_digest(root)
    assert portability.content_digest(moved) == before
    (moved / "data/frames.parquet").write_bytes(b"changed")
    assert portability.content_digest(moved) != before


def test_directory_symlinks_hash_payload_and_reject_cycles(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "info.json").write_text("{}")
    payload = tmp_path / "payload"
    payload.mkdir()
    (payload / "frames.parquet").write_bytes(b"original")
    (root / "data").symlink_to(payload, target_is_directory=True)
    copy = tmp_path / "copy"
    shutil.copytree(root, copy)
    before = portability.content_digest(root)
    assert portability.content_digest(copy) == before
    (payload / "frames.parquet").write_bytes(b"changed")
    assert portability.content_digest(root) != before
    (payload / "loop").symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="cycle"):
        portability.content_digest(root)


def test_vlm_override_does_not_edit_checkpoint(tmp_path):
    from lerobot_policy_vla_ttt.models.evo1_groot.configuration_evo1_groot import Evo1GrootConfig

    cfg = Evo1GrootConfig(device="cpu", vlm_model_name="/old/machine/vlm")
    cfg.save_pretrained(tmp_path / "checkpoint")
    target = tmp_path / "vlm"
    target.mkdir()
    updated = portability.relocate_vlm(cfg, target)
    assert updated.vlm_model_name == str(target)
    assert cfg.vlm_model_name == "/old/machine/vlm"
    saved = json.loads((tmp_path / "checkpoint/config.json").read_text())
    assert saved["vlm_model_name"] == "/old/machine/vlm"
    with pytest.raises(ValueError, match="VLM"):
        portability.relocate_vlm(cfg, tmp_path / "missing")


def test_rank_log_keeps_python_native_output_and_exception(tmp_path):
    import os
    import sys

    from lerobot_policy_vla_ttt.training.episode_parallel.run_files import rank_log

    original = sys.stdout
    with pytest.raises(RuntimeError, match="sample failure"):
        with rank_log(tmp_path / "rank.log"):
            print("python output")
            os.write(2, b"native error\n")
            raise RuntimeError("sample failure")
    log = (tmp_path / "rank.log").read_text()
    assert "python output" in log and "native error" in log
    assert "RuntimeError: sample failure" in log
    assert sys.stdout is original
    with pytest.raises(FileExistsError):
        with rank_log(tmp_path / "rank.log"):
            pass


@pytest.mark.parametrize("override", [False, True])
def test_robomme_machine_uses_own_assets(tmp_path, override):
    config = tmp_path / "machine.toml"
    config.write_text(
        '[assets]\ninitial_checkpoint="libero-model"\nlibero_dataset="libero"\n'
        'robomme_initial_checkpoint="robomme-model"\nrobomme_dataset="robomme"\n'
    )
    args = arguments(machine=config, benchmark="robomme")
    if override:
        args.checkpoint = tmp_path / "explicit-model"
        args.dataset_root = tmp_path / "explicit-data"
    portability.resolve_machine(args)
    assert args.checkpoint == tmp_path / ("explicit-model" if override else "robomme-model")
    assert args.dataset_root == tmp_path / ("explicit-data" if override else "robomme")


def test_robomme_does_not_fall_back_to_libero_assets(tmp_path):
    config = tmp_path / "machine.toml"
    config.write_text('[assets]\ninitial_checkpoint="libero-model"\nlibero_dataset="libero"\n')
    args = arguments(machine=config, benchmark="robomme")
    portability.resolve_machine(args)
    assert args.checkpoint is None
    assert args.dataset_root is None
