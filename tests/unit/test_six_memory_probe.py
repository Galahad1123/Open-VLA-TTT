import runpy
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def probe():
    return runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "scripts/probe_robomme_six_memory.py")
    )


def test_probe_refuses_busy_gpu_without_starting_trainer(probe, monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="3271888\n")

    monkeypatch.setattr(probe["subprocess"], "run", run)
    with pytest.raises(RuntimeError, match="active compute"):
        probe["assert_idle"]("0")
    assert len(calls) == 1
    assert calls[0][0] == "nvidia-smi"


def test_log_bundle_excludes_weights_and_optimizer(probe, tmp_path):
    (tmp_path / "aqmem").mkdir()
    (tmp_path / "aqmem/step-00000003").mkdir()
    (tmp_path / "aqmem/step-00000003/model.safetensors").write_bytes(b"not logs")
    (tmp_path / "aqmem/step-00000003/training.pt").write_bytes(b"not logs")
    for name in ["summary.json", "aqmem.log", "aqmem/run.json", "aqmem/metrics.jsonl"]:
        (tmp_path / name).write_text("{}")
    archive = probe["write_log_bundle"](tmp_path)
    with tarfile.open(archive) as bundle:
        assert set(bundle.getnames()) == {
            "summary.json",
            "aqmem.log",
            "aqmem/run.json",
            "aqmem/metrics.jsonl",
        }
