import hashlib
import importlib.util
from pathlib import Path

import pytest


def load_downloader():
    path = Path(__file__).parents[2] / "scripts/download_rmbench.py"
    spec = importlib.util.spec_from_file_location("download_rmbench", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mirror_payload_bypasses_proxy_and_verifies_hash(tmp_path, monkeypatch):
    module = load_downloader()

    def curl(argv, **kwargs):
        assert argv[argv.index("--noproxy") + 1] == "*"
        assert "-4" in argv
        Path(argv[argv.index("--output") + 1]).write_bytes(b"payload")

    monkeypatch.setattr(module.subprocess, "run", curl)
    output = tmp_path / "mesh.glb"
    module.download(
        "objects/mesh.glb",
        output,
        7,
        hashlib.sha256(b"payload").hexdigest(),
        "https://hf-mirror.com",
    )
    assert output.read_bytes() == b"payload"
    assert not output.with_suffix(".glb.partial").exists()


def test_invalid_payload_not_published(tmp_path, monkeypatch):
    module = load_downloader()
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda argv, **kwargs: Path(argv[argv.index("--output") + 1]).write_bytes(b"bad"),
    )
    output = tmp_path / "mesh.glb"
    with pytest.raises(ValueError, match="checksum"):
        module.download("objects/mesh.glb", output, 3, "wrong", "https://hf-mirror.com")
    assert not output.exists()
