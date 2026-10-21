import shutil
import subprocess
import tarfile
from pathlib import Path


def test_sdist_excludes_models_datasets_and_virtual_environments(tmp_path):
    root = Path(__file__).resolve().parents[2]
    shutil.copy(root / "pyproject.toml", tmp_path)
    shutil.copy(root / "README.md", tmp_path)
    shutil.copytree(
        root / "environments/v100",
        tmp_path / "environments/v100",
        ignore=shutil.ignore_patterns(".venv"),
    )
    # A source export need not carry VCS ignore metadata; release safety must
    # come from the build's own selection rules, not the checkout's .gitignore.
    shutil.copytree(root / "src", tmp_path / "src")
    for name in (
        "artifacts/model.safetensors",
        "outputs/data.parquet",
        "environments/libero/.venv/private-data",
        "third_party/foreign.py",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"must not ship")
    subprocess.run(
        ["uv", "build", "--offline", "--sdist", "--project", str(tmp_path)],
        check=True,
        capture_output=True,
    )
    with tarfile.open(next((tmp_path / "dist").glob("*.tar.gz"))) as archive:
        members = archive.getnames()
    assert any("src/lerobot_policy_vla_ttt/evaluation/server.py" in name for name in members)
    for name in ("pyproject.toml", "uv.lock"):
        assert any(f"environments/v100/{name}" in entry for entry in members)
    assert not any(
        "must not ship" in name
        or name.endswith(".safetensors")
        or "/.venv/" in name
        or "/outputs/" in name
        or "/third_party/" in name
        for name in members
    )
