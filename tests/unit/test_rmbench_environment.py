import importlib.util
from pathlib import Path


def load_environment_script(name):
    path = Path(__file__).parents[2] / "environments/rmbench" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runtime_keeps_mutable_configs_private(tmp_path):
    attach = load_environment_script("prepare_runtime").attach_assets
    assets = tmp_path / "assets"
    robot = assets / "embodiments/aloha-agilex"
    robot.mkdir(parents=True)
    (assets / "objects").mkdir()
    (robot / "meshes").mkdir()
    (robot / "curobo_left.yml").write_text("source")
    runtime = tmp_path / "runtime"
    attach(runtime, assets)
    target = runtime / "assets/embodiments/aloha-agilex/curobo_left.yml"
    assert not target.is_symlink()
    target.write_text("local")
    assert (robot / "curobo_left.yml").read_text() == "source"
    assert (target.parent / "meshes").resolve() == robot / "meshes"


def test_dependency_patch_does_not_modify_uv_cache_hardlink(tmp_path):
    import os

    replace = load_environment_script("patch_dependencies").replace
    cache, installed = tmp_path / "cache.py", tmp_path / "installed.py"
    cache.write_text("original")
    os.link(cache, installed)
    replace(installed, "original", "patched")
    assert installed.read_text() == "patched"
    assert cache.read_text() == "original"


def test_oidn_overlay_preserves_shared_old_library(tmp_path):
    import os

    install = load_environment_script("install_oidn").install_libraries
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    target.mkdir()
    for name in [
        "libOpenImageDenoise",
        "libOpenImageDenoise_core",
        "libOpenImageDenoise_device_cuda",
    ]:
        (source / f"{name}.so.2.3.3").write_text("new")
    cached = tmp_path / "cached.so"
    cached.write_text("old")
    os.link(cached, target / "libOpenImageDenoise.so.2.0.1")
    install(source, target)
    assert cached.read_text() == "old"
    assert (target / "libOpenImageDenoise.so.2.0.1").read_text() == "new"


def test_setup_forces_its_private_uv_environment(tmp_path):
    import os
    import shutil
    import subprocess

    script = Path(__file__).parents[2] / "environments/rmbench/setup.sh"
    shutil.copy2(script, tmp_path / "setup.sh")
    binary = tmp_path / "bin"
    binary.mkdir()
    uv = binary / "uv"
    uv.write_text('#!/bin/sh\nprintf "%s" "$UV_PROJECT_ENVIRONMENT" > "$REVIEW_OUTPUT"\nexit 42\n')
    uv.chmod(0o755)
    output = tmp_path / "observed"
    env = dict(
        os.environ,
        PATH=str(binary) + ":" + os.environ["PATH"],
        UV_PROJECT_ENVIRONMENT="/tmp/foreign-policy-env",
        REVIEW_OUTPUT=str(output),
    )
    result = subprocess.run(["bash", str(tmp_path / "setup.sh")], env=env, capture_output=True)
    assert result.returncode == 42
    assert output.read_text() == str(tmp_path / ".venv")
