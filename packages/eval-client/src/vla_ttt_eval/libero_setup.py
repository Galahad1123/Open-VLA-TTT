"""Noninteractive LIBERO setup using pre-downloaded, revision-pinned local assets."""

import importlib.util
import json
import os
import sys
from pathlib import Path


def write_config(directory, package_root, assets):
    directory, package_root, assets = map(Path, (directory, package_root, assets))
    for folder in (
        "articulated_objects",
        "stable_scanned_objects",
        "turbosquid_objects",
        "stable_hope_objects",
        "scenes",
    ):
        if not (assets / folder).is_dir():
            raise FileNotFoundError(
                f"Download pinned LIBERO assets first: missing {assets / folder}"
            )
    config = {
        "benchmark_root": str(package_root.resolve()),
        "bddl_files": str((package_root / "bddl_files").resolve()),
        "init_states": str((package_root / "init_files").resolve()),
        "datasets": str((directory / "datasets").resolve()),
        "assets": str(assets.resolve()),
    }
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "config.yaml"
    # JSON is a YAML subset; no PyYAML dependency in the lightweight client.
    if path.exists():
        if path.read_text() != json.dumps(config, indent=2) + "\n":
            raise ValueError(
                f"Existing LIBERO config differs; use a fresh config directory: {path}"
            )
    else:
        with path.open("x") as stream:
            stream.write(json.dumps(config, indent=2) + "\n")
    return path


def configure(assets):
    spec = importlib.util.find_spec("libero")
    if spec is None:
        raise ImportError("Install environments/libero, not the policy environment")
    root = Path(next(iter(spec.submodule_search_locations))) / "libero"
    directory = Path(sys.prefix) / "vla-ttt-libero-config"
    write_config(directory, root, assets)
    os.environ["LIBERO_CONFIG_PATH"] = str(directory)
    import libero.libero

    # hf-libero 0.1.4 ignores config['assets'] inside get_assets_path; fill its cache
    # before importing envs to prevent its unpinned automatic network download.
    libero.libero._assets_path_cache = str(Path(assets).resolve())
