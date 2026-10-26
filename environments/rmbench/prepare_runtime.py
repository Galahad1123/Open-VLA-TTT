"""Create a pinned runtime with private robot configs and shared read-only meshes."""

import argparse
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

from vla_ttt_eval.rmbench import UPSTREAM_REVISION


def link(path, target):
    if path.is_symlink() and path.resolve() == target.resolve():
        return
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.symlink_to(target.resolve(), target_is_directory=target.is_dir())


def attach_assets(runtime, assets):
    root = runtime / "assets"
    root.mkdir(parents=True, exist_ok=True)
    link(root / "objects", assets / "objects")
    robots = root / "embodiments"
    if robots.is_symlink():
        if robots.resolve() != (assets / "embodiments").resolve():
            raise ValueError("Unexpected existing embodiment link")
        robots.unlink()
    robots.mkdir(exist_ok=True)
    for source in (assets / "embodiments").iterdir():
        if not source.is_dir():
            continue
        local = robots / source.name
        local.mkdir(exist_ok=True)
        for item in source.iterdir():
            destination = local / item.name
            if item.is_dir():
                link(destination, item)
            else:
                if destination.is_symlink():
                    raise ValueError("Robot configs must be private files")
                shutil.copy2(item, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    args = parser.parse_args()
    machine = args.machine.expanduser().resolve()
    values = tomllib.loads(machine.read_text())["assets"]

    def path(name):
        value = Path(values[name]).expanduser()
        return (machine.parent / value).resolve() if not value.is_absolute() else value.resolve()

    source, assets = path("rmbench_source"), path("rmbench_assets")
    runtime = args.runtime.expanduser().resolve()
    if not (assets / "embodiments/aloha-agilex/config.yml").is_file():
        raise ValueError("Download the pinned assets before preparing a runtime")
    if not runtime.exists():
        subprocess.run(["git", "clone", "--no-local", str(source), str(runtime)], check=True)
        subprocess.run(["git", "-C", str(runtime), "checkout", UPSTREAM_REVISION], check=True)
    revision = subprocess.check_output(
        ["git", "-C", str(runtime), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError("Existing runtime has a different revision")
    attach_assets(runtime, assets)
    subprocess.run(
        [sys.executable, str(runtime / "scripts/update_embodiment_config_path.py")],
        cwd=runtime,
        check=True,
    )
    print(f"Runtime ready: {runtime}")


if __name__ == "__main__":
    main()
