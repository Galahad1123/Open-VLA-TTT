"""Apply only the compatibility edits in the pinned official install script."""

from importlib.util import find_spec
from pathlib import Path


def replace(path, before, after):
    source = path.read_text()
    if before in source:
        temporary = path.with_suffix(path.suffix + ".vla-ttt-tmp")
        temporary.write_text(source.replace(before, after))
        temporary.replace(path)  # break uv's hardlink to its shared package cache
    elif after not in source:
        raise ValueError(f"Unsupported dependency source: {path}")


def main():
    sapien = Path(find_spec("sapien").origin).parent / "wrapper/urdf_loader.py"
    mplib = Path(find_spec("mplib").origin).parent / "planner.py"
    replace(sapien, 'open(urdf_file, "r")', 'open(urdf_file, "r", encoding="utf-8")')
    replace(sapien, 'open(srdf_file, "r")', 'open(srdf_file, "r", encoding="utf-8")')
    replace(
        mplib,
        "if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:",
        "if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:",
    )


if __name__ == "__main__":
    main()
