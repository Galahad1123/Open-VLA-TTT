"""Overlay official OIDN 2.3.3 for Blackwell while retaining SAPIEN's OIDN renderer."""

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
from importlib.util import find_spec
from pathlib import Path


def install_libraries(source, target):
    for stem in (
        "libOpenImageDenoise",
        "libOpenImageDenoise_core",
        "libOpenImageDenoise_device_cuda",
    ):
        if not (source / f"{stem}.so.2.3.3").is_file():
            raise ValueError(f"Missing required OIDN library: {stem}")
    target.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.is_dir():
            continue
        destination = target / item.name
        temporary = destination.with_name(destination.name + ".vla-ttt-tmp")
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
        if item.is_symlink():
            temporary.symlink_to(item.readlink())
        else:
            shutil.copy2(item, temporary)
        temporary.replace(destination)
    # SAPIEN 3.0.0b1 requests these exact names. Never modify shared uv-cache inodes.
    for stem in ("libOpenImageDenoise", "libOpenImageDenoise_core"):
        alias = target / f"{stem}.so.2.0.1"
        temporary = alias.with_name(alias.name + ".vla-ttt-tmp")
        temporary.symlink_to(f"{stem}.so.2.3.3")
        temporary.replace(alias)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Use an already downloaded official archive")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "oidn-package.json").read_text())
    archive = args.archive or root / "vendor/oidn-2.3.3.x86_64.linux.tar.gz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        if args.archive:
            raise FileNotFoundError(archive)
        partial = archive.with_suffix(".partial")
        for url in (manifest["url"], "https://ghfast.top/" + manifest["url"]):
            result = subprocess.run(
                [
                    "curl",
                    "--noproxy",
                    "*",
                    "-4",
                    "-fL",
                    "--connect-timeout",
                    "15",
                    "--max-time",
                    "180",
                    "--continue-at",
                    "-",
                    "-o",
                    str(partial),
                    url,
                ]
            )
            if result.returncode == 0:
                partial.rename(archive)
                break
        else:
            raise RuntimeError("OIDN download failed; rerun to resume or pass --archive")
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != manifest["sha256"] or archive.stat().st_size != manifest["size"]:
        raise ValueError("OIDN archive checksum/size differs from pinned package")
    destination = root / "vendor/oidn"
    with tarfile.open(archive) as tar:
        tar.extractall(destination, filter="data")
    package = Path(find_spec("sapien").origin).parent
    install_libraries(destination / "oidn-2.3.3.x86_64.linux/lib", package / "oidn_library")
    print("Installed official OIDN 2.3.3 Blackwell runtime into the isolated SAPIEN environment")


if __name__ == "__main__":
    main()
