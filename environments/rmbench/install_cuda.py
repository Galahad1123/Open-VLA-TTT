"""Install the pinned user-local CUDA compiler without touching system CUDA."""

import hashlib
import json
import platform
import subprocess
import tarfile
import zipfile
from pathlib import Path

import zstandard


def main():
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("This toolchain manifest is for Linux x86_64")
    root = Path(__file__).resolve().parent
    prefix = root / "toolkit/cuda-12.8"
    cache = root / "toolkit/packages"
    cache.mkdir(parents=True, exist_ok=True)
    for item in json.loads((root / "cuda-packages.json").read_text()):
        archive = cache / Path(item["path"]).name
        if not archive.is_file():
            partial = archive.with_suffix(".partial")
            subprocess.run(
                [
                    "curl",
                    "--noproxy",
                    "*",
                    "-4",
                    "-fL",
                    "--retry",
                    "3",
                    "--connect-timeout",
                    "15",
                    "--continue-at",
                    "-",
                    "-o",
                    str(partial),
                    "https://conda.anaconda.org/nvidia/" + item["path"],
                ],
                check=True,
            )
            partial.rename(archive)
        with archive.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != item["sha256"]:
            raise ValueError(f"CUDA package checksum mismatch: {archive}")
        with zipfile.ZipFile(archive) as bundle:
            payload = next(name for name in bundle.namelist() if name.startswith("pkg-"))
            with bundle.open(payload) as compressed:
                with zstandard.ZstdDecompressor().stream_reader(compressed) as stream:
                    with tarfile.open(fileobj=stream, mode="r|") as tar:
                        tar.extractall(prefix, filter="data")
    for name, destination in [
        ("include", "targets/x86_64-linux/include"),
        ("lib64", "targets/x86_64-linux/lib"),
    ]:
        if not (prefix / name).exists():
            (prefix / name).symlink_to(destination, target_is_directory=True)
    subprocess.run([str(prefix / "bin/nvcc"), "--version"], check=True)


if __name__ == "__main__":
    main()
