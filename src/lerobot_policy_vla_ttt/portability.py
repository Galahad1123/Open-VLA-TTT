"""Machine-local paths and content identities, shared by training and serving."""

import dataclasses
import hashlib
import os
import sys
import tomllib
from pathlib import Path

MACHINE_FIELDS = {
    "assets": {
        "initial_checkpoint",
        "vlm",
        "libero_dataset",
        "robomme_initial_checkpoint",
        "robomme_dataset",
        "rmbench_source",
        "rmbench_assets",
        "rmbench_raw",
        "rmbench_dataset",
    },
    "storage": {"runs_root", "hf_home"},
}


def read_machine(path):
    if path is None:
        return {}
    path = Path(path).expanduser().resolve()
    with path.open("rb") as stream:
        values = tomllib.load(stream)
    if set(values) - MACHINE_FIELDS.keys():
        raise ValueError(f"Unknown machine sections: {sorted(set(values) - MACHINE_FIELDS.keys())}")
    resolved = {}
    for section, fields in values.items():
        if not isinstance(fields, dict) or set(fields) - MACHINE_FIELDS[section]:
            raise ValueError(f"Unknown fields in machine [{section}]")
        resolved[section] = {}
        for name, value in fields.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"machine {section}.{name} must be a nonempty path string")
            item = Path(value).expanduser()
            resolved[section][name] = str(
                (item if item.is_absolute() else path.parent / item).resolve()
            )
    return resolved


def resolve_machine(args):
    """CLI paths are relative to cwd; TOML paths are relative to the TOML file."""
    machine = read_machine(args.machine)
    assets, storage = machine.get("assets", {}), machine.get("storage", {})
    benchmark = getattr(args, "benchmark", None) or "libero"
    defaults = {"dataset_root": assets.get(f"{benchmark}_dataset"), "vlm_path": assets.get("vlm")}
    if not args.resume:
        defaults["checkpoint"] = assets.get(
            "robomme_initial_checkpoint" if benchmark == "robomme" else "initial_checkpoint"
        )
    for name, default in defaults.items():
        if getattr(args, name) is None and default is not None:
            setattr(args, name, Path(default))
    if args.run_name is not None:
        if args.run_name in ("", ".", "..") or Path(args.run_name).name != args.run_name:
            raise ValueError("run-name must be a single directory name")
        if args.output is not None:
            raise ValueError("Use either --output or --run-name")
        if "runs_root" not in storage:
            raise ValueError("--run-name requires storage.runs_root in --machine")
        args.output = Path(storage["runs_root"]) / args.run_name
    for name in ("checkpoint", "resume", "dataset_root", "vlm_path", "output"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, Path(value).expanduser().resolve())
    return machine


def configure_cache(machine, module, argv):
    """Restart once so HF libraries see HF_HOME before their import-time constants.

    The plugin's __init__ imports Transformers before a module CLI can parse TOML.
    Re-exec preserves Accelerate rank environment and avoids patching library globals.
    """
    home = machine.get("storage", {}).get("hf_home")
    if home is not None and os.environ.get("HF_HOME") != home:
        env = dict(os.environ, HF_HOME=home)
        os.execvpe(sys.executable, [sys.executable, "-m", module, *argv], env)


def relocate_vlm(config, path):
    if path is None:
        return config
    path = Path(path).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"VLM directory does not exist: {path}")
    return dataclasses.replace(config, vlm_model_name=str(path))


def local_vlm(config):
    """Resolve the supported Hub name to its pinned, already-downloaded snapshot."""
    path = Path(config.vlm_model_name).expanduser()
    if path.is_dir():
        return relocate_vlm(config, path)
    if config.vlm_model_name == "OpenGVLab/InternVL3-1B-hf":
        from huggingface_hub import snapshot_download

        path = snapshot_download(
            config.vlm_model_name,
            revision="014c0583a0d4bedf29fbe2dbff4f865eb998e171",
            local_files_only=True,
        )
        return relocate_vlm(config, path)
    raise ValueError(
        f"VLM asset unavailable: {config.vlm_model_name}; set --vlm-path or assets.vlm"
    )


def content_digest(root):
    """Hash relative filenames and bytes, excluding hidden cache/transfer metadata.

    Follow file/directory symlinks; reject cycles and never hash absolute paths.
    This deliberately streams complete assets, not just their size or mtime.
    """
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"Asset directory does not exist: {root}")
    digest, count = hashlib.sha256(), 0

    def files(directory, ancestors):
        stat = directory.stat()
        inode = (stat.st_dev, stat.st_ino)
        if inode in ancestors:
            raise ValueError(f"Asset directory symlink cycle: {directory}")
        for path in sorted(directory.iterdir()):
            if path.name.startswith("."):
                continue
            if path.is_symlink() and not path.exists():
                raise ValueError(f"Broken asset symlink: {path}")
            if path.is_dir():
                yield from files(path, ancestors | {inode})
            elif path.is_file():
                yield path

    for path in files(root, set()):
        relative = path.relative_to(root)
        count += 1
        digest.update(relative.as_posix().encode() + b"\0")
        with path.open("rb") as stream:
            digest.update(hashlib.file_digest(stream, "sha256").digest())
    if not count:
        raise ValueError(f"Empty asset directory: {root}")
    return digest.hexdigest()
