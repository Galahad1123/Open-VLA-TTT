"""Shared pinned RMBench protocol and local resource validation."""

import importlib.util
import subprocess
from pathlib import Path

from vla_ttt_eval.rmbench import CAMERAS, DATA_REVISION, TASKS, UPSTREAM_REVISION, resolve_tasks

__all__ = ["CAMERAS", "DATA_REVISION", "TASKS", "UPSTREAM_REVISION", "resolve_tasks", "load_codec"]


def load_codec(source):
    source = Path(source).resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError(f"RMBench source must be {UPSTREAM_REVISION}, got {revision}")
    path = source / "data/decode_image_bit.py"
    expected = subprocess.check_output(
        ["git", "-C", str(source), "show", "HEAD:data/decode_image_bit.py"]
    )
    if path.read_bytes() != expected:
        raise ValueError("Official image codec has uncommitted changes")
    spec = importlib.util.spec_from_file_location("_rmbench_official_codec", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
