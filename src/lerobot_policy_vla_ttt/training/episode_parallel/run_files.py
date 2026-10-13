"""Reserve a run directory and capture each rank's stdout/stderr without shell tee."""

import importlib.metadata
import os
import platform
import subprocess
import sys
import threading
import traceback
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path

from .runtime import collective_error


def environment(accelerator):
    import torch

    root = Path(__file__).resolve().parents[4]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL, timeout=5
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root, text=True, timeout=5
            ).strip()
        )
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, None
    result = {
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": commit,
        "git_dirty": dirty,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "lerobot", "accelerate")
        },
        "cuda_runtime": torch.version.cuda,
        "device": str(accelerator.device),
    }
    if accelerator.device.type == "cuda":
        properties = torch.cuda.get_device_properties(accelerator.device)
        result["gpu"] = {
            "name": properties.name,
            "total_memory": properties.total_memory,
            "capability": [properties.major, properties.minor],
        }
    return result


@contextmanager
def rank_log(path):
    """Tee OS stdout/stderr, including pre-existing logging handlers/native output."""
    sys.stdout.flush()
    sys.stderr.flush()
    streams = []
    lock = threading.Lock()
    with Path(path).open("xb", buffering=0) as log:

        def pump(reader, terminal):
            with os.fdopen(reader, "rb", buffering=0) as source:
                while chunk := source.read(8192):
                    with lock:
                        log.write(chunk)
                    try:
                        os.write(terminal, chunk)
                    except OSError:
                        pass  # A closed terminal must not discard the file log.

        try:
            for target in (1, 2):
                terminal = os.dup(target)
                reader, writer = os.pipe()
                os.dup2(writer, target)
                os.close(writer)
                thread = threading.Thread(target=pump, args=(reader, terminal), daemon=True)
                thread.start()
                streams.append((target, terminal, thread))
            # Python streams may have been redirected separately (e.g. by an
            # embedding caller); route them through the same captured OS pipes.
            with (
                os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8") as out,
                os.fdopen(os.dup(2), "w", buffering=1, encoding="utf-8") as err,
                redirect_stdout(out),
                redirect_stderr(err),
            ):
                try:
                    yield
                except BaseException:
                    traceback.print_exc()
                    raise
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            for target, terminal, _ in streams:
                os.dup2(terminal, target)
            for _, terminal, thread in streams:
                thread.join()
                os.close(terminal)


@contextmanager
def run_directory(output, accelerator):
    error = None
    if accelerator.is_main_process:
        try:
            output.mkdir(parents=True, exist_ok=False)
            (output / "logs").mkdir()
        except OSError as exc:
            error = f"Cannot reserve run directory: {exc}"
    collective_error(error, accelerator.device)
    accelerator.wait_for_everyone()
    with rank_log(output / "logs" / f"rank-{accelerator.process_index}.log"):
        yield
