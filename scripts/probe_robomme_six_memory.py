#!/usr/bin/env python3
"""Serial GPU probes; refuse an occupied GPU and never start a long training run."""

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import tarfile
import time
from pathlib import Path


def assert_idle(gpu, *, allow_busy=False):
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            gpu,
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    processes = []
    for row in csv.reader(result.stdout.splitlines()):
        if not row or not row[0].strip():
            continue
        if len(row) == 1 and row[0].strip().rstrip(".") == "No running processes found":
            continue
        if not row[0].strip().isdigit():
            raise RuntimeError(f"Unrecognized GPU process query output: {row!r}")
        processes.append(", ".join(field.strip() for field in row))
    if processes:
        details = "; ".join(processes)
        if not allow_busy:
            raise RuntimeError(
                f"GPU has active compute processes (PID, name, MiB): {details}. "
                "Even small allocations are blocked by default. If sharing is intentional, "
                "pass --allow-busy-gpu. Existing jobs were not touched."
            )
        print(
            f"Sharing GPU by explicit request (PID, name, MiB): {details}. "
            "Timing may include contention.",
            flush=True,
        )


def write_log_bundle(output):
    """Share only run metadata/logs, never model or optimizer tensors."""
    archive = output / "probe-logs.tar.gz"
    files = sorted(output.glob("*.json")) + sorted(output.glob("*.log"))
    for name in ("baseline", "aqmem"):
        files += [output / name / f for f in ("run.json", "metrics.jsonl")]
    with tarfile.open(archive, "w:gz") as bundle:
        for path in files:
            if path.is_file():
                bundle.add(path, arcname=str(path.relative_to(output)))
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument(
        "--allow-busy-gpu",
        action="store_true",
        help="Explicitly permit sharing with listed compute processes; timings may be affected",
    )
    parser.add_argument(
        "--episode-microbatch",
        type=int,
        choices=(1, 2, 3, 6),
        help="Experimental static episode batch; 1 is its serial reference",
    )
    args = parser.parse_args()
    if args.updates < 2:
        parser.error("Use at least two updates for a steady-state estimate")
    root = Path(__file__).resolve().parents[1]
    checkpoint = args.checkpoint.resolve()
    config = json.loads((checkpoint / "config.json").read_text())
    if (
        config.get("type") != "evo1_groot"
        or config.get("robomme_action_representation") != "delta_joint"
    ):
        parser.error("Expected an evo1_groot delta_joint baseline checkpoint")
    if not (checkpoint / "model.safetensors").is_file():
        parser.error("Checkpoint weights are missing")
    assert_idle(args.gpu, allow_busy=args.allow_busy_gpu)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=args.gpu,
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        TOKENIZERS_PARALLELISM="false",
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + str(root / "packages/eval-client/src")
    recipes = root / "configs/training/robomme-screen"
    hardware = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            args.gpu,
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    (output / "hardware.log").write_text(hardware.stdout)
    results = {
        "complete": False,
        "allow_busy_gpu": args.allow_busy_gpu,
        "episode_microbatch_size": args.episode_microbatch,
    }
    sampled = None
    for architecture in ["evo1_groot", "aqmem"]:
        assert_idle(args.gpu, allow_busy=args.allow_busy_gpu)
        name = "baseline" if architecture == "evo1_groot" else "aqmem"
        recipe = json.loads((recipes / f"multitask-six-frozen-{name}.json").read_text())
        recipe.update(steps=args.updates, warmup_steps=1, save_every=args.updates)
        if args.episode_microbatch is not None:
            recipe["episode_microbatch_size"] = args.episode_microbatch
        recipe_path = output / f"{name}-probe-recipe.json"
        recipe_path.write_text(json.dumps(recipe, indent=2))
        command = [
            sys.executable,
            "-m",
            "lerobot_policy_vla_ttt.training.train_episode_parallel",
            "--machine",
            str(args.machine.resolve()),
            "--checkpoint",
            str(checkpoint),
            "--architecture",
            architecture,
            "--benchmark",
            "robomme",
            "--recipe",
            str(recipe_path),
            "--output",
            str(output / name),
            "--stop-after",
            str(args.updates),
        ]
        if architecture == "aqmem":
            command += ["--memory-config", str(recipes / "multitask-six-frozen-memory.json")]
        print(
            f"Starting {name}: {args.updates} full-episode updates; "
            f"log: {output / (name + '.log')}",
            flush=True,
        )
        started = time.monotonic()
        try:
            with (output / f"{name}.log").open("w") as log:
                subprocess.run(
                    command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
                )
        except subprocess.CalledProcessError as error:
            results["failed_stage"] = name
            results["returncode"] = error.returncode
            (output / "summary.json").write_text(json.dumps(results, indent=2))
            print(f"Probe failed; send {write_log_bundle(output)}", flush=True)
            raise
        wall_seconds = time.monotonic() - started
        metrics = [
            json.loads(line) for line in (output / name / "metrics.jsonl").read_text().splitlines()
        ]
        pairing = [
            (m["sampled_episode_ids"], m["decisions"], m["supervised_scalars"]) for m in metrics
        ]
        if sampled is not None and sampled != pairing:
            raise RuntimeError("Baseline/AQMEM sampling or supervision differs")
        sampled = pairing
        seconds = statistics.median(m["update_seconds"] for m in metrics[1:])
        results[name] = dict(
            total_wall_seconds=wall_seconds,
            steady_seconds_per_update=seconds,
            peak_allocated_gib=max(m["peak_allocated_gib"] for m in metrics),
            peak_reserved_gib=max(m["peak_reserved_gib"] for m in metrics),
            losses=[m["loss"] for m in metrics],
            estimated_1000_update_hours=seconds * 1000 / 3600,
        )
        (output / "summary.json").write_text(json.dumps(results, indent=2))
    results["complete"] = True
    results["paired_updates_for_10h_aqmem"] = int(
        10 * 3600 / results["aqmem"]["steady_seconds_per_update"]
    )
    results["note"] = (
        "Short-run estimate; excludes startup/checkpoint overhead. "
        "Both runs use the AQMEM-derived budget."
    )
    (output / "summary.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"Copy this log bundle back: {write_log_bundle(output)}", flush=True)


if __name__ == "__main__":
    main()
