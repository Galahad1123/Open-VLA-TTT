"""Run one repeat-task experiment; parent records completion and probe timing."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    from lerobot_policy_vla_ttt.portability import read_machine

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind", choices=["baseline", "aqmem"], required=True)
    p.add_argument("--machine", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--probe", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    checkpoint = args.checkpoint.expanduser().resolve()
    for name in [
        "model.safetensors",
        "config.json",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    ]:
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(checkpoint / name)
    if json.loads((checkpoint / "config.json").read_text())["type"] != "evo1_groot":
        raise ValueError("Both arms must initialize from the common baseline")
    machine = read_machine(args.machine)
    output = Path(machine["storage"]["runs_root"]) / args.run_name
    if output.exists():
        raise FileExistsError(output)
    recipe = root / f"configs/training/robomme-screen/repeat-two-frozen-{args.kind}.json"
    command = [
        sys.executable,
        "-m",
        "lerobot_policy_vla_ttt.training.train_episode_parallel",
        "--machine",
        str(args.machine.resolve()),
        "--checkpoint",
        str(checkpoint),
        "--architecture",
        "aqmem" if args.kind == "aqmem" else "evo1_groot",
        "--benchmark",
        "robomme",
        "--recipe",
        str(recipe),
        "--run-name",
        args.run_name,
        "--progress",
        "plain",
    ]
    if args.kind == "aqmem":
        command += [
            "--memory-config",
            str(root / "configs/training/robomme-screen/multitask-six-frozen-memory.json"),
        ]
    if args.probe:
        command += ["--stop-after", "30"]
    if args.dry_run:
        print(json.dumps(dict(command=command, output=str(output)), indent=2))
        return
    launchdir = Path(machine["storage"]["runs_root"]) / "launchers"
    launchdir.mkdir(exist_ok=True, parents=True)
    record = launchdir / f"{args.run_name}.json"
    if record.exists():
        raise FileExistsError(record)
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="0",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        TOKENIZERS_PARALLELISM="false",
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHONHASHSEED="0",
        PYTHONUNBUFFERED="1",
        PYTHONPATH=str(root / "src") + ":" + str(root / "packages/eval-client/src"),
    )
    metadata = dict(
        command=command,
        parent_pid=os.getpid(),
        started=time.time(),
        output=str(output),
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        recipe=json.loads(recipe.read_text()),
        state="starting",
    )
    record.write_text(json.dumps(metadata, indent=2))
    with (launchdir / f"{args.run_name}.log").open("x") as log:
        child = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        metadata.update(state="running", child_pid=child.pid)
        record.write_text(json.dumps(metadata, indent=2))
        code = child.wait()
    metadata.update(
        returncode=code, finished=time.time(), state="complete" if code == 0 else "failed"
    )
    record.write_text(json.dumps(metadata, indent=2))
    if code:
        raise SystemExit(code)
    rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    complete = output / f"step-{rows[-1]['step']:08d}/COMPLETE"
    if not complete.exists():
        raise RuntimeError("Missing final COMPLETE marker")
    import numpy as np

    times = [row["update_seconds"] for row in rows[5:]]
    summary = dict(
        updates=len(rows),
        steady_p75_s=float(np.percentile(times, 75)) if times else None,
        steady_median_s=float(np.median(times)) if times else None,
        peak_allocated_gib=max(row.get("peak_allocated_gib", 0) for row in rows),
        peak_reserved_gib=max(row.get("peak_reserved_gib", 0) for row in rows),
        last_save_seconds=max(
            0, complete.stat().st_mtime - (output / "metrics.jsonl").stat().st_mtime
        ),
        elapsed_seconds=metadata["finished"] - metadata["started"],
    )
    (output / "timing_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
