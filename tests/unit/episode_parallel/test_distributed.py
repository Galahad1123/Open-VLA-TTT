import os
import subprocess
import sys
from pathlib import Path

import torch


def test_cpu_two_ranks_match_one_rank_for_all_variants(tmp_path):
    root = Path(__file__).resolve().parents[3]
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHONPATH=f"{root / 'src'}:{root / 'tests/unit'}",
    )
    for world in (1, 2):
        command = [sys.executable]
        if world == 2:
            command += ["-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2"]
        command += [
            "-m",
            "episode_parallel.distributed_probe",
            "--output",
            str(tmp_path / str(world)),
        ]
        result = subprocess.run(
            command, env=env, cwd=root, capture_output=True, text=True, timeout=240
        )
        assert result.returncode == 0, result.stdout + result.stderr
        for rank in range(world):
            log = (tmp_path / str(world) / f"logged/logs/rank-{rank}.log").read_text()
            assert f"rank {rank} ready" in log
    single = torch.load(tmp_path / "1/rank-0.pt", weights_only=True)
    assert "evo1_groot" in single
    assert single["evo1_groot"]["counts"] == {"policy": 4}
    assert "baseline_stream" in single
    assert "aqmem_joint_mixed" in single
    ranks = [torch.load(tmp_path / f"2/rank-{r}.pt", weights_only=True) for r in range(2)]
    # assert_close handles tensors/numbers, not group metadata strings.
    for case, phase in (("aqmem_joint_mixed", "joint"), ("baseline_stream", "policy")):
        groups = [
            item[case]["control"]["optimizers"][phase].pop("param_groups")
            for item in (single, *ranks)
        ]
        assert groups[0] == groups[1] == groups[2]
        precision = [item[case]["control"].pop("precision") for item in (single, *ranks)]
        assert precision[0] == precision[1] == precision[2]
    torch.testing.assert_close(ranks[0], ranks[1], atol=0, rtol=0)
    torch.testing.assert_close(single, ranks[0], atol=2e-6, rtol=2e-4)


def test_fp16_cpu_two_ranks_overflow_retry_and_resume(tmp_path):
    root = Path(__file__).resolve().parents[3]
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHONPATH=f"{root / 'src'}:{root / 'tests/unit'}",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            "-m",
            "episode_parallel.precision_probe",
            str(tmp_path),
        ],
        env=env,
        cwd=root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
