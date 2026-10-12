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
            command, env=env, cwd=root, capture_output=True, text=True, timeout=150
        )
        assert result.returncode == 0, result.stdout + result.stderr
    single = torch.load(tmp_path / "1/rank-0.pt", weights_only=True)
    ranks = [torch.load(tmp_path / f"2/rank-{r}.pt", weights_only=True) for r in range(2)]
    torch.testing.assert_close(ranks[0], ranks[1], atol=0, rtol=0)
    torch.testing.assert_close(single, ranks[0], atol=2e-6, rtol=2e-4)
