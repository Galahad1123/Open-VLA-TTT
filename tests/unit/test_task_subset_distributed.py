import os
import subprocess
import sys
from pathlib import Path

import torch


def test_cpu_gloo_task_subset_global_batch_matches_single_process(tmp_path):
    root = Path(__file__).resolve().parents[2]
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
        command += ["-m", "task_subset_distributed_probe", "--output", str(tmp_path / str(world))]
        result = subprocess.run(
            command, env=env, cwd=root, capture_output=True, text=True, timeout=120
        )
        assert result.returncode == 0, result.stdout + result.stderr

    single = torch.load(tmp_path / "1/rank-0.pt", weights_only=True)
    ranks = [torch.load(tmp_path / f"2/rank-{rank}.pt", weights_only=True) for rank in range(2)]
    assert any(
        not torch.equal(single["initial"][key], weight) for key, weight in single["weights"].items()
    )
    for result in (single, *ranks):
        assert result["world_size"] * result["local_batch"] == 12
        assert result["updates"] == 8
        assert result["samples"] == 96
        assert result["scheduler_steps"] == list(range(1, 9))
        assert result["skipped_updates"] == [0] * 8
        assert result["learning_rates"] == single["learning_rates"]
        assert result["learning_rates"][-1] < result["learning_rates"][0]
        torch.testing.assert_close(result["weights"], single["weights"], atol=1e-12, rtol=1e-12)
    assert single["world_size"] == 1
    assert all(rank["world_size"] == 2 for rank in ranks)
    torch.testing.assert_close(ranks[0]["weights"], ranks[1]["weights"], atol=0, rtol=0)

    for step, expected_ids in enumerate(single["ids"]):
        global_ids = [sample for rank in ranks for sample in rank["ids"][step]]
        assert all(len(rank["ids"][step]) == 6 for rank in ranks)
        assert len(global_ids) == len(set(global_ids)) == 12
        assert global_ids == expected_ids
    for start in (0, 4):
        assert sorted(
            sample for batch in single["ids"][start : start + 4] for sample in batch
        ) == list(range(48))
    assert single["ids"][:4] != single["ids"][4:]
