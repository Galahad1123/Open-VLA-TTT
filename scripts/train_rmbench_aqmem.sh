#!/usr/bin/env bash
set -euo pipefail
# Usage: script CHECKPOINT RECIPE RUN_NAME [extra trainer args]
checkpoint=${1:?local baseline or memory checkpoint required}
recipe=${2:?measured recipe JSON required}
run_name=${3:?new run name required}
shift 3
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
exec "${TRAIN_PYTHON:-.venv/bin/python}" -m torch.distributed.run \
  --standalone --nproc_per_node="${NPROC_PER_NODE:-2}" \
  -m lerobot_policy_vla_ttt.training.train_episode_parallel \
  --machine "${MACHINE_CONFIG:-configs/machine.local.toml}" \
  --benchmark rmbench --tasks put_back_block swap_blocks \
  --architecture aqmem --checkpoint "$checkpoint" --recipe "$recipe" \
  --memory-config configs/ttt/aqmem/rmbench/memory.json \
  --run-name "$run_name" --progress plain "$@"
