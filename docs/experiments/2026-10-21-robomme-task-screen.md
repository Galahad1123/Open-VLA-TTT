# RoboMME task screening: random-chunk baseline

This first implementation prepares the baseline initialization comparison. It does
not implement AQMEM history interventions or launch full training automatically.

## Assigned experiments

| Machine | First run | Initialization |
| --- | --- | --- |
| Remote RTX 5090 32 GB | PickXtimes once, fixed 8-episode overfit subset | RoboMME Stage1 5k |
| Local 72 GB workstation | Identical subset and recipe | RoboMME Stage2 80k |

Both runs train the VLM and action expert (Stage2 parameter flags), using fresh
optimizer/scheduler state, seed 1000, random execution-frame chunks, batch 4,
horizon 20, execution 10, LR 1e-5, warmup 100, 2000 updates, cosine decay to zero.
The inherited checkpoint dropout and architecture remain unchanged. Compare equal
updates, not equal wall time. Low training loss is not evidence of rollout success.

The 8 original episode IDs are 508, 512, 516, 518, 529, 530, 540, 541. They contain
all three target colors. After checking training-scene rollout behavior, the
23-episode recipe is available for a separate run. Do not automatically expand
data solely because the loss fell. Starting the 23-episode run from a new checkpoint
starts a fresh schedule; this CLI does not offer exact resume.

Also provided: VideoUnmask single-target (75 episodes) and PickHighlight
single-target (50 episodes) baseline recipes. These are not instructions to run all
tasks immediately. Selectors were derived from raw task/subgoal annotations;
privileged subgoal fields are never passed to the policy.

## Portability and remote launch

Use the unchanged `uv.lock` and existing RoboMME dataset/VLM. Machine paths belong
in `configs/machine.local.toml`, never in the recipe or checkpoint files:

```toml
[assets]
robomme_initial_checkpoint = "/srv/models/robomme-stage1-5k/pretrained_model"
robomme_dataset = "/srv/datasets/robomme"
vlm = "/srv/models/InternVL3-1B-hf"

[storage]
runs_root = "/srv/experiments/vla-ttt"
hf_home = "/srv/cache/huggingface"
```

Replace these example paths with the existing remote paths. The checkpoint must
be the complete baseline `pretrained_model/` directory including processors, not
a T2MEM checkpoint. `--checkpoint` overrides the machine default. Do not alter
the checkpoint's embedded old machine path; `assets.vlm` relocates it in memory.

First inspect the resolved configuration without GPU access:

```bash
CUDA_VISIBLE_DEVICES='' uv run --frozen python \
  -m lerobot_policy_vla_ttt.training.train_task_subset \
  --machine configs/machine.local.toml \
  --recipe configs/training/robomme-screen/pickxtimes-once-overfit8.json \
  --run-name pickxtimes-once-init5k-dry --dry-run
```

Confirm `source_training_stage=stage1`, the correct checkpoint path, 8 episodes,
and `training_stage=stage2` (trainable parameters, not initialization identity).

Then run a remote GPU probe; local 72 GB measurements do not establish 5090 speed:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
uv run --frozen python \
  -m lerobot_policy_vla_ttt.training.train_task_subset \
  --machine configs/machine.local.toml \
  --recipe configs/training/robomme-screen/pickxtimes-once-overfit8.json \
  --run-name pickxtimes-once-init5k-profile --stop-after 40
```

`--stop-after` preserves the 2000-update schedule but deliberately omits model
checkpoints. The summary reports `profile_complete`, not completed training.
After the probe passes, launch the assigned training in a persistent terminal:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
uv run --frozen python \
  -m lerobot_policy_vla_ttt.training.train_task_subset \
  --machine configs/machine.local.toml \
  --recipe configs/training/robomme-screen/pickxtimes-once-overfit8.json \
  --run-name pickxtimes-once-init5k-run01
```

Use a new run name each time. Checkpoints are saved every 500 updates. To run the
local 80k counterpart, use the same command with an explicit 80k `--checkpoint`
and a distinct run name. For a different GPU memory budget, do not silently
reduce batch for only one initialization arm; revise both recipes consistently.

## Data and measurement guarantees

- Source parquet/video files and their action-query episode boundaries are untouched.
- Only execution frames from selected original episodes are sampled. Demo frames
  and privileged labels never become control supervision.
- Full training execution-only normalization statistics are retained across subsets.
- `run.json` records original IDs, selected frame count, resolved paths and recipe.
- `metrics.jsonl` records loss, gradient norm, LR, per-update time, time between
  successive updates (including loading), CUDA allocated/reserved peaks and timestamps.
- `summary.json` reports status, measured updates, median step time after discarding
  the first five updates, and a 2000-update estimate excluding initialization and evaluation.
- A standard LeRobot epoch may end with a smaller final batch; both comparison arms
  use the same data order and identical behavior. This is the upstream random-chunk
  trainer, not the fixed-chunk stream trainer.
- Allocated/reserved peaks are process-local PyTorch measurements. Concurrent GPU
  board use must additionally be measured with `nvidia-smi`.

No simulator scores are claimed by the training smoke tests. The next gate is
closed-loop action validation, followed by the memory experiments.

## Local GPU measurements (2026-10-21)

RTX PRO 5000 72 GB Blackwell, batch 4, Stage1 5k initialization, 8-episode subset,
40 updates per process. `OMP_NUM_THREADS=2`, two data workers per process. Timings
discard the first five updates; these are short estimates, not full-run timings.

| Concurrent processes | Median seconds/update per process | Combined updates/s | Peak board memory | Estimated 2000 updates per process |
| --- | --- | --- | --- | --- |
| 1 | 0.954 | 1.048 | 18,143 MiB | 31.8 min |
| 2 | 2.037–2.038 | 0.981 | 35,718 MiB | 67.9 min |
| 3 | 3.058–3.063 | 0.980 | 53,294 MiB | 102.0 min |

All six profiling processes completed with finite loss/gradient norms. Each used
about 13.43 GiB peak allocated and 16.72 GiB peak reserved PyTorch memory. The
single-process 40-step mean loss was 0.0931; this is a pipeline check, not evidence
of action success. Board memory includes CUDA contexts and other resident usage.

For this workload, use one training process per GPU: 2–3 concurrent processes fit
but reduce combined throughput by about 6–7%. Two separate machines can run the
initialization arms concurrently. These results do not establish the remote
5090's memory peak or runtime; use its 40-step probe before full training.

Raw local logs and monitor samples are under
`outputs/robomme-screen-profiles/concurrent-*`; the standalone measurement driver
is `outputs/robomme-screen-profiles/measure_concurrency.py` (local paths, not a
portable training entry point).

Validation: 285 unit cases passed across the suite and targeted reruns. Six
existing tests required localhost socket access or the installed CLI PATH and
passed outside the sandbox with CUDA disabled. Both 5k and 80k weight sources
were exercised on GPU. An additional 80k-initialized six-update run saved a
standard checkpoint; loading that checkpoint and its processors for two further
updates passed (fresh optimizer, not exact resume).
