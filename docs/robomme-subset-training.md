# RoboMME random-chunk subset training on V100

`train_task_subset` fine-tunes an evo1-groot checkpoint with the standard LeRobot
random execution-frame chunk trainer. It supports single GPU and DDP. It is not
the episode/stream trainer and has no memory module or demo-video prefix input.
All VLM/action parameters are fine-tuned (Stage2 flags), regardless of the source
checkpoint's stage. Initialization loads weights and processors only, with a fresh
optimizer and schedule; this entrypoint does not implement exact resume.

## Recipes and batch semantics

Recipes under `configs/training/robomme-screen/` contain original RoboMME episode
IDs. `batch_size` always means **global batch**. A value of 12 on four processes
uses three samples per rank. It must be divisible by WORLD_SIZE; eight processes cannot
use global batch 12. Do not change it to 16 merely to fill eight GPUs if matching
an existing batch-12 experiment is the goal.

| Recipe | Episodes | Steps | Global batch |
| --- | ---: | ---: | ---: |
| `pickxtimes-once-b12-20k.json` | 23 single-cycle | 20000 | 12 |
| `videounmask-single-b12-20k.json` | 75 single-target | 20000 | 12 |
| `pickhighlight-single-b12-20k.json` | 50 single-target | 20000 | 12 |
| `pickxtimes-up-to-two-b12-20k.json` | 48 one/two-cycle | 20000 | 12 |

These recipes use LR 1e-5, 100 warmup updates, cosine decay to zero, seed 1000,
and save every 500 updates. Legacy 8-episode screening and batch-4 recipes are
also retained. The 48-episode experiment changes both data coverage and counting
difficulty, so report one-cycle and two-cycle evaluation separately.

Only execution frames are sampled. Full-training execution-only normalization
statistics remain unchanged. Source episode IDs and compact sampler indices are
kept separate. LeRobot shuffles chunk start frames and Accelerate shards the
permutation across ranks; independently sampled episodes are not the batch unit.
The inherited epoch-tail behavior is unchanged: single GPU may have a smaller
last batch, while distributed Accelerate pads the tail. Most batches have the
configured global size; this is not exact sample-order equivalence across world
sizes at epoch boundaries.

## V100 setup and launch

Use the environment and hardware checks in [v100-training.md](v100-training.md).
The existing machine TOML supplies `assets.robomme_dataset`, `assets.vlm`,
`assets.robomme_initial_checkpoint`, and `storage.runs_root`. Point the checkpoint
to the RoboMME Stage1 5k **pretrained_model directory**, including processors.
`--checkpoint` may override it. Recipes contain no machine-specific paths.

First resolve the configuration without GPU use:

```bash
CUDA_VISIBLE_DEVICES='' WORLD_SIZE=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run --project environments/v100 --frozen python \
  -m lerobot_policy_vla_ttt.training.train_task_subset \
  --machine configs/machine.local.toml --precision fp16 \
  --recipe configs/training/robomme-screen/pickhighlight-single-b12-20k.json \
  --run-name pickhighlight-init5k-b12-20k-check --dry-run
```

Verify 50 episodes, global batch 12, per-rank batch 3, and
`source_training_stage=stage1`. `training_stage=stage2` is expected.

Four-card short test (new output name):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false \
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run --project environments/v100 --frozen \
python -m torch.distributed.run --standalone --nproc-per-node=4 \
  -m lerobot_policy_vla_ttt.training.train_task_subset \
  --machine configs/machine.local.toml --precision fp16 \
  --recipe configs/training/robomme-screen/pickhighlight-single-b12-20k.json \
  --run-name pickhighlight-init5k-b12-20k-profile --stop-after 60
```

For full training remove `--stop-after 60` and use a new run name such as
`pickhighlight-init5k-b12-20k-run01`. For the second independent experiment use
`CUDA_VISIBLE_DEVICES=4,5,6,7`, recipe `pickxtimes-up-to-two-b12-20k.json`, and a
separate run name. `--standalone` allocates an independent rendezvous endpoint.
Keep num_workers modest: the recipe's two workers are **per rank**.

## Precision, outputs, and acceptance

`--precision fp16` selects model FP16 autocast plus Accelerate's one GradScaler;
FP32 parameter/optimizer storage is retained. Do not add a second scaler or use
`.half()` on the model. `--precision fp32` is the diagnostic reference and `bf16`
is the legacy default for newer GPUs. This path does not use the episode trainer's
custom scaler. FP16 gradient overflow retries the same batch with restored torch
RNG, without advancing the update count or LR. Persistent overflow aborts;
nonfinite forward loss is an error. Failed attempts do not pollute metric averages.

Only rank zero writes shared run metadata, metrics, summary and checkpoints.
`run.json` records precision, world size, global/per-rank batch and dataset audit.
`metrics.jsonl` records cross-rank mean loss, maximum rank timings/memory,
cumulative skipped updates and loss scale. A profile retains the full recipe LR
schedule and writes no checkpoints. Its summary estimates recipe runtime excluding
startup, saving and evaluation. Profiling completes with status `profile_complete`.

The previously successful four-V100 AQMEM episode smoke is useful infrastructure
evidence, but does not validate this random-chunk path. Before a full launch run
single-, two-, and four-card random-chunk profiles. Compare finite losses, successful
updates, per-rank batch and peak memory; then verify a saved checkpoint can initialize
a new run. CPU/Gloo integration checks data sharding and equal-batch updates only;
it cannot establish real V100 FP16 stability or throughput.
