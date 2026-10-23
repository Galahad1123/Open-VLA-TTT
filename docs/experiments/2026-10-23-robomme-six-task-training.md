# Six-task RoboMME action-capability training

This single-GPU experiment expands task/scene coverage while keeping the
memoryless evo1-groot architecture and random action chunks. It does not use
AQMEM/T2MEM, demo prefixes or privileged subgoal inputs. No augmentation is enabled.
The previous screening recipes remain available unchanged.

## Data and budget

| Task selector | Original episodes | Execution frames |
| --- | ---: | ---: |
| PickXtimes: one pickup/place cycle | 23 | 6995 |
| VideoUnmask: one target container | 75 | 8442 |
| PickHighlight: one target cube | 50 | 11438 |
| ButtonUnmask: one target container | 75 | 17170 |
| BinFill: one deposited object | 20 | 5861 |
| SwingXtimes: one right/left traversal | 23 | 7730 |
| Total | 266 | 57636 |

IDs in the recipes were selected using execution-frame subgoal annotations, not
blindly by difficulty tags. These annotations select episodes only; they are not
model inputs. Action execution and correct target selection must be evaluated
separately because several tasks still contain temporary/hidden cues.

`task_groups` partition the selected original IDs into six task families, not
instruction-string variants. Sampling is uniform with replacement over execution
frame starts **within each family**. Each batch includes every family, then any
extra slots rotate between families. Batch order is shuffled. Batch 12 supplies
two chunks per family; batch 8 supplies one per family and two rotating extras.
Sampling is deterministic for a seed and does not alter source action-query
boundaries or full-training execution-only normalization statistics. The old
`random_frames` default remains unchanged for legacy recipes.

| Setting | Stage1 | Stage2 |
| --- | ---: | ---: |
| Steps | 750 | 10000 |
| Batch | 12 | 8 |
| VLM | Frozen | Trainable |
| Action modules | Trainable | Trainable |
| LR | 1e-5 | 1e-5 |
| Warmup | 100 | 200 |
| Save interval | 750 | 1000 |

Both use BF16 autocast, FP32 VLM storage, gradient checkpointing where applicable,
seed 1000, two workers, horizon 20 and execution 10. Each stage starts a fresh
optimizer/schedule with cosine decay to zero. Stage2 draws 80000 chunks, about
13333 per family; family exposure differs because frame counts differ. This is a
budget reference, not proof that a specific coverage count guarantees convergence.

## Initialization and launch

The machine TOML supplies local dataset/VLM paths, runs_root and the old RoboMME
checkpoint. **Stage1 `--fresh` reads that checkpoint's configuration only** (model
architecture/features); it loads no policy weights or saved processors. The
pretrained generic InternVL is loaded from the local VLM asset, and action-related
modules are randomly initialized. Fresh processors use audited full execution
statistics. `run.json` records `fresh_policy=true`, `checkpoint=null` and the
separate `config_template`. This does not train InternVL from random weights.

From the repository root, using the ordinary 5090 environment (not environments/v100):

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false \
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run --frozen python -m lerobot_policy_vla_ttt.training.train_task_subset \
  --machine configs/machine.local.toml \
  --recipe configs/training/robomme-screen/multitask-six-stage1.json \
  --run-name robomme-six-fresh-seed1000-stage1 --fresh
```

After Stage1 completes, replace `/ABSOLUTE/RUNS_ROOT` with the machine TOML's
`storage.runs_root` and launch Stage2. Do not omit the checkpoint override and do
not pass `--fresh` here; otherwise the wrong initialization would be used.

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false \
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run --frozen python -m lerobot_policy_vla_ttt.training.train_task_subset \
  --machine configs/machine.local.toml \
  --checkpoint /ABSOLUTE/RUNS_ROOT/robomme-six-fresh-seed1000-stage1/checkpoints/000750/pretrained_model \
  --recipe configs/training/robomme-screen/multitask-six-stage2.json \
  --run-name robomme-six-fresh-seed1000-stage2
```

Append `--dry-run` to inspect configuration without starting training, or
`--stop-after 60` with a separate run name for profiling. Profiles write no
checkpoints and cannot supply the Stage2 initializer. `--checkpoint` initializes
weights, not exact optimizer/data-order resume. This entrypoint remains single
GPU; use neither torchrun nor the V100 environment for this experiment.

## Move a patched 5090 checkout onto the remote branch

The remote branch is `experiment/robomme-task-screen`. On the server currently
on that branch with the old uncommitted patch, first save tracked AND untracked
files, then retain the old branch under another name:

```bash
git stash push -u -m "robomme task screen before remote branch migration"
git branch -m experiment/robomme-task-screen-before-sync
git fetch origin
git switch --track -c experiment/robomme-task-screen origin/experiment/robomme-task-screen
git status
```

If the backup branch name already exists, choose a new unique backup name.
Do not pop the stash onto the new branch: the old implementation/configs have
already been incorporated. Keep it as a backup and inspect it if there were
additional server-only edits. Ignored model/data/output files and machine.local
configuration are not removed by `stash -u`; verify the machine TOML before launch.
After migration ordinary `git pull --ff-only` updates this experiment branch.
Master and `feat/v100-training` are unchanged.
