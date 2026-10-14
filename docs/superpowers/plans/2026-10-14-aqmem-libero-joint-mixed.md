# AQMem LIBERO Joint Training with Mixed Tasks

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable an AQMem LIBERO experiment initialized from Evo1-GR00T Stage2, jointly training policy and memory with task-diverse episode batches and the existing distributed gradient reducer.

**Architecture:** Extend the existing episode trainer with opt-in joint optimization and a separate mixed-task sampler. Preserve independent fast states, chronological episode processing, equal episode loss weighting, and global-slot sharding. Existing alternating recipes retain their behavior.

**Tech Stack:** Python 3.12, PyTorch, LeRobot 0.6.1, Accelerate, pytest, CPU/Gloo for distributed verification.

**Spec:** User's in-chat requirements dated 2026-10-14: AQMem + LIBERO; Stage2 20k or 30k initialization; joint policy/memory training; mixed tasks; full VLM training in this first version; distributed implementation without claiming multi-GPU validation; work only in `.worktrees/observation-memory`.

## Global Constraints

- Worktree: `/home/hello/lizk/vlas/vla-ttt/.worktrees/observation-memory`, branch `feat/observation-memory`; do not modify or merge into master.
- Default initialization for the experiment: existing LIBERO Evo1-GR00T Stage2 `020000/pretrained_model`. Keep checkpoint selection explicit so `030000` is also usable.
- First experiment: AQMem, `memory_gate_init=0.001`, adaptive writes, H=50, E=10, TBPTT=8 events, effective batch=8, microbatch=1 episode.
- Joint training updates VLM (vision, projector, language), action expert, and all active memory components every outer update.
- Keep current learning-rate groups: policy 1e-5; memory including AQMem action gates 7.5e-5; W0/update parameters/bias/norm have no decay. T2Mem-specific high gate LR does not apply to AQMem.
- Initial recipe candidate: 2000 updates, warmup 50, cosine to 10%, clip norm 1.0, save every 250, seed 1000. This is a configurable budget, not a claim of optimal training length.
- Implement and run CPU tests and dry-run configuration checks. Do not launch a long GPU experiment as part of this implementation. CUDA/NCCL multi-GPU execution remains unvalidated.
- Use the existing environment with `PYTHONPATH=src` so imports resolve to this worktree.

## Review Focus

- Mixed batches smaller/larger than the number of tasks must preserve task-uniform marginals and never confuse task count with episode/frame count (Task 1).
- A joint optimizer must update backbone and memory exactly once with correct parameter groups; no duplicate parameters or doubled step counters (Task 2).
- Old checkpoints lacking new recipe keys must retain legacy semantics; changes to sampling/training mode must reject exact resume (Task 3).
- Rank partitioning must produce the same global samples, weights, and gradients despite variable episode lengths (Task 4).
- AQMem zero gates must reproduce the baseline while memory writes remain enabled; experiment startup must preserve preprocessing and source model dimensions (Tasks 2 and 5).

## Task 1: Explicit training modes and mixed-task sampler

**Files:** `training/episode_parallel/config.py`, `data.py`; `tests/unit/episode_parallel/test_data.py` (paths under the existing source/test roots).

**Interfaces:** Add `Recipe.training_mode` (`alternating` default, `joint`) and `Recipe.sampling_mode` (`same_task` default, `mixed_tasks`). Add `MixedTaskEpisodeSampler` with the same constructor and `batch(step)` / `local_batch(step, rank, world_size)` interface as `EpisodeSampler`.

- [ ] Write tests for mixed batches: task uniqueness for B <= task count; balanced task counts for larger B; seed/step reproducibility; independence from episode length; reconstruction of the global batch from 1/2/4 rank slots; invalid mode rejection.
- [ ] Run the new tests and confirm failures are due to absent functionality.
- [ ] Implement task sampling as shuffled complete task cycles plus a shuffled partial cycle, choosing an episode uniformly within each selected task. Task RNG is deterministic by seed and outer step; global sample seeds depend on slot, not rank. Keep the existing sampler unchanged.
- [ ] Extend `Recipe.phase` / `phase_budget` for a single `joint` phase. Verify legacy phase sequences and sampler tests remain unchanged.

## Task 2: Joint outer optimization and AQMem regression

**Files:** `training/episode_parallel/optim.py`; `tests/unit/episode_parallel/test_engine.py`; `tests/unit/observation_memory/test_connections.py`.

**Interfaces:** Preserve `PhaseOptimizers.begin(step)`, `step()`, `state_dict()`, `load_state_dict()`. In joint mode use one AdamW with the union of the existing disjoint policy/memory groups and counts `{"joint": n}`. Set model phase to `joint` before forward. Clip the combined active gradients once.

- [ ] Add real tiny AQMem tests proving vision/language/action/memory receive gradients and update in the same step, parameter identities are unique, LR/decay groups are preserved, and scheduler/counters count outer updates once.
- [ ] Add a zero-gate AQMem regression using enabled memory and prior writes, matching attention backend, fixed inputs/noise, and comparison of velocity and sampled actions to the initialized baseline. Verify the gate can receive an outer gradient at zero initialization.
- [ ] Run tests red, implement joint optimizer behavior, then run engine/connection regression tests green.

## Task 3: Runner, logs, and exact resume

**Files:** `training/episode_parallel/runner.py`, `checkpoint.py`, `training/train_episode_parallel.py`; `tests/unit/episode_parallel/test_checkpoint.py`, `test_cli.py`.

**Interfaces:** Runner chooses the sampler from `sampling_mode`; configuration selects initial model phase from `training_mode`. Save both modes in recipe metadata. Log global batch task diversity and named optimizer categories/LRs so joint updates are auditable.

- [ ] Add uninterrupted-versus-resumed tiny AQMem joint/mixed tests comparing final tensors, optimizer state, counters, and sampled progress.
- [ ] Test rejection of changed batch/mode/identity/world size. Add a legacy recipe fixture without the two new fields and ensure it restores as alternating/same-task with unchanged numerical behavior.
- [ ] Run tests red; normalize only absent new recipe fields to legacy defaults at restore, retaining strict equality for other settings; implement runner/config plumbing.
- [ ] Run checkpoint and CLI tests green, including restoration into a fresh output directory.

## Task 4: Distributed adaptation and CPU verification

**Files:** `tests/unit/episode_parallel/distributed_probe.py`, `test_distributed.py`; reducer/runner source only if tests identify needed changes.

**Interfaces:** Preserve existing global-slot partitioning, loss division by global batch, and explicit gradient `all_reduce(SUM)` without world-size averaging. Fast states remain rank-local and episode-local.

- [ ] Extend the probe with actual mixed-task AQMem joint training through the production runner using variable-length episodes. The fixture must make different task samples observably different.
- [ ] Compare one CPU process against two Gloo ranks for loss, gradients, final parameters, and optimizer progression. Verify within-run ranks agree exactly and inter-world comparisons use existing tolerances.
- [ ] Verify distributed joint resume matches uninterrupted execution at the same world size, including both parameter families' AdamW states.
- [ ] Retain existing alternating-variant checks and globally-unused-gradient handling. Document CPU/Gloo evidence separately from unexecuted CUDA/NCCL support.

## Task 5: Experiment assets, dry run, and final review

**Files:** Create `configs/ttt/aqmem/libero/recipe.json`, `memory.json`, `README.md`; update `docs/episode-parallel-training.md` for modes and compatibility.

- [ ] Add the recipe candidate above and a memory override with gate 0.001 and explicit adaptive writes. Keep machine paths out of portable recipe JSON.
- [ ] Document explicit Stage2 20k initialization, optional 30k selection, single-GPU command, 2/4/8-rank Accelerate commands with global batch 8, stop-after smoke, and exact resume.
- [ ] Run no-CUDA dry-run against the real local Stage2 20k configuration and LIBERO metadata; verify AQMem, joint, mixed-task batch 8, H50/E10, full VLM training, and source processors/augmentation selection.
- [ ] Run the complete pytest suite offline with CUDA hidden; distinguish asset-dependent skips and any sandbox socket restrictions. Run Ruff on changed code.
- [ ] Review the diff for unintended model/legacy recipe changes and verify work remains only on `feat/observation-memory`. Report tests and remaining GPU validation; do not merge to master or launch long training.

## Pre-implementation inspection

- Linked worktree is clean and on the requested branch; no additional worktree is needed.
- Stage2 20k and 30k both contain Evo1-GR00T weights/configs with H50/E10 and FP32 VLM parameter configuration.
- Relevant existing baseline tests: **19 passed**, with 18 upstream TorchScript deprecation warnings.
- Model-level `joint` already exists; the missing functionality is optimizer/runner support, mixed sampling, and integrated resume/distributed coverage.
- No implementation changes made before plan review.
