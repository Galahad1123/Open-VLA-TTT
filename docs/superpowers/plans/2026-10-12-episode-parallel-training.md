# Episode Parallel Training Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans for inline implementation with test-first changes.

**Goal:** Add an independent Accelerate episode-parallel trainer without changing proven trainers.

**Architecture:** Deterministic global task/episode sampling, per-episode TBPTT, explicit boundary gradient reduction. Separate recipe, engine, runtime and CLI modules under training/episode_parallel.

**Tech Stack:** PyTorch, Accelerate, LeRobot 0.6.1, pytest, CPU/Gloo.

**Spec:** ../specs/2026-10-12-episode-parallel-training-design.md

## Global Constraints

- Work only in .worktrees/observation-memory; leave root checkout and existing trainer files untouched.
- No GPU probes, no dependency installation, no long training. Limit CPU threads and hide CUDA during tests.
- Keep global batch=4 independent of process count, microbatch=1 episode, fast states local.
- User requested implementation directly; execute inline without repeating permission prompts.

## Review Focus

- Different episode/segment lengths must never imply different collective call counts.
- Missing local gradients versus globally unused parameters must preserve optimizer semantics.
- Resume rejects world-size, recipe or data mismatches before stepping.
- Augmentation must neither vary within an episode/camera nor perturb model randomness.
- Both phase optimizers and schedules must survive switches and resume.

### Task 1: Recipe and sampling

Files: training/episode_parallel/{config,data}.py; tests/unit/episode_parallel/test_data.py.
Interfaces: Recipe validation; EpisodeSampler.batch(step); EpisodeSource.events(episode_id,seed).
- [x] Write failing tests for global deterministic same-task batches, length-independent sampling, full episode masks and temporal augmentation.
- [x] Run tests and observe missing implementation.
- [x] Implement recipe, stateless sampler, lazy full-episode source and consistent augmentation.
- [x] Run targeted CPU tests.

### Task 2: Engine and optimizers

Files: training/episode_parallel/{engine,optim,runtime}.py; tests/unit/episode_parallel/test_engine.py.
Interfaces: backward_episode(policy, events, recipe, backward); PhaseOptimizers; synchronize_gradients.
- [x] Write tests for equal episode weighting, TBPTT detach, actual memory gradients, phase groups/schedules and unused gradients.
- [x] Run tests before implementation.
- [x] Implement local backward and explicit gradient synchronization at update boundary.
- [x] Verify real tiny models for all five variants and CPU/Gloo 1 versus 2 ranks.

### Task 3: Runnable entry point and recovery

Files: training/train_episode_parallel.py; training/episode_parallel/{checkpoint,runner}.py; tests/unit/episode_parallel/test_checkpoint.py.
Interfaces: run loop with main-rank artifacts and per-rank checkpoint RNG; CLI --dry-run/--resume.
- [x] Test full recovery, incomplete/mismatched checkpoints and CLI validation before initialization.
- [x] Implement initialization from baseline or memory checkpoint, LIBERO source, run manifest, metrics and atomic checkpoints.
- [x] Verify short synthetic runs and recovery across phase boundary.

### Task 4: Documentation and review

Files: docs/episode-parallel-training.md; tests/unit/episode_parallel/distributed_probe.py.
- [x] Document launch, fixed global batch, explicit SUM reducer, resource constraints and deferred GPU acceptance.
- [x] Run CPU-only new tests and existing relevant regression suites; inspect lint and diff.
- [x] Review whole change independently; resolve important findings and record validation limits.

## Progress

- Baseline: 4 selected existing CPU tests passed. Existing user edits in docs/observation-memory.md and the recipe remain untouched.
- Synchronization decision: explicit SUM at outer boundary replaces automatic DDP hooks, retaining Accelerate process orchestration. This trades communication overlap for auditable variable-length and phase semantics.
- Task 1: complete; 4 CPU sampling/augmentation tests passed.
- Task 2: complete; 7 local engine tests and one five-variant single/dual Gloo comparison passed. Gloo required sandbox escalation solely for localhost sockets.
- Task 3: initial checkpoint recovery and real tiny checkpoint/processor CLI tests passed; extending recovery coverage to both populated optimizer states after independent review.
- Independent read-only review: no must-fix bug found; requested stronger resume/world-size/model-contract negative coverage. Implementing those tests before final validation.

- Task 3: complete; CLI loads tiny real checkpoint/processors and resumes; single-process recovery at steps 1/2/3 exactly matches final weights and both optimizer states. World-size/model/data/recipe mismatches and incomplete checkpoints are rejected.
- Task 4: complete; 199 CPU regression tests passed, 7 GPU/simulator integration cases deselected, 18 upstream torch.jit deprecation warnings; lint and format checks passed. No GPU probes, model loads or shared environment changes.
- Fix discovered by extended Gloo recovery: normalize Accelerate cpu:0 to cpu for safetensors loading and resolved CPU policy config; observed failing two-rank recovery then passing full regression.
- Earlier regression failures: new optimizer comparison assertions could not compare string metadata with torch.testing.assert_close (corrected to compare metadata separately); existing CLI discovery required virtualenv bin on process-local PATH. Neither required dependency changes.
- Final command: PATH=/home/hello/lizk/vlas/vla-ttt/.venv/bin:$PATH CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 nice -n 10 /home/hello/lizk/vlas/vla-ttt/.venv/bin/python -m pytest -m 'not integration and not libero' -q
- Real Stage1 configuration-only dry-run passed; no weights loaded. Deferred: actual checkpoint GPU execution, NCCL/BF16 multi-GPU equivalence, steady-state VRAM and throughput.
- Delivery stays unmerged in the existing observation-memory worktree, as requested. No existing source/trainer files or user documentation edits were changed.
