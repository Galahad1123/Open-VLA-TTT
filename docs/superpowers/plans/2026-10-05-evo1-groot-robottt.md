# Evo1-GR00T + RoboTTT implementation

Spec: ../specs/2026-10-05-evo1-groot-robottt-design.md

Implement directly in the existing foundation worktree. Do not change the running
baseline or use its GPU. Paper feature path and cumulative denoising are authorized.

## Task 1: Shared memory

Create `ttt/{functional,fast_models,state,memory}.py` and
`ttt/updates/gradient_descent.py`. Explicit batched fast weights, differentiable W0,
KV MSE averaged over tokens/features, positive learned initial LR 0.1, update then
read, temporal RoPE, gate 0.001. Tests first: manual GD, outer gradients, batch
isolation, masked writes, detach/reset, inference under no_grad. Run CPU pytest;
expected initial missing-module failure then passing mechanism tests.

## Task 2: Independent model and integration

Create `models/robottt/` configuration, action head, policy and processor. Compose
existing token and DiT components with memory after attention before FFN. Explicit
training state; inference-only state persists across denoising calls and resets
with action queue. Strict checkpoint identity and audited baseline initialization.
Reuse established I/O adapters without modifying baseline behavior. Test gradient
flow, cumulative updates, baseline bypass, checkpoint round-trip and causality.
Add processor/factory/evaluation routing; inference uses no_grad, not inference_mode.

## Task 3: Sequential training

Create `training/{sequence,flow_matching,tbptt,train_robottt}.py`. Episode-bounded
windows, decision stride, valid/write/supervision masks, independent flow times,
actual backward per segment and detached numerical carry. Provide a local dataset
CLI with processor/checkpoint persistence. Test windows, segment equivalence,
context-only inputs, padding, and optimizer execution.

## Task 4: Verification and handoff

Run new CPU tests, baseline regression suite and lint. Fresh independent review
of changed code and fix important findings. Record tested scope separately from
GPU convergence and benchmark comparisons, which remain pending. Document CLI,
checkpoint initialization, hyperparameters and overlapping-label diagnostic.

## Review focus

Autograd under evaluation; true TBPTT graph release; state reset/padding/batch
isolation; slow-weight immutability; checkpoint semantic validation; training
normalization and dataset episode boundaries; inherited baseline behavior.
