# AQMem / T²Mem-style observation memory

Implemented in `.worktrees/observation-memory`, branch `feat/observation-memory`.
The source snapshot is recorded in
[the provenance manifest](experiments/2026-10-09-observation-memory-source.json).
This implementation adapts [T²Mem](https://arxiv.org/html/2609.36720v1) to the
existing Evo1-GR00T VLM/action expert; it is not an exact reproduction of its pi0.5 backbone.

## Architectures and reusable code

| Directory under `src/lerobot_policy_vla_ttt/` | Responsibility |
| --- | --- |
| `models/aqmem/` | Independent config/policy/action-head connection: action attention output queries memory |
| `models/t2mem/` | Independent config/policy/action-head connection: VL summary queries memory, retrieval enters interface, next attention consumes it |
| `models/observation_memory/` | Multi-layer VLM export, observation interface, explicit observation cache, policy adapters and physical clock |
| `ttt/headed_memory.py` | Separate pure read and differentiable write; per-sample, per-head fast MLP |
| `ttt/updates/t2mem_adaptive.py` | Alignment/residual and curvature calibrated update |
| `training/sequence.py`, `training/tbptt.py` | Decision-aligned windows, midpoint masks, segmented backward and graph release |

The VLM and action expert retain their original layer counts, widths, parameter
names and current-VL conditioning path. Baseline checkpoints initialize their
weights strictly. Both new architectures use **zero action register tokens**.
The default 16 interface tokens belong to the observation branch. Their learned
initial values restart each observation and their hidden states evolve across
depth; fast weights carry information between observations.

`vlm_layer_indices` selects zero-based retained VLM **block outputs before final
LM normalization**. The default mapping spans the retained stack. For the usual
14 VLM / 8 action layers, AQMem maps to `(0, 2, 4, 6, 7, 9, 11, 13)` and T²Mem-style
to `(0, 2, 4, 6, 9, 11, 13)`. T²Mem-style has 7 effective interfaces: stage i feeds
action block i+1, with no unused final memory module. Explicit indices can repeat
or select fewer source layers while preserving the action expert.

Interfaces attend only to valid VL tokens. Action tokens, proprioception and
flow time cannot enter the write path. This restriction is structural; no optional
unsafe mask mode is exposed. AQMem reads after action attention and before FFN;
T²Mem-style uses interface readouts as extra K/V in subsequent action attention.
`memory_enabled=False` provides a numerical baseline bypass.

The headed fast MLP defaults to 16 independent heads, each `64 → 256 → 64`, exact
GeLU, biases, per-head Q/K/V RMSNorm and FP32 fast state. T²Mem-style applies
interleaved RoPE to interface slots; AQMem omits memory RoPE because its action Q
positions and interface K positions have different meanings.

## Update rule and clock are independent

`memory_update_rule` accepts `gradient_descent` (default), `momentum_decay`, or
`t2mem_adaptive` for both architectures. Choose the latter explicitly for T²Mem's
adaptive rule. Its curvature/alignment calibration is stop-gradient; the actual
inner gradient and learned multiplier retain outer derivatives. The existing
single-head MAC/MAG/MAL memory also accepts this rule through a masked single-head
adapter; its fast-MLP architecture stays unchanged.

For execution length E (positive and even), writes occur every E/2 physical
steps: 0, E/2, E, ... . At a decision, all denoising iterations read the same old
memory; the current observation is committed **once after sampling**. At a
midpoint, only the observation branch runs and writes. Predicted action tails
never become memory-write features, even when the prediction horizon exceeds E.

`select_action(batch)` must receive every physical observation. It automatically
serves queued actions and processes midpoint observations. Explicit
`env_step=...` deduplicates retries; changing batch size requires reset. For chunk
clients, call `predict_action_chunk(batch, env_step=0)`, deliver midpoint with
`observe(batch_mid, env_step=E//2)`, then predict at E. Missing a required observation
raises an error. Do not mix chunk-client and per-step-client calling conventions
within an episode. Call `reset()` at episode boundaries. Inference uses `no_grad`,
not `inference_mode`, because local fast-weight updates need autograd. No slow
parameter or `.grad` buffer is mutated by serving.

MAC/MAG/MAL support this same clock with `memory_update_protocol=half_execution`
and `memory_granularity=observation`, including read-before-write. Their default
legacy `observation` protocol and original RoboTTT behavior are preserved.

## Training

The current LIBERO training decisions and real-checkpoint GPU preflight results
are recorded in the [2026-10-10 training recipe](experiments/2026-10-10-libero-observation-memory-training-recipe.md).
That recipe distinguishes agreed protocol, provisional hyperparameters and pending
trainer work. The command below demonstrates the existing windowed trainer; it
does not yet implement the complete full-episode recipe.

From this worktree, use the existing foundation environment with `PYTHONPATH=src`
to avoid its editable installation importing the other worktree:

```bash
PYTHONPATH=src ../evo1-plugin-foundation/.venv/bin/python -m \
  lerobot_policy_vla_ttt.training.train_robottt \
  --architecture aqmem \
  --checkpoint /absolute/path/to/evo1_groot/checkpoint \
  --dataset-root /absolute/path/to/lerobot/dataset --dataset-repo local/dataset \
  --benchmark libero --output artifacts/aqmem-run --device cuda \
  --memory-update-rule t2mem_adaptive \
  --memory-training alternating --phase-length 100 \
  --sequence-length 8 --tbptt-length 4 --segment-transfer
```

Replace `aqmem` with `t2mem` for the interface-query architecture. The baseline
checkpoint's execution length must be even; action/state normalization is reused.
RoboMME requires the same audited preparation manifest as existing trainers.
For MAC/MAG/MAL comparisons, add `--memory-update-protocol half_execution`.

The new protocol derives observation stride from checkpoint `n_action_steps / 2`;
`--decision-stride` affects legacy training only. Each sampled window starts on
an episode-relative decision boundary. Midpoint frames write without action
supervision or action-noise sampling. Valid/padding, write, supervision and row
reset masks are independent. Odd window lengths overlap at the boundary decision
to maintain alignment. Window starts reset memory; these are **windowed training**
runs, not full-episode training. Full-episode strategy remains a research choice.

`memory_only`, `policy_only`, `joint`, and `alternating` phases are supported.
Alternation switches every `--phase-length` optimizer steps, retaining optimizer
moments for both parameter groups. Frozen blocks still transmit required input
gradients. TBPTT performs backward and detaches memory per segment, releasing the
previous graph; context-only segments skip backward. Use segments spanning
multiple observation events to learn writes from future action losses. A one-event
segment deliberately truncates that write-to-future-loss path.

Checkpoints save slow learned state and preprocessing; runtime fast state and
clocks reset on load. The runner's checkpoint argument is a warm start, **not** an
exact optimizer/RNG/episode resume. Run metadata records resolved configuration,
observation stride, update rule and selected architecture.

## Evaluation and validation

The existing policy server accepts `--implementation aqmem` / `t2mem`. The current
synchronous benchmark client submits each physical observation, so midpoint writes
are retained. Keep `n_action_steps` and the intended write stride consistent with
training when comparing methods.

CPU tests cover real tiny InternVL (vision/projector/Qwen2), strict baseline
initialization and bypass, non-reentrant checkpoint gradients, per-head updates,
midpoints/duplicates/reset, masked sequential training, actual LeRobot dataset
training, checkpoint reload and the full inference service. See the
[validation record](experiments/2026-10-09-observation-memory-implementation.md).
Pretrained-size GPU memory/throughput, convergence, LIBERO and RoboMME task success
remain to be measured; this change does not claim benchmark gains.
