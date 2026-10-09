# AQMem and T²Mem-style on unchanged Evo1-GR00T backbones

User-approved design, 2026-10-09. Implement in `.worktrees/observation-memory`,
never master. Foundation source snapshot is recorded separately.

Independent `models/aqmem/` and `models/t2mem/`; shared mechanics under
`models/observation_memory/` and `ttt/`. Existing RoboTTT retained unchanged.
No additional action registers. State and action tokens keep the baseline layout.
Use 16 observation-only interface tokens, evolving across depth. Select VLM block
outputs explicitly (zero-based indices), evenly spanning the retained language
layers by default, one per action block. No changes to VLM/AE layer counts,
parameters or direct current-VL path. Disable memory to recover baseline exactly.

AQMem: action/state attention outputs query old memory; gated residual before FFN.
T²Mem-style: interface summary queries old memory; gated retrieval enters interface
FFN and becomes additional K/V for subsequent action attention. Last interface
update has no consumer, so omit its memory. No action/time/state path into writes.
Interface summary for AQMem evolves through observation-only attention and FFNs.

New shared memory: independent heads, default 16 x (64 -> 256 -> 64), exact GeLU,
bias, W0 normal std .02, zero biases; QKV per-head RMSNorm, FP32 state; gate .01.
AQMem memory Q/K RoPE disabled; T²Mem interface-slot interleaved RoPE theta 10000.
Ordinary GD, Titans momentum/decay and T²Mem adaptive updates all supported.
T²Mem per-head beta uses residual/alignment and curvature calibrated step, both
stop-gradient, while recurrent inner gradients and learned multiplier differentiate.

Clock: physical environment step, even execution horizon E, stride E/2, starts at
step 0. Read old state through full denoising, then commit one observation write.
At midpoint, observation-only write, no action sampling. Explicit duplicate-step
handling; reject skipped required observations. Fast state detached during inference.
Reset clears clocks, caches, queues, weights, updater auxiliary state.

Training: decision-aligned episode-bounded windows sampled every E/2 environment
steps, action loss only every E; all valid positions can write. Context-only steps
must not consume action labels or sample action noise. TBPTT backward/release per
segment; skip empty-supervision segments' backward, carry detached state. Memory-only,
policy-only, joint and alternating phase schedule available. Warm-start checkpoint;
exact resume remains out of scope. MAC/MAG/MAL receive opt-in same clock/read-before-
write scheduling for comparisons without changing their legacy defaults.

Paper source: https://arxiv.org/html/2609.36720v1 sections 3.3, Appendix A/B.
This is a backbone adaptation, not exact pi0.5 reproduction. No GPU experiments
in this session. CPU mechanisms and tiny end-to-end pipeline are acceptance gates;
actual task convergence, performance and simulator benchmarks remain pending.
