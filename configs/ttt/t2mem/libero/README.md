# LIBERO T²Mem episode recipe

`recipe-72gb.json` is an explicit outer-loop recipe for
`python -m lerobot_policy_vla_ttt.training.train_episode_parallel --recipe ...`.
It is not a standalone model config. Model dimensions, processors and action
normalization come from the common Evo1-GR00T Stage1 5k checkpoint.

The tested resource settings are one episode per microbatch, global batch 4
and TBPTT 64 observation events. The runtime `Recipe` class retains its conservative
8-event default; pass this JSON explicitly to use 64. The formal schedule remains
2000 updates, alternating 500 Memory / 500 Policy updates, unlike the short
resource test that switches every update. Do not initialize a formal run from a
resource-test checkpoint.

See the [GPU validation report](../../../../docs/experiments/2026-10-12-bptt-resource-validation.md)
for measured scope, limits and readiness. Other architectures and other GPU models
require their own resource smoke tests before adopting these settings.

The current agreed working choice remains TBPTT=8 (the trainer default).
This 64-event file is an explicitly opt-in resource-tested candidate, not a
final decision to replace that choice or an equal-budget baseline comparison.
