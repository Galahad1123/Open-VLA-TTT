# RoboMME official-style recipe implementation ledger

Approved scope: only robomme-task-screen, no RMBench and no long training launch.

- Seven arm targets relative to raw chunk anchor; gripper unchanged. Quantile normalization.
- Full training execution statistics across padded action chunks, no validation data.
- Official-style train-only camera augmentation; preserve old absolute checkpoints.
- Stage1 fresh head, 3000x12, warmup300; Stage2 30000x8 warmup1000.
- Peak LR1e-5, cosine floor1e-6; saves1000/5000; H20/E10, no EMA.
- Tests, real GPU short saved runs, reload and video rollout, documentation, remote branch push.

Implementation split: dataset/stats/augmentation; processor/inference boundary; recipe/scheduler/launcher.
Ruling: delta inference is restored at the policy-server boundary before queuing, because the
existing independent pre/post pipelines cannot safely exchange an anchor across subsequent observations.
Direct delta select_action must reject unsupported use rather than silently returning relative actions
as absolute targets; predict_action_chunk retains normalized model-space output semantics.

Validation and final resource evidence are recorded in the recipe guide and artifact directory.
