# Observation memory implementation plan

Spec: ../specs/2026-10-09-observation-memory.md
Base snapshot: 8eb56ea, branch feat/observation-memory. No master edits or merge.

1. Shared headed memory and adaptive optimizer. RED tests for reference per-head
   update, masks/outer derivatives/read purity; implement and run CPU tests.
2. Multi-layer VLM export and independent AQMem/T2Mem connections. RED tests for
   token topology, baseline equivalence, action-independent writes, frozen reads.
3. Policy clock, midpoint writes, reset, strict checkpoints and service routing.
   RED queue/clock/slow-weight immutability and checkpoint integration tests.
4. Aligned sequence windows, true TBPTT and phase training CLI; shared clock option
   for MAC/MAG/MAL. RED midpoint supervision, detach and actual dataset CLI tests.
5. Full CPU suite, lint, independent fresh reviewer, fix important defects, docs
   and scoped commits. Do not launch CUDA training or merge to master.

Review focus: curvature math/outer gradients; observation-only interface paths;
layer hooks/checkpointing; stale clocks/duplicate requests; graph release; sequence
alignment/padding; preservation of baseline and legacy model behavior.

Status: steps 1–5 complete in the isolated worktree. Final suite 181 passed / 7
explicit integration skips; Ruff clean. Independent fresh review found one AQMem
query-source mismatch, reproduced RED and fixed GREEN; complete suite passed
again. No merge or CUDA experiments. See ../../experiments/2026-10-09-observation-memory-implementation.md.
