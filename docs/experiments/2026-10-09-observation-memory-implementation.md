# Observation memory implementation validation — 2026-10-09

Worktree: `.worktrees/observation-memory`; branch: `feat/observation-memory`.
Source snapshot: `8eb56ea`, from foundation `8580b54` plus captured foundation work.
The foundation worktree and master were not edited. The LIBERO baseline preset
was copied as snapshot completion to match the source recipe tests; this feature
introduces no new baseline recipe. Source hashes are in the adjacent JSON manifest.

## Scope

AQMem/T²Mem-style, shared headed MLP and adaptive update, observation-only multi-layer
interfaces, physical E/2 writes, pure denoising reads, midpoint service updates,
strict baseline/checkpoint loading, sequential masked TBPTT, phase-aware training
and optional matching MAC/MAG/MAL protocol. See [usage](../observation-memory.md).

## Evidence

Final CPU command (foundation environment, importing this worktree's source):

```bash
PYTHONPATH=src PATH="../evo1-plugin-foundation/.venv/bin:$PATH" \
OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q
```

Before independent review: **180 passed, 7 skipped**, 18 upstream torch.jit
DeprecationWarnings. The seven existing real-checkpoint/CUDA/benchmark integration
tests were skipped because their explicit opt-in fixtures were absent. Real tiny
InternVL, LeRobot image dataset, processor/checkpoint/service tests ran on CPU.
The first sandboxed run had socket/cache permission failures; the same tests
passed with those accesses authorized, without code changes for these failures.

Mechanism tests cover all three update rules, per-head quadratic reference,
masked rows, outer gradients, read purity, VL padding invariance, baseline bypass,
non-reentrant VLM checkpointing, action/state-independent writes, explicit clock
retry and skipped-midpoint rejection, atomic state on sampling failure, full slow
parameter/gradient immutability during serving, checkpoint reset, row reset and
TBPTT detachment, actual two-phase local dataset training and reload.

Extra edge tests exposed and fixed stale batch-size acceptance, doubled Titans
half-execution decision counters, and memory-disabled half-execution inference.
Each was reproduced with a failing test before its fix.

## Implementation decisions

- Source layer indices are zero-based raw transformer block outputs before final
  LM norm; the default evenly spans the retained stack, using the last layer for
  a single effective interface stage. Different choices are configurable; changing
  the default changes the experimental representation and should be recorded.
- Physical-clock mechanics are shared; connection topology remains in independent
  model directories. Legacy default schedules remain unchanged. Incorrect sharing
  would couple model clocks, covered by parameterized regression tests.
- Adaptive updates on the existing MAC/MAG/MAL fast MLP use a single-head adapter
  with mask-scaled objective. This preserves checkpoint tensor topology; it is an
  update-rule ablation, not the new multi-head memory architecture.
- Window starts reset fast state; odd observation lengths overlap at one decision
  to preserve alignment. This intentionally does not claim full-episode training.

## Remaining empirical work

No GPU training, pretrained-scale throughput/peak-memory measurement or new
LIBERO/RoboMME task scores were run. These remain necessary for performance and
scientific conclusions. Exact optimizer/RNG/episode resume remains out of scope.

## Independent review

Fresh-context reviewer `gpt-6-astra` inspected the implementation against the spec
and ran an AQMem query-source probe. One P2 mismatch was found: AQMem read from
attention plus input residual instead of the attention output itself. A regression
test first failed on all 128 query elements; the connection now reads from raw
attention output and fuses the gated retrieval into the residual before FFN.
The five connection tests pass after the fix. No other concrete findings or
minor deferrals were reported. The reviewer did not assess GPU/task performance.

Final post-review verification: **181 passed, 7 skipped**, 18 upstream deprecation
warnings, 28.25 s. `ruff check src tests` and `git diff --check` pass. No deferred
minor review findings. Branch kept for user review, no merge/push.
