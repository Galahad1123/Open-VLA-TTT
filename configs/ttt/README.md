# TTT experiment settings

Use the sequence trainer with a local trained `evo1_groot` checkpoint. The source
checkpoint supplies VLM assets, dimensions, processors and normalization; a free-standing
partial LeRobot config must not replace that provenance.

Common choices:

| CLI option | Default for a new model | Alternatives |
| --- | --- | --- |
| `--architecture` | `robottt` | `mac`, `mag`, `mal` |
| `--memory-update-rule` | `gradient_descent` | `momentum_decay` |
| `--memory-layers` (VLM only) | `10 12` (zero-based) | retained Qwen2 layer indices |
| `--memory-input` (VLM only) | `all` | `visual` (Q/K/V and fusion selection) |
| `--memory-granularity` (VLM only) | `observation` | `token` |
| `--num-persistent-tokens` (VLM only) | `4` | nonnegative integer |
| `--memory-dim`, `--memory-hidden-dim` | `128`, `256` | positive (memory dim even) |
| `--memory-learning-rate` | `0.1` | positive |
| `--memory-training` | `memory_only` | `joint` |
| `--memory-momentum-init`, `--memory-decay-init` | `0.9`, `0.01` | values in `(0,1)` |
| `--memory-momentum`, `--memory-forgetting` | enabled | `--no-memory-momentum`, `--no-memory-forgetting` |
| `--memory-data-dependent` | enabled | `--no-memory-data-dependent` |

Update flags only affect `momentum_decay`. To recover ordinary GD from that rule,
disable momentum and forgetting and disable data-dependent coefficients. Other
memory operations and initial learning rate must match too.

When loading an existing memory checkpoint, omitted architecture settings inherit
its configuration. Explicit incompatible changes fail strict loading. This command
is a weights/processors warm start, not exact optimizer/RNG/data-position resume.

Commands and adaptation boundaries: [Titans VLM guide](../../docs/titans.md).
