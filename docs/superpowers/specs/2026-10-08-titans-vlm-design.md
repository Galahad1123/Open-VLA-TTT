# Evo1-GR00T VLM MAC/MAG/MAL implementation scope

User-authorized implementation, 2026-10-08. This updates the earlier design's
assumption that MAC/MAG/MAL would connect to the action DiT: they instead connect
inside the retained Qwen2 language stack. The existing DiT is reused unchanged.

Implement three independently registered models, share low-level TTT with RoboTTT,
and add optional first-order Titans momentum/forgetting to all four structures.
Default update granularity is one full observation; token updates are an ablation.
Default input is all valid VLM positions; visual-only positions are an ablation.

The concrete topology, normalization, token-selection meaning, clocks, masks,
checkpoint boundaries, CLI, sources and deviations are maintained in
[docs/titans.md](../../titans.md). Defaults preserve within-observation causal
attention; historical raw-token sliding KV caches are not part of this version.
MAL directly feeds memory readout to attention, without the earlier proposed extra
pre-attention residual adapter. Common block residuals and FFNs are retained.

VLA adaptations must not be described as a complete Titans LLM reproduction.
Mechanism correctness, real-model GPU training, and benchmark effectiveness are
separate validation stages. No existing baseline checkpoint/configuration or
running GPU job is modified by this implementation.
