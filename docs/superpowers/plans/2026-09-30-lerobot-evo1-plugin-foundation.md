# LeRobot Evo-1 Plugin Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a small, independently installable `lerobot-policy-vla-ttt` package that LeRobot discovers as `policy.type=vla_ttt` while delegating Evo-1 modeling, datasets, processors, training, checkpoints, and rollout to LeRobot 0.6.1.

**Architecture:** `VLATTTConfig` and `VLATTTPolicy` are thin subclasses of LeRobot's Evo-1 classes; processor entry points delegate to the upstream factories, and normal training/rollout stays in LeRobot's CLIs. Project-owned code is limited to compatibility checks, an empty future-extension hook, strict Evo-1 checkpoint conversion, and tests proving that the baseline has not diverged.

**Tech Stack:** Python 3.12, PyTorch, LeRobot 0.6.1 with `evo1` and `training` extras, Hugging Face Hub, pytest, uv, Apache-2.0.

**Spec:** `docs/superpowers/specs/2026-09-30-lerobot-evo1-plugin-foundation-design.md`

## Global Constraints

- Support exactly Python `>=3.12,<3.14` and `lerobot[evo1,training]==0.6.1` in this milestone.
- Package/distribution name is `lerobot-policy-vla-ttt`; import package is `lerobot_policy_vla_ttt`; registered policy type is `vla_ttt`.
- Do not vendor, copy, patch in place, or require a checkout of LeRobot or Evo-1.
- Do not wrap `LeRobotDataset`; baseline training uses LeRobot's dataset and `lerobot-train` directly.
- Do not implement RoboTTT, Titans, memory layers, sequence sampling, TBPTT, a custom trainer, or a custom robot loop.
- Do not silently use `strict=False`; checkpoint conversion must report every missing, unexpected, and shape-mismatched tensor.
- CPU unit tests must not instantiate or download InternVL3-1B or Evo-1 weights.
- GPU tests use batch size 1 and bf16 on the single RTX PRO 5000 72GB; they remain opt-in through the `integration` marker.
- Before any model or dataset download, stop and tell the user the repository ID, revision, destination/cache, and dry-run byte estimate. Continue only after that notification; never commit weights or datasets.
- Commit `uv.lock` and `UPSTREAM.md`; record exact model/dataset revisions and the verified CUDA/driver/GPU combination.

## Review Focus

- Processor loading from an existing Evo-1 checkpoint must preserve upstream state/action-stat padding, including LeRobot's resume behavior; Task 3 pins delegation for both fresh and pretrained factories.
- A checkpoint whose `policy.type` is neither `evo1` nor `vla_ttt` must fail with a targeted message before allocating the model; Tasks 5 and 6 pin both paths.
- One malformed checkpoint may contain missing, unexpected, and wrong-shaped tensors simultaneously; Task 5 requires one aggregated diagnostic containing all three classes.
- Plugin construction must not mutate global LeRobot registries or replace `Evo1Model`; Task 4 tests instance-local, no-op extension installation.
- Reset and action-queue behavior must remain inherited from Evo-1 across checkpoint round trips; Tasks 7 and 8 test reset and repeated inference rather than only one forward call.

---

## File Map

- `pyproject.toml` — build metadata, exact LeRobot dependency, extras, pytest markers, and package discovery.
- `uv.lock` — reproducible dependency resolution for Python 3.12.
- `.gitignore` — exclude caches, virtual environments, outputs, downloaded assets, and local secrets.
- `LICENSE` — Apache License 2.0 text.
- `README.md` — installation, plugin training/rollout, checkpoint conversion, integration tests, and download policy.
- `UPSTREAM.md` — exact upstream releases/revisions and verified hardware/software matrix.
- `src/lerobot_policy_vla_ttt/__init__.py` — public plugin API and discovery-triggering imports.
- `src/lerobot_policy_vla_ttt/compatibility.py` — strict LeRobot version check and optional-extra diagnostics.
- `src/lerobot_policy_vla_ttt/configuration_vla_ttt.py` — registered `VLATTTConfig` subclass and Evo-1 config conversion.
- `src/lerobot_policy_vla_ttt/processor_vla_ttt.py` — naming-convention entry points that delegate to Evo-1 processors.
- `src/lerobot_policy_vla_ttt/modeling_vla_ttt.py` — thin policy subclass, extension hook, and explicit checkpoint loaders.
- `src/lerobot_policy_vla_ttt/checkpoint.py` — checkpoint type inspection, state-dict compatibility report, and complete checkpoint conversion.
- `tests/unit/*` — CPU-only behavior tests with upstream model construction mocked.
- `tests/integration/conftest.py` — opt-in environment checks and synthetic observation builder.
- `tests/integration/test_evo1_equivalence.py` — real-checkpoint numerical equivalence and round trip.
- `tests/integration/test_rollout_smoke.py` — one-episode standard LeRobot rollout smoke test.

### Task 1: Package Skeleton and Strict Upstream Compatibility

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `LICENSE`
- Create: `UPSTREAM.md`
- Create: `src/lerobot_policy_vla_ttt/__init__.py`
- Create: `src/lerobot_policy_vla_ttt/compatibility.py`
- Create: `tests/unit/test_compatibility.py`
- Create: `uv.lock`

**Interfaces:**
- Produces: `SUPPORTED_LEROBOT_VERSION: Final[str] = "0.6.1"`.
- Produces: `ensure_supported_lerobot_version(installed_version: str | None = None) -> None`.
- Produces: `require_optional_module(module_name: str, *, install_command: str) -> None`.

- [ ] **Step 1: Write failing compatibility tests**

Add tests named `test_accepts_exact_supported_lerobot_version`, `test_rejects_other_lerobot_version_with_both_versions`, and `test_missing_optional_module_names_install_command`. Monkeypatch `importlib.metadata.version` and `importlib.util.find_spec`; assert exact-version success and actionable `RuntimeError` messages.

- [ ] **Step 2: Run the focused tests and confirm the initial failure**

Run: `python -m pytest tests/unit/test_compatibility.py -v`

Expected: FAIL because the package and functions do not exist.

- [ ] **Step 3: Add package metadata and minimal compatibility implementation**

Configure hatchling for the `src` layout, `requires-python = ">=3.12,<3.14"`, dependency `lerobot[evo1,training]==0.6.1`, optional extra `libero = ["lerobot[libero]==0.6.1"]`, and a development dependency group containing pytest and ruff. Register pytest markers `integration` and `libero`. Implement the three interfaces above without importing any model module.

- [ ] **Step 4: Add repository hygiene and provenance files**

Ignore `.venv/`, caches, `outputs/`, `artifacts/`, `.env*`, and common checkpoint extensions. Add the unmodified Apache-2.0 license and an `UPSTREAM.md` table with LeRobot release `0.6.1`; leave model, dataset, driver, CUDA, and GPU values explicitly marked `not yet verified` rather than inventing revisions.

- [ ] **Step 5: Resolve dependencies and verify the package**

Run: `uv lock && uv run pytest tests/unit/test_compatibility.py -v`

Expected: lock succeeds and all three tests PASS. This may fetch Python package metadata/packages, but must not download model weights or datasets.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock .gitignore LICENSE UPSTREAM.md src/lerobot_policy_vla_ttt tests/unit/test_compatibility.py
git commit -m "build: initialize LeRobot policy plugin"
```

### Task 2: Registered Evo-1-Compatible Configuration

**Files:**
- Create: `src/lerobot_policy_vla_ttt/configuration_vla_ttt.py`
- Modify: `src/lerobot_policy_vla_ttt/__init__.py`
- Create: `tests/unit/test_configuration.py`
- Create: `tests/unit/test_discovery.py`

**Interfaces:**
- Consumes: `SUPPORTED_LEROBOT_VERSION` from Task 1.
- Produces: `@PreTrainedConfig.register_subclass("vla_ttt") class VLATTTConfig(Evo1Config)`.
- Produces: `VLATTTConfig.from_evo1_config(config: Evo1Config) -> VLATTTConfig`.
- `VLATTTConfig` adds only `plugin_schema_version: int = 1` and `tested_lerobot_version: str = "0.6.1"`.

- [ ] **Step 1: Write failing config round-trip and conversion tests**

Test that inherited Evo-1 fields survive `to_dict()` plus reconstruction, plugin metadata has the exact defaults above, and `from_evo1_config()` copies every init field declared by `dataclasses.fields(Evo1Config)` without sharing mutable values.

- [ ] **Step 2: Write the failing discovery test**

Import `lerobot_policy_vla_ttt`, then exercise LeRobot 0.6.1's public config/policy discovery factory with `policy.type=vla_ttt`; assert its resolved config class is `VLATTTConfig`. Do not test private registry dictionaries.

- [ ] **Step 3: Run the focused tests and confirm failure**

Run: `uv run pytest tests/unit/test_configuration.py tests/unit/test_discovery.py -v`

Expected: FAIL because `VLATTTConfig` is absent.

- [ ] **Step 4: Implement the registered dataclass and public import**

Use `dataclasses.fields(Evo1Config)` to select parent constructor fields in `from_evo1_config`; deep-copy field values and reject a non-`Evo1Config` argument with `TypeError`. Export `VLATTTConfig` from the package root so importing the distribution performs registration.

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/unit/test_configuration.py tests/unit/test_discovery.py -v`

Expected: PASS without network access or model construction.

- [ ] **Step 6: Commit**

```bash
git add src/lerobot_policy_vla_ttt tests/unit/test_configuration.py tests/unit/test_discovery.py
git commit -m "feat: register VLA-TTT Evo-1 configuration"
```

### Task 3: Exact Evo-1 Processor Delegation

**Files:**
- Create: `src/lerobot_policy_vla_ttt/processor_vla_ttt.py`
- Create: `tests/unit/test_processor_delegation.py`

**Interfaces:**
- Consumes: `VLATTTConfig` from Task 2.
- Produces: `make_vla_ttt_pre_post_processors(config, dataset_stats=None, **kwargs) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]`.
- Produces: `make_vla_ttt_pre_post_processors_from_pretrained(config, pretrained_path, *, revision=None, dataset_stats=None, dataset_meta=None, preprocessor_overrides=None, postprocessor_overrides=None, preprocessor_config_filename=..., postprocessor_config_filename=...) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]`.
- Both functions return the upstream Evo-1 factory result unchanged.

- [ ] **Step 1: Write failing delegation tests**

Monkeypatch both upstream Evo-1 processor factories with sentinels. Assert each plugin factory forwards the same config object and every positional/keyword argument exactly once, returns the identical pair, and propagates the upstream exception unchanged.

- [ ] **Step 2: Add a processor-stat regression test**

With synthetic 7-dimensional state/action statistics, call both plugin and upstream fresh factories and compare pipeline step types plus serialized state. Repeat through the pretrained factory using a temporary processor checkpoint. This pins Evo-1 padding/resume semantics without duplicating them.

- [ ] **Step 3: Run the focused tests and confirm failure**

Run: `uv run pytest tests/unit/test_processor_delegation.py -v`

Expected: FAIL because the convention-named processor module is absent.

- [ ] **Step 4: Implement thin factory functions**

Import the factory functions from LeRobot 0.6.1's public Evo-1 processor module and call them directly. Do not construct, subclass, copy, or edit any processor step.

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/unit/test_processor_delegation.py -v`

Expected: PASS; network remains unused.

- [ ] **Step 6: Commit**

```bash
git add src/lerobot_policy_vla_ttt/processor_vla_ttt.py tests/unit/test_processor_delegation.py
git commit -m "feat: delegate VLA-TTT processors to Evo-1"
```

### Task 4: Thin Policy Subclass and No-Op Extension Seam

**Files:**
- Create: `src/lerobot_policy_vla_ttt/modeling_vla_ttt.py`
- Modify: `src/lerobot_policy_vla_ttt/__init__.py`
- Create: `tests/unit/test_policy.py`
- Modify: `tests/unit/test_discovery.py`

**Interfaces:**
- Consumes: `VLATTTConfig` and `ensure_supported_lerobot_version()`.
- Produces: `class VLATTTPolicy(Evo1Policy)` with `config_class = VLATTTConfig` and `name = "vla_ttt"`.
- Produces: `VLATTTPolicy.install_extensions(self) -> None`, a default no-op called once after `Evo1Policy.__init__`.

- [ ] **Step 1: Write failing construction tests without allocating Evo-1**

Monkeypatch `Evo1Policy.__init__`, the compatibility guard, and `VLATTTPolicy.install_extensions`. Assert call order is guard, parent initializer, extension hook; each is called once; the identical config and `vlm_hub_kwargs` reach the parent.

- [ ] **Step 2: Pin the empty extension behavior**

Use a fake parent initializer that sets a sentinel `model`. After default construction, assert `policy.model` is that exact sentinel, the model class was not replaced, no global Evo-1 class attributes changed, and `reset`, `select_action`, and `predict_action_chunk` resolve to inherited methods.

- [ ] **Step 3: Extend discovery coverage**

Use LeRobot's public policy factory with model construction mocked and assert `policy.type=vla_ttt` resolves `VLATTTPolicy`, not `Evo1Policy`.

- [ ] **Step 4: Run focused tests and confirm failure**

Run: `uv run pytest tests/unit/test_policy.py tests/unit/test_discovery.py -v`

Expected: FAIL because the policy class is absent.

- [ ] **Step 5: Implement the minimal subclass and exports**

Match Evo-1 0.6.1's constructor signature, run the version guard before allocation, call `super().__init__`, then call the instance hook. Add no forward, action, queue, reset, optimizer, or model-building override.

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/unit/test_policy.py tests/unit/test_discovery.py -v`

Expected: PASS without loading InternVL3.

- [ ] **Step 7: Commit**

```bash
git add src/lerobot_policy_vla_ttt tests/unit/test_policy.py tests/unit/test_discovery.py
git commit -m "feat: add thin VLA-TTT policy subclass"
```

### Task 5: Strict Checkpoint Loading and Evo-1 Warm Start

**Files:**
- Create: `src/lerobot_policy_vla_ttt/checkpoint.py`
- Modify: `src/lerobot_policy_vla_ttt/modeling_vla_ttt.py`
- Create: `tests/unit/test_checkpoint_mapping.py`

**Interfaces:**
- Produces: immutable `StateDictCompatibility` with sorted `missing_keys`, `unexpected_keys`, and `shape_mismatches` fields plus `is_compatible: bool`.
- Produces: `compare_state_dicts(source: Mapping[str, Tensor], target: Mapping[str, Tensor]) -> StateDictCompatibility`.
- Produces: `load_state_dict_strict(target: nn.Module, source: Mapping[str, Tensor]) -> None`.
- Produces: `VLATTTPolicy.from_pretrained(pretrained_name_or_path: str | Path, **kwargs) -> VLATTTPolicy`, which accepts only a `vla_ttt` checkpoint and otherwise fails before model allocation.
- Produces: `VLATTTPolicy.from_evo1_pretrained(pretrained_name_or_path: str | Path, *, revision: str | None = None, local_files_only: bool = False, **kwargs) -> VLATTTPolicy`.

- [ ] **Step 1: Write failing state-dict diagnostic tests**

Create tiny fake modules and assert compatible dictionaries load exactly. In one incompatible input include one missing key, one unexpected key, and one shared key with a wrong shape; assert one exception contains all names, source/target shapes, and no target parameter changes.

- [ ] **Step 2: Write failing warm-start tests with mocked loaders**

Mock `Evo1Policy.from_pretrained` to return a tiny source policy. Assert `from_evo1_pretrained()` converts its config with `VLATTTConfig.from_evo1_config`, creates `VLATTTPolicy`, performs strict loading, forwards revision/local-only arguments, and returns a plugin whose serialized `policy.type` is `vla_ttt`.

- [ ] **Step 3: Pin both checkpoint entry points and wrong types before allocation**

Mock lightweight config inspection. Assert `from_pretrained()` accepts `vla_ttt`, rejects `evo1` with an instruction to call `from_evo1_pretrained()`, and rejects an unrelated type with its observed value. Assert `from_evo1_pretrained()` accepts only `evo1` and rejects both `vla_ttt` and unrelated types before any model loader runs.

- [ ] **Step 4: Run focused tests and confirm failure**

Run: `uv run pytest tests/unit/test_checkpoint_mapping.py -v`

Expected: FAIL because checkpoint helpers and the classmethod are absent.

- [ ] **Step 5: Implement comparison and transactional strict loading**

Compare key sets and shapes before calling `target.load_state_dict(source, strict=True)`. Format all discrepancies in one deterministic error; never partially load when validation fails.

- [ ] **Step 6: Implement the two explicit loading paths**

Use a lightweight config-file inspection before model allocation. For `from_pretrained()`, accept only `vla_ttt` and otherwise raise the targeted error before delegating to the inherited loader. For `from_evo1_pretrained()`, accept only `evo1`, load through `Evo1Policy.from_pretrained`, convert only the config, construct the plugin, and strictly copy the source state dict.

- [ ] **Step 7: Run tests**

Run: `uv run pytest tests/unit/test_checkpoint_mapping.py -v`

Expected: PASS without network or real checkpoints.

- [ ] **Step 8: Commit**

```bash
git add src/lerobot_policy_vla_ttt/checkpoint.py src/lerobot_policy_vla_ttt/modeling_vla_ttt.py tests/unit/test_checkpoint_mapping.py
git commit -m "feat: strictly initialize from Evo-1 checkpoints"
```

### Task 6: Self-Contained Plugin Checkpoint Conversion

**Files:**
- Modify: `src/lerobot_policy_vla_ttt/checkpoint.py`
- Modify: `src/lerobot_policy_vla_ttt/__init__.py`
- Create: `tests/unit/test_checkpoint_conversion.py`

**Interfaces:**
- Consumes: `VLATTTPolicy.from_evo1_pretrained()` and both Task 3 processor factories.
- Produces: `convert_evo1_checkpoint(source: str | Path, destination: Path, *, revision: str | None = None, local_files_only: bool = False) -> Path`.
- The destination contains plugin policy config/weights, preprocessor, postprocessor, and `vla_ttt_conversion.json`; it has no runtime dependency on the source directory.

- [ ] **Step 1: Write a failing artifact-conversion test**

Use temporary directories and fake policy/processors implementing LeRobot's `save_pretrained` contract. Assert each object saves once into the destination, the policy config says `vla_ttt`, metadata records source and revision, and deleting the fake source still permits the fake plugin checkpoint loader to load destination artifacts.

- [ ] **Step 2: Test destination safety and source-type errors**

Assert a non-empty destination is rejected without overwriting files. Assert a source tagged with an unrelated policy type fails before policy or processor construction and names both accepted source type `evo1` and the observed type.

- [ ] **Step 3: Run focused tests and confirm failure**

Run: `uv run pytest tests/unit/test_checkpoint_conversion.py -v`

Expected: FAIL because `convert_evo1_checkpoint` is absent.

- [ ] **Step 4: Implement conversion using LeRobot save/load APIs**

Load the upstream processors from `source`, warm-start the plugin policy, create the destination only after all source artifacts validate, save with public `save_pretrained` methods, then atomically write JSON metadata. Do not manually copy LeRobot processor files or reinterpret their schemas.

- [ ] **Step 5: Run checkpoint and processor tests together**

Run: `uv run pytest tests/unit/test_checkpoint_mapping.py tests/unit/test_checkpoint_conversion.py tests/unit/test_processor_delegation.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/lerobot_policy_vla_ttt tests/unit/test_checkpoint_conversion.py
git commit -m "feat: convert complete Evo-1 checkpoints"
```

### Task 7: Real-Checkpoint Numerical Equivalence and Round Trip

**Files:**
- Create: `tests/integration/conftest.py`
- Create: `tests/integration/test_evo1_equivalence.py`
- Modify: `UPSTREAM.md`

**Interfaces:**
- Consumes: environment variable `VLA_TTT_EVO1_CHECKPOINT` as either a local path or pinned Hub repo; optional `VLA_TTT_EVO1_REVISION` is mandatory for Hub IDs.
- Produces: pytest fixture `evo1_checkpoint() -> tuple[str, str | None]` that skips unless CUDA, bf16, and checkpoint configuration are available.
- Produces: `make_synthetic_raw_batch(config: Evo1Config, *, batch_size: int = 1) -> dict[str, Any]`, derived from declared input features rather than hard-coded camera names.

- [ ] **Step 1: Add opt-in fixture and download notification gate**

If the checkpoint is absent locally, do not download it immediately. First run a Hub dry-run/metadata query, report the exact repo, revision, cache destination, total bytes, and largest files to the user, then proceed only after the notification. Record the confirmed repo and immutable revision in `UPSTREAM.md`.

- [ ] **Step 2: Write the failing equivalence test**

Load native Evo-1 plus its processors and load `VLATTTPolicy.from_evo1_pretrained` from the same source. Preprocess one deterministic synthetic observation, reset both policies, restore the same CPU/CUDA RNG state before each `predict_action_chunk`, and assert identical shape plus `torch.testing.assert_close` at explicitly recorded bf16 tolerances.

- [ ] **Step 3: Write the failing checkpoint round-trip test**

Convert to a temporary plugin checkpoint, delete all references to the loaded native policy, load plugin policy and processors with standard LeRobot discovery, reset both plugin instances, reuse identical RNG state, and assert equal action chunks. Call inference twice around `reset()` to verify inherited action-queue clearing.

- [ ] **Step 4: Run the integration tests**

Run: `uv run pytest -m integration tests/integration/test_evo1_equivalence.py -v -s`

Expected: both tests PASS on the RTX PRO 5000 72GB; missing local assets or GPU produce an explicit SKIP, never an implicit download from a CPU unit run.

- [ ] **Step 5: Record the verified runtime matrix**

Update `UPSTREAM.md` with checkpoint revision, InternVL3 revision resolved by that checkpoint, PyTorch, Transformers, CUDA runtime, NVIDIA driver, GPU name, dtype, and measured tolerances.

- [ ] **Step 6: Commit**

```bash
git add tests/integration/conftest.py tests/integration/test_evo1_equivalence.py UPSTREAM.md
git commit -m "test: verify Evo-1 numerical equivalence"
```

### Task 8: Standard LeRobot Training and Rollout Surface

**Files:**
- Create: `README.md`
- Create: `tests/integration/test_rollout_smoke.py`
- Modify: `UPSTREAM.md`

**Interfaces:**
- Consumes: LeRobot's unmodified `lerobot-train` and `lerobot-rollout` console commands.
- Consumes: `VLA_TTT_LIBERO_CHECKPOINT`, immutable `VLA_TTT_LIBERO_REVISION`, and any benchmark asset variables required by LeRobot 0.6.1.
- Produces no project-owned trainer, dataset wrapper, rollout loop, robot transport, or action queue.

- [ ] **Step 1: Write README commands against public LeRobot CLIs**

Document installation, `policy.type=vla_ttt` Stage-1 training on `lerobot/libero`, Python checkpoint conversion, plugin-native resume, standard rollout, fast/unit versus opt-in integration tests, supported versions, non-affiliation, and the rule that model/data downloads are announced first. Mark commands needing external assets clearly.

- [ ] **Step 2: Add a CLI configuration smoke test**

Invoke both LeRobot CLIs only through argument/config parsing with heavyweight constructors mocked. Assert the training path selects `VLATTTConfig` while retaining LeRobotDataset, and the rollout path selects the plugin policy and plugin processor factory from a plugin checkpoint.

- [ ] **Step 3: Add the opt-in one-episode LIBERO test**

Run the standard LeRobot rollout entry point with the base strategy, batch size 1, one episode, and a fixed seed. Assert successful reset, at least one correctly shaped action, and normal episode termination; do not duplicate the loop in test or package code.

- [ ] **Step 4: Apply the model/data download gate**

Before fetching any missing LIBERO policy, dataset, environment asset, or benchmark bundle, report its repository/URL, immutable revision/checksum, destination, and dry-run byte estimate to the user. After use, record exact revisions in `UPSTREAM.md`; keep all artifacts outside Git.

- [ ] **Step 5: Run smoke and full CPU suites**

Run: `uv run pytest tests/unit -v`

Expected: all unit tests PASS with network disabled.

Run: `uv run pytest -m "integration and libero" tests/integration/test_rollout_smoke.py -v -s`

Expected: PASS when explicitly provisioned; otherwise an explanatory SKIP.

- [ ] **Step 6: Verify the repository contains no copied upstream or large assets**

Run: `git status --short && git ls-files | sort && find . -type f -size +10M -not -path './.git/*' -print`

Expected: only intended source/docs/tests/lock files are tracked, and no file over 10 MiB is reported.

- [ ] **Step 7: Commit**

```bash
git add README.md UPSTREAM.md tests/integration/test_rollout_smoke.py
git commit -m "docs: document LeRobot training and rollout workflow"
```

### Task 9: Final Acceptance Verification

**Files:**
- Modify only files required by verified defects; do not broaden scope.

**Interfaces:**
- Consumes: all interfaces and tests from Tasks 1-8.
- Produces: a clean, reproducible milestone satisfying all seven acceptance criteria in the spec.

- [ ] **Step 1: Run formatting and static checks configured in `pyproject.toml`**

Run: `uv run ruff check . && uv run ruff format --check .`

Expected: PASS.

- [ ] **Step 2: Run the network-free suite**

Run: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run pytest tests/unit -v`

Expected: PASS, proving unit tests cannot download weights or data.

- [ ] **Step 3: Build and inspect the wheel**

Run: `uv build && unzip -l dist/lerobot_policy_vla_ttt-*.whl`

Expected: wheel contains only plugin source and distribution metadata; no tests, model weights, datasets, outputs, or LeRobot source tree.

- [ ] **Step 4: Test installation in a fresh temporary environment**

Install the built wheel into a temporary Python 3.12 environment, import `lerobot_policy_vla_ttt`, and run the public discovery check for `vla_ttt` without a LeRobot repository checkout.

Expected: import and discovery PASS.

- [ ] **Step 5: Run provisioned GPU tests if assets were approved and made available**

Run: `uv run pytest -m integration tests/integration -v -s`

Expected: equivalence, round trip, and LIBERO smoke PASS; otherwise report precisely which tests skipped and why, without claiming GPU acceptance.

- [ ] **Step 6: Review acceptance evidence and working tree**

Run: `git status --short && git log --oneline --decorate -12`

Expected: no unexplained changes and one focused commit per completed task.

- [ ] **Step 7: Commit verification-only fixes if necessary**

```bash
git add <only-the-files-fixed-during-verification>
git commit -m "fix: address plugin acceptance findings"
```

Skip this commit when verification required no code or documentation changes.
