# LeRobot Evo-1 插件基础实施计划

> **供智能体执行者使用：** 必须使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans`，逐项实施本计划。步骤使用复选框（`- [ ]`）跟踪。

**目标：** 构建一个小型、可独立安装的 `lerobot-policy-vla-ttt` 包，使 LeRobot 能以 `policy.type=vla_ttt` 自动发现它，同时复用 LeRobot 0.6.1 的 Evo-1 模型、数据集、processor、训练、checkpoint 和 rollout。

**架构：** `VLATTTConfig` 与 `VLATTTPolicy` 是 LeRobot Evo-1 对应类的薄子类；processor 入口委托上游 factory，普通训练与 rollout 继续使用 LeRobot CLI。项目代码仅负责兼容性检查、为空的未来扩展钩子、严格 checkpoint 转换及基线等价性测试。

**技术栈：** Python 3.12、PyTorch、带 `evo1` 与 `training` extras 的 LeRobot 0.6.1、Hugging Face Hub、pytest、uv、Apache-2.0。

**规格：** `docs/superpowers/specs/2026-09-30-lerobot-evo1-plugin-foundation-design.md`

## 全局约束

- 只支持 Python `>=3.12,<3.14` 和 `lerobot[evo1,training]==0.6.1`。
- 分发包名为 `lerobot-policy-vla-ttt`，导入包名为 `lerobot_policy_vla_ttt`，policy 类型为 `vla_ttt`。
- 不 vendor、复制或原地修改 LeRobot/Evo-1，也不依赖 LeRobot 仓库 checkout。
- 不包装 `LeRobotDataset`；基线训练直接使用 LeRobotDataset 和 `lerobot-train`。
- 不实现 RoboTTT、Titans、memory layer、序列采样、TBPTT、自定义 trainer 或自定义机器人循环。
- 不得静默使用 `strict=False`；checkpoint 转换必须报告所有 missing、unexpected 和 shape-mismatched tensor。
- CPU 单元测试不得实例化或下载 InternVL3-1B、Evo-1 权重。
- GPU 测试在单张 RTX PRO 5000 72GB 上使用 batch size 1 和 bf16，并通过 `integration` marker 显式启用。
- 下载任何模型或数据集前，必须暂停并告知用户仓库 ID、revision、目标缓存位置和 dry-run 预计字节数。告知后才能继续；权重和数据绝不提交到 Git。
- 提交 `uv.lock` 和 `UPSTREAM.md`，记录模型/数据集精确 revision 及验证过的 CUDA、驱动和 GPU 组合。

## 审查重点

- 从 Evo-1 checkpoint 加载 processor 时须保留上游 state/action 统计量 padding 与 resume 行为；任务 3 覆盖新建和 pretrained 两条路径。
- checkpoint 的 `policy.type` 不是预期值时，须在分配模型前给出针对性错误；任务 5、6 覆盖此项。
- 同一损坏 checkpoint 可能同时含 missing、unexpected 和错误 shape 的 tensor；任务 5 要求一次完整报告三类问题。
- 插件构造不得修改全局 LeRobot registry 或替换 `Evo1Model`；任务 4 验证实例级空扩展。
- checkpoint round-trip 后 reset 与 action queue 行为须保持继承自 Evo-1；任务 7、8 验证 reset 和重复推理。

---

## 文件职责

- `pyproject.toml`：构建元数据、精确依赖、extras、pytest markers 与包发现。
- `uv.lock`：Python 3.12 的可复现依赖解析结果。
- `.gitignore`：排除缓存、虚拟环境、输出、下载资产和本地 secret。
- `LICENSE`：Apache License 2.0 全文。
- `README.md`：安装、训练/rollout、checkpoint 转换、测试和下载规则。
- `UPSTREAM.md`：上游 release/revision 与验证过的软硬件矩阵。
- `src/lerobot_policy_vla_ttt/__init__.py`：公共 API 与触发 discovery 的导入。
- `src/lerobot_policy_vla_ttt/compatibility.py`：LeRobot 版本和 optional extra 检查。
- `src/lerobot_policy_vla_ttt/configuration_vla_ttt.py`：`VLATTTConfig` 注册与配置转换。
- `src/lerobot_policy_vla_ttt/processor_vla_ttt.py`：委托 Evo-1 processor 的入口。
- `src/lerobot_policy_vla_ttt/modeling_vla_ttt.py`：薄 policy 子类、扩展钩子和 checkpoint loader。
- `src/lerobot_policy_vla_ttt/checkpoint.py`：类型检查、state-dict 兼容报告和完整转换。
- `tests/unit/*`：不构造上游大模型的 CPU 单元测试。
- `tests/integration/conftest.py`：显式启用条件和合成 observation 构造器。
- `tests/integration/test_evo1_equivalence.py`：真实 checkpoint 等价性与 round-trip。
- `tests/integration/test_rollout_smoke.py`：标准 LeRobot 单 episode rollout 冒烟测试。

### Task 1：项目骨架与严格上游兼容性

**文件：**
- 新建：`pyproject.toml`、`.gitignore`、`LICENSE`、`UPSTREAM.md`、`uv.lock`
- 新建：`src/lerobot_policy_vla_ttt/__init__.py`
- 新建：`src/lerobot_policy_vla_ttt/compatibility.py`
- 新建：`tests/unit/test_compatibility.py`

**接口：**
- 产出：`SUPPORTED_LEROBOT_VERSION: Final[str] = "0.6.1"`。
- 产出：`ensure_supported_lerobot_version(installed_version: str | None = None) -> None`。
- 产出：`require_optional_module(module_name: str, *, install_command: str) -> None`。

- [ ] **步骤 1：编写失败的兼容性测试**

添加 `test_accepts_exact_supported_lerobot_version`、`test_rejects_other_lerobot_version_with_both_versions`、`test_missing_optional_module_names_install_command`。Monkeypatch `importlib.metadata.version` 和 `importlib.util.find_spec`；验证精确版本成功，以及错误包含当前/支持版本或明确安装命令。

- [ ] **步骤 2：确认测试初始失败**

运行：`python -m pytest tests/unit/test_compatibility.py -v`

预期：因包和函数不存在而失败。

- [ ] **步骤 3：添加包元数据与最小实现**

使用 hatchling 和 `src` 布局；设置 `requires-python = ">=3.12,<3.14"`、依赖 `lerobot[evo1,training]==0.6.1`、extra `libero = ["lerobot[libero]==0.6.1"]`，以及包含 pytest、ruff 的开发依赖组。注册 `integration`、`libero` markers。实现三个接口，但不导入模型模块。

- [ ] **步骤 4：添加仓库卫生与来源记录**

忽略 `.venv/`、缓存、`outputs/`、`artifacts/`、`.env*` 和常见 checkpoint 后缀。加入未经修改的 Apache-2.0 许可证。`UPSTREAM.md` 记录 LeRobot `0.6.1`，其余未验证字段明确写为“尚未验证”。

- [ ] **步骤 5：解析依赖并验证**

运行：`uv lock && uv run pytest tests/unit/test_compatibility.py -v`

预期：lock 成功，三个测试通过。可获取 Python 包元数据/依赖，但不得下载模型或数据集。

- [ ] **步骤 6：提交**

```bash
git add pyproject.toml uv.lock .gitignore LICENSE UPSTREAM.md src/lerobot_policy_vla_ttt tests/unit/test_compatibility.py
git commit -m "build: initialize LeRobot policy plugin"
```

### Task 2：注册与 Evo-1 兼容的配置

**文件：**
- 新建：`src/lerobot_policy_vla_ttt/configuration_vla_ttt.py`
- 修改：`src/lerobot_policy_vla_ttt/__init__.py`
- 新建：`tests/unit/test_configuration.py`、`tests/unit/test_discovery.py`

**接口：**
- 消费：任务 1 的 `SUPPORTED_LEROBOT_VERSION`。
- 产出：`@PreTrainedConfig.register_subclass("vla_ttt") class VLATTTConfig(Evo1Config)`。
- 产出：`VLATTTConfig.from_evo1_config(config: Evo1Config) -> VLATTTConfig`。
- 只新增 `plugin_schema_version: int = 1`、`tested_lerobot_version: str = "0.6.1"`。

- [ ] **步骤 1：编写失败的 round-trip 与转换测试**

验证继承字段经 `to_dict()` 和重建保持不变，元数据默认值精确匹配；`from_evo1_config()` 复制 `dataclasses.fields(Evo1Config)` 的所有 init 字段且不共享可变值。

- [ ] **步骤 2：编写失败的 discovery 测试**

导入插件后，通过 LeRobot 0.6.1 公开 factory 解析 `policy.type=vla_ttt`，断言配置类为 `VLATTTConfig`；不测试私有 registry。

- [ ] **步骤 3：确认测试失败**

运行：`uv run pytest tests/unit/test_configuration.py tests/unit/test_discovery.py -v`

预期：因 `VLATTTConfig` 不存在而失败。

- [ ] **步骤 4：实现注册类与公共导出**

用 `dataclasses.fields(Evo1Config)` 选择父类构造字段并 deep-copy；非 `Evo1Config` 参数抛出 `TypeError`。从包根导出 `VLATTTConfig`，使导入时完成注册。

- [ ] **步骤 5：运行测试**

运行：`uv run pytest tests/unit/test_configuration.py tests/unit/test_discovery.py -v`

预期：全部通过，不访问网络、不构造模型。

- [ ] **步骤 6：提交**

```bash
git add src/lerobot_policy_vla_ttt tests/unit/test_configuration.py tests/unit/test_discovery.py
git commit -m "feat: register VLA-TTT Evo-1 configuration"
```

### Task 3：精确委托 Evo-1 Processor

**文件：**
- 新建：`src/lerobot_policy_vla_ttt/processor_vla_ttt.py`
- 新建：`tests/unit/test_processor_delegation.py`

**接口：**
- 消费：任务 2 的 `VLATTTConfig`。
- 产出：`make_vla_ttt_pre_post_processors(config, dataset_stats=None, **kwargs) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]`。
- 产出：`make_vla_ttt_pre_post_processors_from_pretrained(config, pretrained_path, *, revision=None, dataset_stats=None, dataset_meta=None, preprocessor_overrides=None, postprocessor_overrides=None, preprocessor_config_filename=..., postprocessor_config_filename=...) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]`。
- 两个函数都原样返回上游 factory 结果。

- [ ] **步骤 1：编写失败的委托测试**

将两个上游 factory monkeypatch 为 sentinel；验证相同 config 和全部参数只转发一次，返回同一对象对，并原样传播上游异常。

- [ ] **步骤 2：添加统计量回归测试**

用 7 维 state/action 合成统计量比较插件与上游的新建 pipeline step 类型和序列化状态；再通过临时 processor checkpoint 比较 pretrained 路径，固定 padding/resume 语义而不复制实现。

- [ ] **步骤 3：确认测试失败**

运行：`uv run pytest tests/unit/test_processor_delegation.py -v`

预期：因 processor 模块不存在而失败。

- [ ] **步骤 4：实现薄 factory**

从 LeRobot 0.6.1 公开 Evo-1 processor 模块导入并直接调用 factory；不构造、继承、复制或修改任何 processor step。

- [ ] **步骤 5：运行测试并提交**

运行：`uv run pytest tests/unit/test_processor_delegation.py -v`

预期：全部通过且不访问网络。

```bash
git add src/lerobot_policy_vla_ttt/processor_vla_ttt.py tests/unit/test_processor_delegation.py
git commit -m "feat: delegate VLA-TTT processors to Evo-1"
```

### Task 4：薄 Policy 子类与空扩展接口

**文件：**
- 新建：`src/lerobot_policy_vla_ttt/modeling_vla_ttt.py`、`tests/unit/test_policy.py`
- 修改：`src/lerobot_policy_vla_ttt/__init__.py`、`tests/unit/test_discovery.py`

**接口：**
- 消费：`VLATTTConfig`、`ensure_supported_lerobot_version()`。
- 产出：`class VLATTTPolicy(Evo1Policy)`，其中 `config_class = VLATTTConfig`、`name = "vla_ttt"`。
- 产出：默认空操作 `VLATTTPolicy.install_extensions(self) -> None`，在父类初始化后调用一次。

- [ ] **步骤 1：编写不分配大模型的失败测试**

Monkeypatch 父类初始化、兼容检查和扩展钩子；验证调用顺序为兼容检查、父类初始化、扩展钩子，各调用一次，config 和 `vlm_hub_kwargs` 原样传入父类。

- [ ] **步骤 2：固定空扩展行为**

假父类初始化设置 sentinel `model`；验证构造后仍为同一对象、模型类未替换、Evo-1 全局类属性未变，且 `reset`、`select_action`、`predict_action_chunk` 解析到继承方法。

- [ ] **步骤 3：扩展 discovery 测试**

Mock 模型构造后调用公开 policy factory，验证 `policy.type=vla_ttt` 得到 `VLATTTPolicy` 而非 `Evo1Policy`。

- [ ] **步骤 4：确认测试失败**

运行：`uv run pytest tests/unit/test_policy.py tests/unit/test_discovery.py -v`

- [ ] **步骤 5：实现最小子类**

匹配 Evo-1 0.6.1 构造签名；先检查版本，再调用 `super().__init__` 和扩展钩子。不覆盖 forward、action、queue、reset、optimizer 或模型构造。

- [ ] **步骤 6：运行测试并提交**

运行：`uv run pytest tests/unit/test_policy.py tests/unit/test_discovery.py -v`

预期：通过且不加载 InternVL3。

```bash
git add src/lerobot_policy_vla_ttt tests/unit/test_policy.py tests/unit/test_discovery.py
git commit -m "feat: add thin VLA-TTT policy subclass"
```

### Task 5：严格 Checkpoint 加载与 Evo-1 Warm Start

**文件：**
- 新建：`src/lerobot_policy_vla_ttt/checkpoint.py`、`tests/unit/test_checkpoint_mapping.py`
- 修改：`src/lerobot_policy_vla_ttt/modeling_vla_ttt.py`

**接口：**
- 产出：不可变 `StateDictCompatibility`，含排序后的 `missing_keys`、`unexpected_keys`、`shape_mismatches` 及 `is_compatible: bool`。
- 产出：`compare_state_dicts(source: Mapping[str, Tensor], target: Mapping[str, Tensor]) -> StateDictCompatibility`。
- 产出：`load_state_dict_strict(target: nn.Module, source: Mapping[str, Tensor]) -> None`。
- 产出：`VLATTTPolicy.from_pretrained(pretrained_name_or_path: str | Path, **kwargs) -> VLATTTPolicy`，只接受 `vla_ttt`。
- 产出：`VLATTTPolicy.from_evo1_pretrained(pretrained_name_or_path: str | Path, *, revision: str | None = None, local_files_only: bool = False, **kwargs) -> VLATTTPolicy`。

- [ ] **步骤 1：编写失败的 state-dict 诊断测试**

验证兼容字典精确加载；再用同时含 missing、unexpected、错误 shape 的输入验证单一异常列出全部名称与 source/target shape，且 target 参数未发生部分修改。

- [ ] **步骤 2：编写 mock warm-start 测试**

Mock `Evo1Policy.from_pretrained` 返回小型 policy；验证配置转换、插件构造、严格加载、revision/local-only 转发，以及返回配置的 `policy.type=vla_ttt`。

- [ ] **步骤 3：固定两种加载入口的类型检查**

验证 `from_pretrained()` 接受 `vla_ttt`，对 `evo1` 提示改用 `from_evo1_pretrained()`，并报告其他实际类型；验证 `from_evo1_pretrained()` 只接受 `evo1`。所有拒绝都发生在模型 loader 前。

- [ ] **步骤 4：确认测试失败**

运行：`uv run pytest tests/unit/test_checkpoint_mapping.py -v`

- [ ] **步骤 5：实现事务式严格加载**

先比较 key 集合和 shape，再调用 `target.load_state_dict(source, strict=True)`；以确定顺序聚合全部差异，验证失败时不部分加载。

- [ ] **步骤 6：实现两个加载路径**

模型分配前轻量检查配置。`from_pretrained()` 类型正确时委托父 loader；`from_evo1_pretrained()` 通过 `Evo1Policy.from_pretrained` 加载，只转换配置，构造插件并严格复制 state dict。

- [ ] **步骤 7：运行测试并提交**

运行：`uv run pytest tests/unit/test_checkpoint_mapping.py -v`

预期：通过，不访问网络或真实 checkpoint。

```bash
git add src/lerobot_policy_vla_ttt/checkpoint.py src/lerobot_policy_vla_ttt/modeling_vla_ttt.py tests/unit/test_checkpoint_mapping.py
git commit -m "feat: strictly initialize from Evo-1 checkpoints"
```

### Task 6：自包含插件 Checkpoint 转换

**文件：**
- 修改：`src/lerobot_policy_vla_ttt/checkpoint.py`、`src/lerobot_policy_vla_ttt/__init__.py`
- 新建：`tests/unit/test_checkpoint_conversion.py`

**接口：**
- 消费：`VLATTTPolicy.from_evo1_pretrained()` 与任务 3 的 processor factories。
- 产出：`convert_evo1_checkpoint(source: str | Path, destination: Path, *, revision: str | None = None, local_files_only: bool = False) -> Path`。
- destination 含插件配置/权重、preprocessor、postprocessor、`vla_ttt_conversion.json`，不再依赖 source。

- [ ] **步骤 1：编写失败的完整产物转换测试**

用临时目录和假 policy/processors 验证每个对象保存一次、配置类型为 `vla_ttt`、metadata 记录 source/revision；删除假 source 后仍可从 destination 加载。

- [ ] **步骤 2：测试 destination 与 source 安全性**

非空 destination 必须被拒绝且不覆盖文件；无关 policy 类型须在构造 policy/processor 前失败，并报告接受类型 `evo1` 与实际类型。

- [ ] **步骤 3：确认测试失败**

运行：`uv run pytest tests/unit/test_checkpoint_conversion.py -v`

- [ ] **步骤 4：用 LeRobot 公共 API 实现转换**

从 source 加载上游 processors 并 warm-start policy；全部验证通过后才创建 destination；调用公开 `save_pretrained`，最后原子写入 metadata。不得手工复制 processor 文件或解释其 schema。

- [ ] **步骤 5：联合测试并提交**

运行：`uv run pytest tests/unit/test_checkpoint_mapping.py tests/unit/test_checkpoint_conversion.py tests/unit/test_processor_delegation.py -v`

预期：全部通过。

```bash
git add src/lerobot_policy_vla_ttt tests/unit/test_checkpoint_conversion.py
git commit -m "feat: convert complete Evo-1 checkpoints"
```

### Task 7：真实 Checkpoint 数值等价性与 Round-Trip

**文件：**
- 新建：`tests/integration/conftest.py`、`tests/integration/test_evo1_equivalence.py`
- 修改：`UPSTREAM.md`

**接口：**
- 消费：`VLA_TTT_EVO1_CHECKPOINT`（本地路径或固定 Hub repo）；Hub ID 必须同时提供 `VLA_TTT_EVO1_REVISION`。
- 产出：`evo1_checkpoint() -> tuple[str, str | None]` fixture；缺少 CUDA、bf16 或配置时显式 skip。
- 产出：`make_synthetic_raw_batch(config: Evo1Config, *, batch_size: int = 1) -> dict[str, Any]`，按 input features 构造，不硬编码 camera 名称。

- [ ] **步骤 1：添加 opt-in fixture 与下载告知闸门**

本地缺少 checkpoint 时不得立即下载。先查询 Hub dry-run/metadata，向用户报告 repo、revision、缓存目标、总字节数和最大文件；告知后才能继续。将不可变 revision 写入 `UPSTREAM.md`。

- [ ] **步骤 2：编写等价性测试**

从同一 source 加载原生 Evo-1/processors 和插件；预处理确定性合成 observation，reset 两者，每次 `predict_action_chunk` 前恢复相同 CPU/CUDA RNG state；验证 shape 相同并以明确 bf16 tolerance 调用 `torch.testing.assert_close`。

- [ ] **步骤 3：编写 round-trip 测试**

转换到临时插件 checkpoint，释放原生 policy，再经标准 discovery 加载插件及 processors。以相同 RNG 比较 action chunk；在 `reset()` 前后重复推理，验证 action queue 清空。

- [ ] **步骤 4：运行集成测试**

运行：`uv run pytest -m integration tests/integration/test_evo1_equivalence.py -v -s`

预期：RTX PRO 5000 72GB 上通过；缺少资产/GPU 时明确 skip，CPU 测试不得隐式下载。

- [ ] **步骤 5：记录运行时矩阵并提交**

在 `UPSTREAM.md` 记录 checkpoint/InternVL3 revision、PyTorch、Transformers、CUDA、驱动、GPU、dtype 和实测 tolerance。

```bash
git add tests/integration/conftest.py tests/integration/test_evo1_equivalence.py UPSTREAM.md
git commit -m "test: verify Evo-1 numerical equivalence"
```

### Task 8：标准 LeRobot 训练与 Rollout 接口

**文件：**
- 新建：`README.md`、`tests/integration/test_rollout_smoke.py`
- 修改：`UPSTREAM.md`

**接口：**
- 消费：未经修改的 `lerobot-train`、`lerobot-rollout`。
- 消费：`VLA_TTT_LIBERO_CHECKPOINT`、`VLA_TTT_LIBERO_REVISION` 及 LeRobot 0.6.1 所需 benchmark 资产变量。
- 不产出自定义 trainer、dataset wrapper、rollout loop、transport 或 action queue。

- [ ] **步骤 1：编写 README**

说明安装、`policy.type=vla_ttt` 的 LIBERO Stage-1 训练、Python checkpoint 转换、插件 resume、标准 rollout、测试分层、支持版本、非隶属声明和下载前告知规则；标明需要外部资产的命令。

- [ ] **步骤 2：添加 CLI 配置冒烟测试**

Mock 重型构造器后只运行参数/配置解析；验证训练选择 `VLATTTConfig` 且保留 LeRobotDataset，rollout 从插件 checkpoint 选择插件 policy 与 processor factory。

- [ ] **步骤 3：添加 opt-in 单 episode LIBERO 测试**

用标准 rollout 入口运行 base strategy、batch size 1、一个 episode、固定 seed；验证 reset、至少一个正确 shape 的 action 和正常结束，不复制 rollout loop。

- [ ] **步骤 4：执行下载告知闸门**

获取任何 LIBERO policy、dataset、环境资产或 benchmark bundle 前，报告仓库/URL、revision/checksum、目标位置和 dry-run 字节数；使用后记录在 `UPSTREAM.md`，资产保留在 Git 外。

- [ ] **步骤 5：运行测试**

运行：`uv run pytest tests/unit -v`

预期：禁网时全部通过。

运行：`uv run pytest -m "integration and libero" tests/integration/test_rollout_smoke.py -v -s`

预期：资产已配置时通过，否则明确 skip。

- [ ] **步骤 6：确认没有上游副本或大型资产**

运行：`git status --short && git ls-files | sort && find . -type f -size +10M -not -path './.git/*' -print`

预期：仅跟踪预期源码/文档/测试/lock 文件，没有超过 10 MiB 的文件。

- [ ] **步骤 7：提交**

```bash
git add README.md UPSTREAM.md tests/integration/test_rollout_smoke.py
git commit -m "docs: document LeRobot training and rollout workflow"
```

### Task 9：最终验收

**文件：**
- 只修改验证发现的缺陷所必需的文件，不扩大范围。

**接口：**
- 消费：任务 1–8 的全部接口与测试。
- 产出：满足规格七项验收标准的干净、可复现里程碑。

- [ ] **步骤 1：运行格式与静态检查**

运行：`uv run ruff check . && uv run ruff format --check .`

预期：通过。

- [ ] **步骤 2：运行无网络测试集**

运行：`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run pytest tests/unit -v`

预期：通过，证明单元测试不下载权重或数据。

- [ ] **步骤 3：构建并检查 wheel**

运行：`uv build && unzip -l dist/lerobot_policy_vla_ttt-*.whl`

预期：wheel 只含插件源码和分发元数据，不含测试、模型、数据、输出或 LeRobot 源码树。

- [ ] **步骤 4：在全新环境验证安装**

将 wheel 安装到临时 Python 3.12 环境，不依赖 LeRobot checkout；导入插件并运行公开 `vla_ttt` discovery 检查。

预期：导入与发现通过。

- [ ] **步骤 5：若资产已告知并配置，运行 GPU 测试**

运行：`uv run pytest -m integration tests/integration -v -s`

预期：等价性、round-trip、LIBERO smoke 全部通过；否则准确报告 skip 原因，不宣称已完成 GPU 验收。

- [ ] **步骤 6：检查验收证据和工作树**

运行：`git status --short && git log --oneline --decorate -12`

预期：无无法解释的改动，每个已完成任务有一个聚焦提交。

- [ ] **步骤 7：如有必要，提交验证修复**

```bash
git add <only-the-files-fixed-during-verification>
git commit -m "fix: address plugin acceptance findings"
```

若验证没有引起代码或文档修改，则跳过此提交。
