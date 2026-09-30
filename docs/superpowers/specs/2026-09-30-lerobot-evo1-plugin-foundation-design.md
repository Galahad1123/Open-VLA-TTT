# LeRobot Evo-1 插件基础设计

日期：2026-09-30

## 1. 目标

建立一个独立、可安装、可开源的 LeRobot policy plugin，作为 VLA+TTT 研究平台的第一个子项目。该插件必须复用 LeRobot 已有的 Evo-1 模型、LeRobotDataset、Evo-1 processors、checkpoint 约定和 rollout 实现，而不复制 LeRobot 仓库或其他 policy 的源代码。

首个里程碑只建立可靠的 Evo-1 基线和扩展边界，不实现 RoboTTT、Titans、长序列采样或 TBPTT。完成标准是：插件能被 LeRobot 自动发现；在关闭所有未来扩展时，其训练、推理和 rollout 行为与同版本的 LeRobot Evo-1 一致；项目自身保持小型、清晰且可独立发布。

## 2. 范围

### 2.1 本子项目包含

- 初始化独立 Python 包 `lerobot-policy-vla-ttt`。
- 将 LeRobot 固定为显式上游依赖，而不是 vendor、submodule 或仓库 fork。
- 定义 `VLATTTConfig` 和 `VLATTTPolicy`，分别扩展 LeRobot 的 `Evo1Config` 和 `Evo1Policy`。
- 通过 LeRobot 的 out-of-tree policy discovery 注册 `policy.type=vla_ttt`。
- 直接复用 LeRobot 的 Evo-1 模型、processor factory、LeRobotDataset、训练 CLI、checkpoint API 和 rollout CLI。
- 提供普通 Evo-1 checkpoint 到插件 checkpoint 的显式初始化入口。
- 建立无需下载 1B 模型即可运行的单元测试，以及使用真实 checkpoint 的可选 GPU 集成测试。
- 记录并锁定经过验证的 LeRobot、Python、PyTorch、Transformers 和 CUDA 版本。

### 2.2 本子项目不包含

- RoboTTT、TTT-Linear、TTT-MLP、Titans 或 Gated DeltaNet。
- 连续轨迹 sampler、sequence action forcing 或 TBPTT。
- GR00T 后端。
- RoboMME、LIBERO 之外的新评测适配。
- 多卡或分布式长序列训练。
- 对 LeRobot 主仓库的修改或上游 PR。

这些能力将在 Evo-1 基线等价性得到验证后作为后续独立子项目实现。

## 3. 方案选择

采用 LeRobot 官方支持的 out-of-tree policy plugin。项目通过正常 Python 依赖引用 LeRobot；项目仓库只包含自身的配置、薄包装、兼容性检查、checkpoint 初始化逻辑和测试。

不选择以下方案：

- 不 fork LeRobot：避免长期维护与上游的大范围冲突。
- 不复制 Evo-1 核心文件：避免失去上游修复，并减少许可证与同步负担。
- 不使用 Git submodule：submodule 会把源码布局耦合进项目，并不能提供稳定的 Python API 边界。
- 不在首个里程碑重写 trainer：普通 Evo-1 基线应先完全复用 `lerobot-train`；只有长序列状态生命周期确定后，才增加专用 trainer。

## 4. 上游版本策略

初始兼容目标为 LeRobot `0.6.1`，Python `>=3.12,<3.14`。直接依赖 `lerobot[evo1,training]==0.6.1`，LIBERO 支持作为单独 optional extra 提供。项目提交 `uv.lock`，保证开发和 CI 使用同一依赖解析结果。LeRobot 仓库 `main` 中尚未发布的版本号不作为依赖目标。

项目不支持未锁定的 LeRobot `main`。升级 LeRobot 必须通过独立变更完成，并重新运行基线等价性、checkpoint round-trip 和真实 rollout 测试。

根目录 `UPSTREAM.md` 记录：

- LeRobot release 和 commit；
- Evo-1 checkpoint revision；
- InternVL3 checkpoint revision；
- 已验证的 CUDA、驱动和 GPU；
- 所使用的公开 benchmark 数据 revision。

## 5. 包结构

```text
vla-ttt/
├── pyproject.toml
├── uv.lock
├── README.md
├── LICENSE
├── UPSTREAM.md
├── src/
│   └── lerobot_policy_vla_ttt/
│       ├── __init__.py
│       ├── configuration_vla_ttt.py
│       ├── modeling_vla_ttt.py
│       ├── processor_vla_ttt.py
│       ├── checkpoint.py
│       └── compatibility.py
├── tests/
│   ├── unit/
│   │   ├── test_discovery.py
│   │   ├── test_configuration.py
│   │   ├── test_processor_delegation.py
│   │   └── test_checkpoint_mapping.py
│   └── integration/
│       ├── test_evo1_equivalence.py
│       └── test_rollout_smoke.py
└── docs/
```

首个里程碑不预先创建空的 `memory/`、`sequence/` 或 `trainers/` 层级。需要实现对应能力时再增加，避免提前固化错误接口。

## 6. 组件设计

### 6.1 `VLATTTConfig`

`VLATTTConfig` 继承 LeRobot `Evo1Config`，以 `vla_ttt` 类型注册。首个里程碑不增加 memory 超参数，只增加项目元数据和严格的上游兼容版本字段。

所有 Evo-1 参数继续由父配置定义，包括图像特征、state/action 维度、action chunk、InternVL3、训练 stage 和优化器设置。这样普通基线的命令行参数、序列化格式和 LeRobot processor 均能直接复用。

未来 memory 配置将采用嵌套配置对象，而不是持续向顶层添加互不相关的字段；该扩展不属于本里程碑。

### 6.2 `VLATTTPolicy`

`VLATTTPolicy` 继承 `Evo1Policy`，首个里程碑只承担以下职责：

- 声明插件名称和配置类；
- 执行上游版本兼容性检查；
- 保留一个受测试的模型扩展钩子；
- 支持插件原生 checkpoint 的保存和加载；
- 提供从普通 Evo-1 checkpoint 初始化的显式入口。

它不复制图像收集、state/action padding、AMP、action queue、flow-matching 或 rollout 逻辑。默认模型仍是 LeRobot 的 `Evo1Model`，因此未启用未来扩展时不应产生数值差异。

扩展钩子是显式方法，例如 `install_extensions()`，默认实现为空操作。后续 TTT 子项目将通过包装 action-head blocks 使用该钩子，而不是修改 LeRobot 安装目录或 monkey-patch 全局类。

### 6.3 Processor

`processor_vla_ttt.py` 只委托 LeRobot Evo-1 的 processor factory。插件不得复制归一化、图像处理、feature mapping 或 action postprocessing 逻辑。

委托层存在的原因是保持插件发现约定，并为未来增加 memory reset metadata 留下单一入口。在首个里程碑中，给定相同配置时，委托层返回的 processor 行为必须与 Evo-1 processor 一致。

### 6.4 Dataset

普通基线训练直接使用 LeRobotDataset 和 `lerobot-train`。插件不包装、不继承、不复制 LeRobotDataset。

后续长序列子项目将新增组合式 `TrajectorySequenceDataset`：它持有一个 LeRobotDataset，并基于 episode index 生成连续控制时刻。该未来 wrapper 不改变底层数据格式，也不会成为当前基线的依赖。

### 6.5 Rollout

插件依赖 LeRobot policy discovery，使标准 `lerobot-rollout` 能通过插件 checkpoint 中的 `policy.type=vla_ttt` 自动构造 policy 和 processor。项目不实现自己的机器人设备循环、action queue 或 transport。

首个 rollout smoke test只验证：

- 插件可被发现；
- checkpoint 可以加载；
- policy reset 能清空 LeRobot 原有 action queue；
- 单个 observation 能产生形状正确的 action chunk。

不在 CI 中启动真实机器人。

### 6.6 Checkpoint 初始化

普通 Evo-1 checkpoint 与插件 checkpoint 的用途必须明确区分：

- `from_pretrained()` 只加载已经保存为 `vla_ttt` 类型的插件 checkpoint。
- `from_evo1_pretrained()` 专门加载普通 Evo-1 checkpoint，并创建对应的插件 policy。

`from_evo1_pretrained()` 执行严格的键、形状和配置检查。首个里程碑中，转换后所有模型参数必须一一匹配，不能使用无报告的 `strict=False`。转换结果保存为插件格式后，后续加载不再依赖原始 checkpoint。

未来加入 memory 参数时，该入口先严格加载基础 Evo-1 权重，再显式初始化新增模块和零门控，并输出新增参数清单。

## 7. 数据流

普通训练和 rollout 的数据流保持为：

```text
LeRobotDataset / robot observation
        ↓
LeRobot Evo-1 preprocessor
        ↓
VLATTTPolicy（首个里程碑为空扩展）
        ↓
LeRobot Evo1Model
        ↓
LeRobot Evo-1 postprocessor / action queue
        ↓
loss 或 rollout action
```

首个里程碑不得改变 feature name、归一化统计、action chunk 语义或 flow-matching convention。后续序列训练会在 dataset 与 policy 之间增加连续轨迹维度，但不会把 action-chunk 维度误作机器人历史时间维。

## 8. 命令行体验

安装项目后，LeRobot 自动发现插件：

```bash
uv sync --extra train

uv run lerobot-train \
  --policy.type=vla_ttt \
  --policy.training_stage=stage1 \
  --dataset.repo_id=lerobot/libero \
  --output_dir=outputs/evo1-baseline
```

插件 checkpoint 使用标准 rollout：

```bash
uv run lerobot-rollout \
  --policy.path=/path/to/vla-ttt-checkpoint \
  --strategy.type=base \
  --robot.type=<robot-type> \
  --task="<instruction>"
```

LIBERO 等仿真依赖通过独立 extra 安装，不进入默认开发环境。

## 9. 错误处理

- LeRobot 版本不在已验证范围时，在 policy 构造阶段给出包含当前版本和支持版本的明确错误。
- 缺少 `evo1`、`training` 或 benchmark extra 时，错误信息必须给出具体安装命令。
- checkpoint 类型错误时，提示使用 `from_evo1_pretrained()`，而不是静默尝试加载。
- 参数键或形状不匹配时，完整报告 missing、unexpected 和 mismatched keys，然后中止。
- processor 的输入 feature 与配置不符时，沿用 LeRobot 的原始校验，不在插件中自动猜测或重命名。
- 集成测试所需的 checkpoint、GPU 或数据不存在时，测试显式 skip；单元测试不得因此失败。

## 10. 测试策略

### 10.1 CPU 单元测试

- 安装后的包可被 LeRobot policy discovery 找到。
- `VLATTTConfig` 能序列化并恢复所有 Evo-1 父配置字段。
- processor factory 确实委托 Evo-1 factory。
- 普通 Evo-1 state dict 到插件 state dict 的映射不存在静默遗漏。
- 不兼容的 LeRobot 版本和错误 checkpoint 类型能 fail fast。

CPU 单元测试使用小型 fake modules 或 mock，不实例化 InternVL3-1B。

### 10.2 GPU 集成测试

- 固定 checkpoint revision 和固定输入下，原生 `Evo1Policy` 与 `VLATTTPolicy` 共用相同 flow-matching 初始噪声时，输出在规定容差内一致。
- 插件 checkpoint save/load round-trip 后输出一致。
- 在单个 LIBERO episode 上完成 rollout smoke test。

GPU 集成测试通过 marker 隔离，不作为每次本地快速测试的强制项。

## 11. 72GB 单卡开发约束

RTX PRO 5000 72GB 足以完成本里程碑和后续冻结 VLM 的早期 TTT 实验。首个里程碑优先使用 bf16 和 Evo-1 Stage 1 设置，不尝试全量 VLM 微调。真实等价性测试批量为 1，避免测试基础设施本身占用过多显存。

后续长序列训练将优先缓存冻结 VLM 的视觉语言 embedding；是否缓存及缓存格式将在长序列子项目中单独设计，不能在当前基线层预设。

## 12. 开源与维护

- 项目采用与 LeRobot 兼容的 Apache-2.0 许可证。
- README 明确说明项目依赖 LeRobot，但不隶属于 Hugging Face 或 Evo-1 作者。
- 不把 LeRobot、Evo-1 或模型权重提交到仓库。
- `uv.lock`、`UPSTREAM.md` 和集成测试共同定义受支持的上游组合。
- CI 只测试一个受支持的 LeRobot release；上游新版本由人工发起兼容升级。
- 项目代码不得依赖 LeRobot 仓库 checkout 的相对路径。

## 13. 验收标准

首个子项目完成时必须同时满足：

1. 一个全新环境能从本仓库安装 plugin，而无需 clone LeRobot。
2. `policy.type=vla_ttt` 可被 LeRobot 自动发现。
3. 普通 Evo-1 checkpoint 能通过显式转换入口严格初始化插件。
4. 插件复用 LeRobotDataset、Evo-1 processors、checkpoint API 和 rollout；仓库内没有它们的复制实现。
5. 默认插件与原生 LeRobot Evo-1 通过固定输入的数值等价性测试。
6. CPU 单元测试全部通过，可选 GPU 集成测试在 RTX PRO 5000 72GB 上通过。
7. README、锁文件和 `UPSTREAM.md` 足以让第三方复现实验环境。

满足这些条件后，下一子项目才开始设计 TTT memory interface 与 action-head block 包装。
