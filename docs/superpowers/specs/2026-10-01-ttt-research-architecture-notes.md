# Evo1 GR00T 与 TTT 研究设计记录

**记录日期：** 2026-10-01<br>
**架构约定更新：** 2026-10-05；记忆架构改为与 `evo1_groot` 并列的模型目录。<br>
**状态：** 讨论结论与参考代码核对记录；不是已完成实现，也不是冻结的实施规格。

本文记录新增 `evo1_groot` baseline、RoboTTT 与 Titans 风格记忆的实现参考、待决策事项和代码布局，供后续设计、实施与消融对照。总体实验背景见[研究路线图](../plans/2026-10-01-vla-ttt-research-roadmap.md)。

核心安排是：保留原生与本地 Evo-1 的等价性实验；另建独立的 `evo1_groot` 无记忆 baseline；以它为对照构建并列的记忆模型变体，将记忆接入结构与 fast-weight 更新规则分开配置。各变体可复用基座组件，也可实现自己的组件。优先落实 RoboTTT-style 与普通梯度更新，之后扩展组合实验。

## 已确认的研究边界

### 三种 baseline 的职责

| 模型 | 职责 | 与其他模型的关系 |
| --- | --- | --- |
| LeRobot 原生 Evo-1 | 提供原生实现参照 | 保留现有 `policy.type=evo1` 路径 |
| 本地 Evo-1 | 检查源码迁移、checkpoint、训练和评测管线是否引入差异 | 保留现有本地 `policy.type=vla_ttt` baseline 行为；另一个会话正在推进对照 |
| `evo1_groot` | 复用 Evo-1 组件构建轻量 GR00T-style VLA，提供新的无记忆研究基座 | **作为单独的 baseline 模型实现，不替换本地 Evo-1，不将其混入原生/本地等价性结论** |

`evo1_groot` 应具备独立的配置、模型入口、checkpoint 身份与实验记录。具体 LeRobot 注册名称和配置字段在实施规格中确定；模型身份不能仅靠某个隐藏的 TTT 开关区分。

新基座计划复用 Evo-1 的 VLM、兼容的状态/动作编码组件和轻量尺寸，参考 GR00T 的 token 交互与动作头组织。改变 attention、时间条件或 readout 后，不再宣称与原 Evo-1 数值等价。可复用权重、需转换权重、新初始化参数应逐项记录。

记忆实验的主要对照是 `evo1_groot` 无记忆版本与其带记忆版本。原 Evo-1 的结果仍可用于比较架构改造的影响，但不能单独作为记忆收益的归因依据。

### 两个可组合的研究维度

| 维度 | 已确认的研究选项 | 职责 |
| --- | --- | --- |
| 记忆接入结构 | RoboTTT-style、MAC、MAG、MAL | 定义 memory 与 attention 的连接、读写输入、读写顺序及输出融合 |
| 内循环更新规则 | RoboTTT 普通梯度下降、Titans-style 动量与遗忘 | 定义梯度如何更新 fast weights，以及是否维护动量和衰减旧记忆 |

目标是允许研究者组合四种结构与两种更新规则，并替换其中一个维度研究新方法。动量、遗忘可分别关闭以支持后续消融。配置空间不意味着当前已实现或验证全部八种组合。

RoboTTT、MAC、MAG、MAL 分别在 `models/robottt/`、`models/mac/`、`models/mag/`、`models/mal/` 中组织实现，与 `models/evo1_groot/` 并列；不再约定为该基座内部的四个 layout 单文件。每种架构可以拥有自己的配置、模型组装、动作头、推理状态管理及其他组件，按实际复杂度拆分文件或子包。MAC/MAG/MAL 目录表示采用相应记忆连接方式的 VLA 模型变体，不自动代表完整 Titans 复现。

`RoboTTT-style + Titans update` 属于组合实验。只有同时核对所选 Titans 架构、记忆网络、persistent memory、归一化和更新细节后，才能说明与对应论文变体的符合程度；不能仅替换更新器便声称完整复现 Titans。

## 参考资料与固定版本

论文用于确定机制；社区实现优先用于核对工程细节；TTT-Video 作为补充。社区实现与论文不一致时，应记录差异，不默认继承其行为。

| 资料 | 用途 | 阅读版本或位置 |
| --- | --- | --- |
| [RoboTTT 论文](https://arxiv.org/html/2607.15275) | 机制、动作头接入、sequence action forcing、TBPTT | arXiv v1 |
| [RoboTTT 官方项目页](https://research.nvidia.com/labs/gear/robottt/) | 对照方法说明和架构图 | 本轮检查未找到官方代码入口，不据此断言官方代码一定不存在 |
| 用户笔记 `RoboTTT.md` | 补充理解 | 主工作区根目录；不替代论文和源码证据 |
| [RoboTTT 社区实现](https://github.com/lucidrains/robo_ttt/tree/98eb8837f66ec28a0d131fc8bfabb8d6c7bb017e) | 优先核对 memory、wrapper、采样与训练接口 | `98eb8837f66ec28a0d131fc8bfabb8d6c7bb017e` |
| [TTT-Video 论文](https://arxiv.org/html/2504.05298v1)及[官方代码](https://github.com/test-time-training/ttt-video-dit/tree/e43707e9b8594ace5a73c4a4438f486d5d17729c) | DiT 接入、时间聚合、MLP 更新运算 | `e43707e9b8594ace5a73c4a4438f486d5d17729c` |
| [Titans 论文](https://papers.nips.cc/paper_files/paper/2025/file/a4ca07aa108036f80cbb5b82285fd4b1-Paper-Conference.pdf) | 动量、遗忘以及 MAC/MAG/MAL | NeurIPS 2025 版本 |
| [Titans 社区实现](https://github.com/lucidrains/titans-pytorch/tree/1d40c445fa1794fb28582c721786482b5b683e47) | 函数式更新、状态、分块读写 | `1d40c445fa1794fb28582c721786482b5b683e47` |
| [NVIDIA Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T/tree/51d4c89f72fda44cbf77285c6a8114b52676b8a1) | DiT 与 action-head 组织 | `51d4c89f72fda44cbf77285c6a8114b52676b8a1` |
| 本地安装的 LeRobot | GR00T 接入与既有基础设施 | `lerobot==0.6.1`，`lerobot/policies/groot/` |

这些是本轮阅读版本，不代表后续直接采用全部依赖或复制整个仓库。实际借用代码时记录文件、commit、修改及许可证归属。

## 需要实现的内容与可用参考

| 实现内容 | 可参考代码 | 已知可借鉴部分 | 仍需本项目完成 |
| --- | --- | --- | --- |
| `evo1_groot` 无记忆模型 | LeRobot `groot_n1_7.py`、`action_head/cross_attention_dit.py`；NVIDIA `gr00t/model/modules/dit.py`、`gr00t/model/gr00t_n1d7/gr00t_n1d7.py` | state/action token 组织、交替 self/cross attention、时间条件、逐位置动作解码 | 轻量尺寸、Evo-1 组件复用、权重映射、独立注册与 baseline 验证 |
| fast model 与可微 KV 写入 | 社区 RoboTTT `MemoryKeyValueBind`；Titans `neural_memory.py`、`memory_models.py` | `functional_call`、`grad`、`vmap`，小型 MLP，显式参数计算 | 明确损失归约、学习率参数化、网络结构和训练梯度路径 |
| RoboTTT 时间聚合与输出门控 | 社区 `TTTWrapper`；TTT-Video `ttt_layer.py` 与 `cogvideo/dit.py` | 沿时间更新状态、小门控接入预训练网络 | 显式插入位置、正确残差语义、因果性与 baseline 回归 |
| 动量与遗忘更新 | Titans `neural_memory.py` | 梯度驱动的动量状态和旧权重衰减 | 与普通梯度更新共享输入约定；独立开关和退化情形验证 |
| MAC 结构 | Titans 论文及 `mac_transformer.py` | 记忆检索、attention、写入的组织参考 | 迁移到动作 DiT，明确持久 tokens、检索位置和融合方式 |
| MAG 与 MAL 结构 | Titans 论文 | 并行门控、顺序层式组合的定义 | 本轮尚未核实到可直接对照的独立 MAG/MAL 社区模块，需继续对照论文实现 |
| Sequence Action Forcing | 社区 `sample_action_times()`、`RoboTTT.forward()`；GR00T action head | 为每个序列位置独立采样 flow 时间 | 与新动作头噪声分布、时间条件和动作 mask 对齐 |
| 上下文与监督分离 | 社区 `loss_mask` 路径 | 屏蔽动作监督但保留 memory 计算 | 独立的有效数据、写入、监督和 reset 标记；padding 不得污染状态 |
| TBPTT | 社区 wrapper 中的周期性 detach | 截断 fast-weight 梯度，继续携带状态值 | 分段前向/反向、图释放、loss 归一化、optimizer 更新时机 |
| 去噪状态传递 | 社区 `_sample()`、`_forward_with_ttt()`；TTT-Video `ttt_layer.py`、`cogvideo/sampler.py` | 两种不同的状态处理方式可供对照 | 选定本平台的状态提交协议，并验证训练与推理匹配 |
| 机器人动作执行与序列采样 | 现有 LeRobot dataset/runner、项目 benchmark client | 数据与闭环执行基础设施 | 预测 horizon、执行 horizon、采样 stride、观测处理频率及重叠标签处理 |
| 评测时局部求梯度 | 社区函数式更新；PyTorch autograd | 内循环仅求 fast-weight 梯度 | 调整本地评测梯度上下文，验证状态更新且 slow weights 不变 |

源码入口：[社区 RoboTTT](https://github.com/lucidrains/robo_ttt/blob/98eb8837f66ec28a0d131fc8bfabb8d6c7bb017e/robo_ttt/robo_ttt.py)、[社区测试](https://github.com/lucidrains/robo_ttt/blob/98eb8837f66ec28a0d131fc8bfabb8d6c7bb017e/tests/test_robo_ttt.py)、[TTT-Video layer](https://github.com/test-time-training/ttt-video-dit/blob/e43707e9b8594ace5a73c4a4438f486d5d17729c/ttt/models/ssm/ttt_layer.py)、[Titans neural memory](https://github.com/lucidrains/titans-pytorch/blob/1d40c445fa1794fb28582c721786482b5b683e47/titans_pytorch/neural_memory.py)、[Titans MAC](https://github.com/lucidrains/titans-pytorch/blob/1d40c445fa1794fb28582c721786482b5b683e47/titans_pytorch/mac_transformer.py)。

## 参考实现的核对结论与限制

### RoboTTT 社区实现

- **基本机制可参考。** 对 `[B,T,N,D]` 沿 `T` 顺序处理，各时刻的 token 集合参与 KV 学习，更新后读取；训练入口为 `B×T` 位置独立采样噪声等级。
- **存在重复残差。** 所读版本的 wrapper 已返回 `x + gate × memory(x)`，外层 hook 再加一次 `x`。抽取未修改的相关函数，使用最小 CPU 输入及替代记忆模块验证：零门控时 wrapper 返回 `x`，完整 hook 返回 `2x`。这是局部复现，不是该仓库完整集成测试。
- **`learned_forget` 名称不代表 Titans 遗忘。** 该分支缩放新增梯度更新，没有按 `(1−α)W` 衰减旧记忆。其作用更接近写入强度控制。
- **去噪与记忆时钟绑定于 hook 调用。** 局部计数验证中，一次挂载点调用推进一步，四次推进四步。若挂载点处于去噪循环内部，去噪次数会影响累计更新与位置计数。实际结果取决于底层模型和挂载位置。
- **TBPTT 支持不等于完整训练调度。** 状态 detach 并不自动释放整段输出及其余网络的计算图；必须另行实现分段反向与显存验证。
- **未提供完整机器人协议。** `sample()` 委托底层模型，未统一定义执行 horizon、轨迹 stride 或队列期间观测处理。已有多步采样测试未充分验证上述状态时钟语义。

这些发现仅针对固定版本。后续采用新版本时需复核，不能用测试文件存在或张量 shape 正确替代机制一致性验证。

### TTT-Video

DiT 局部 attention 与 TTT 长时间聚合的组合具有参考价值。但其双向扫描和视频位置编码不直接用于机器人跨时间因果路径；其归一化、残差重建目标也需与 RoboTTT 的 KV 目标逐项比较。

所读实现每次 TTT forward 从学习到的初始参数建立 fast weights，并在该次输入序列内扫描。它没有直接提供机器人跨动作决策持续传递状态的同等接口，因此不能用其默认行为解决全部部署问题。

### Titans

普通梯度更新和 Titans-style 更新可共享梯度计算及 fast-model 运算；后者另维护动量并衰减旧权重。MAC/MAG/MAL 的读写时机属于结构层职责，不放入更新器。

Persistent memory 是另一类学习到的固定参数，不能与 episode fast weights 或 RoboTTT register tokens 自动视为同一事物。社区实现还含论文之外的可选扩展，需要明确开关和出处。

## 评测梯度上下文

现有 [policy server 推理入口](../../../src/lerobot_policy_vla_ttt/evaluation/policy.py) 使用 `torch.inference_mode()`。普通 baseline 可以使用这一模式，但自动微分版本的 TTT 仍需在推理中对小型 fast model 求梯度。

`model.eval()` 控制 dropout 等模块行为，不禁止梯度；`no_grad()` 可以局部重新开启梯度；`inference_mode()` 的限制更强，仅在内部添加 `enable_grad()` 不足以恢复正常求导。参见 [PyTorch 2.11 autograd 文档](https://docs.pytorch.org/docs/2.11/notes/autograd.html#locally-disabling-gradient-computation)。

本轮在 PyTorch `2.11.0+cu130` 环境中使用 CPU 对 `L(w)=(3w−1)^2, w=2` 做局部验证，正确梯度为 30：

| 上下文与求导方法 | 观察结果 |
| --- | --- |
| `no_grad` 内局部 `enable_grad`，`autograd.grad` | 30 |
| `inference_mode` 内局部 `enable_grad`，`autograd.grad` | 报错 |
| `no_grad` 下 `torch.func.grad` | 30 |
| `inference_mode` 下 `torch.func.grad` | 0 |

这些是最小例子的实测结果，不是所有 PyTorch 版本或所有 TTT 后端的普遍结果。解析梯度后端可能无需运行时自动微分，应单独验证。

拟采用的推理约定：模型保持 eval；slow weights 不更新；仅在 fast-weight 更新处启用必要的自动微分；运行状态在适当边界 detach，避免部署期间计算图随 episode 增长。训练时则需保留片段内的外层梯度，包括经内循环更新回传到 `W0` 的路径。新路径不能改变现有 Evo-1 baseline 的评测行为。

## 待敲定事项

以下事项尚未定案，后续实施规格需逐项给出选择、依据与验证方式。

| 事项 | 需要确定的内容 | 验证或决策依据 |
| --- | --- | --- |
| 新基座结构与初始化 | 轻量宽度、层数、self/cross attention 排列、时间条件、readout、权重复用边界 | GR00T 源码与 Evo-1 参数映射；先建立无记忆 baseline |
| 独立模型身份 | LeRobot 注册名、配置 schema、checkpoint 类型和加载入口 | 三个 baseline 不可混淆，加载错误能显式报告 |
| RoboTTT 插入位置 | 每个 attention 子层还是一组 self/cross block 后插入；门控位置、register 数量 | 对照论文、实际 DiT 层级和门控关闭回归 |
| fast model 细节 | head 数、MLP 宽度、bias、归一化、RoPE 索引、损失归约、学习率参数化 | 论文确定部分与工程补充选择分别标注；逐步数值对照 |
| 去噪状态协议（首版已选） | 沿用社区方案，每次去噪累计更新；其他提交方式留作消融 | 不等同于已核实官方协议；记录训练/部署更新次数与成本，见下方专项记录 |
| 时间尺度 | 环境步、动作决策步、去噪步、TTT mini-batch、TBPTT segment 的单位 | 配置和日志分别记录，不能共用一个含糊的 `step` |
| 动作与轨迹采样 | 预测 horizon、执行 horizon、训练 stride、队列期间是否写入新观测 | 首版保留论文完整 chunk 特征路径；审计重叠标签信息与训练/部署差异，不预先宣称无泄漏 |
| mask 与 episode 边界 | 有效数据、写入、动作监督、reset 的区别；无标签上下文如何构造输入 | padding 不写入；上下文可无监督；批内轨迹不串状态 |
| Titans 更新 mask | 禁止写入时是否同时停止动量和遗忘 | 仅将梯度置零不一定保持状态不变，需明确语义 |
| TBPTT 调度 | segment 长度、分段 backward、loss 归一化、外层 optimizer 更新边界 | 图及时释放；状态值跨段延续；`W0` 梯度符合设计 |
| MAC/MAG/MAL 迁移 | persistent tokens、attention 可见范围、检索/写入输入、门控与读写顺序 | 保留结构差异；跨环境时间不能看到未来观测 |
| 数值与性能 | fast-state 精度、混合精度、自动微分或解析梯度、优化 kernel 的引入顺序 | 先有可读参考实现，再核对输出、状态及外层梯度 |
| 实验公平性 | 新基座训练预算、参数量、无记忆/短历史对照、评测种子和 checkpoint 规则 | 区分架构变化、额外算力和真正的记忆收益 |

2026-10-05 用户确认：首版使用论文描述的完整 register/state/action attention 隐藏特征，去噪时沿用社区逐次累计更新；“一次决策只提交一次”、仅已执行前缀写入等改造留作后续研究。论文文字与已读代码尚不足以唯一确定官方去噪状态调度及完整采样/执行配置。相关证据、重叠 chunk 风险与验证建议集中记录于 [TTT 更新时机与隐藏特征专项记录](2026-10-05-robottt-update-timing-and-hidden-features.md)，不将疑虑当成已证实缺陷，也不宣称首版已排除这些风险。

## 代码布局约定

下面是已确认的目录层级与目标职责划分，均属于 foundation 中的同一个项目。`evo1_groot` 列出当前主要文件；其他模型内部文件仅为示意，可以按研究需要扩展或拆分，不限制为单文件，也不要求立即建立所有空目录。配置、policy 和 processor 入口命名应满足 LeRobot factory 约定。

```text
src/lerobot_policy_vla_ttt/
├── configuration_vla_ttt.py       # 现有本地 Evo-1 配置
├── modeling_vla_ttt.py            # 现有本地 Evo-1 入口
├── evo1/                         # 现有 Evo-1 固定快照
│
├── ttt/
│   ├── functional.py             # 显式参数前向与可微梯度运算
│   ├── fast_models.py            # Linear / MLP 基础组件
│   ├── state.py                  # 状态树复制、detach、reset 等操作
│   ├── memory.py                 # query 读取、KV 写入及更新器组合
│   └── updates/
│       ├── gradient_descent.py   # RoboTTT 普通梯度更新
│       └── momentum_decay.py     # Titans-style 动量与遗忘
│
├── models/
│   ├── evo1_groot/              # 独立无记忆 baseline，可供其他模型复用组件
│   │   ├── configuration_evo1_groot.py
│   │   ├── modeling_evo1_groot.py   # LeRobot policy 入口
│   │   ├── processor_evo1_groot.py
│   │   ├── model.py                # 模型组装
│   │   ├── action_head.py          # 轻量 GR00T-style 动作头
│   │   └── tokens.py               # state/action/register token 组织
│   ├── robottt/                 # RoboTTT-style VLA，自行组织所需组件
│   │   ├── configuration_robottt.py
│   │   ├── modeling_robottt.py     # 独立 policy 入口
│   │   ├── action_head.py          # 记忆与 attention 的连接
│   │   ├── inference.py            # 去噪、动作执行与记忆状态提交协议
│   │   └── ...                     # 按需拆分模型、tokens、状态管理等
│   ├── mac/                     # MAC-style VLA，内部按需拆分文件/子包
│   ├── mag/                     # MAG-style VLA，内部按需拆分文件/子包
│   └── mal/                     # MAL-style VLA，内部按需拆分文件/子包
│
├── training/
│   ├── sequence.py               # 连续轨迹、边界与 masks
│   ├── flow_matching.py          # 序列加噪与动作监督
│   └── tbptt.py                  # 分段前反向与状态传递
│
└── evaluation/                   # 复用现有通信与结果记录

configs/
├── baselines/evo1_groot/          # 新模型无记忆实验
└── ttt/                          # 结构、更新规则及消融组合

tests/
├── unit/ttt/                     # 更新、状态、梯度、mask
├── unit/evo1_groot/              # 模型身份、结构与基线行为
├── unit/robottt/                 # RoboTTT 连接、去噪协议与状态生命周期
├── unit/mac/                     # 对应模型的结构与状态语义
├── unit/mag/
├── unit/mal/
└── integration/                  # 训练与流式推理一致性、闭环 smoke
```

### 依赖与状态约定

- `ttt/` 提供不依赖具体 VLA、LeRobot dataset 或仿真器的公共记忆计算：函数式前向、fast model、状态操作、KV 学习与可组合更新器。不让所有机制继承一个同时包办 attention、采样和训练的巨大 `TTTLayer`。
- `models/robottt/`、`models/mac/`、`models/mag/`、`models/mal/` 与 `models/evo1_groot/` 并列，各自拥有架构演进空间。模型层负责记忆与 attention 的连接、读写时机、读写输入、输出融合，以及跨帧和去噪状态生命周期；不强制所有变体使用同一个动作头。
- 优先组合和复用组件，避免通过继承完整 policy/model 将新架构绑在 `evo1_groot` 的内部流程上。各模型可显式复用其 token 编码器、DiT 组件等，也可独立实现；出现多个模型稳定共享的需求后，再提取公共组件，不提前建立大一统模型基类。`evo1_groot` 不反向依赖这些记忆模型。
- 更新器只决定如何更新权重与自身辅助状态；何时读写、输入什么信息由模型架构决定。四种架构均通过公共更新接口组合普通梯度或动量/遗忘规则，避免在各模型目录复制更新逻辑。
- 公共 `training/` 与 `evaluation/` 负责连续序列采样、TBPTT 调度、episode 边界通知等流程协调；各模型解释这些事件并管理自己的状态。架构特有的训练或推理逻辑可放在对应模型目录，再通过公共接口接入。
- `W0` 是学习到的模型参数，`Wt` 是 episode 运行状态。动量等辅助量属于对应更新器的运行状态。
- 普通模型 checkpoint 保存 slow weights、`W0` 和配置，不隐式携带上一 episode 的记忆。序列中途精确恢复训练属于另外的状态保存需求。
- `reset` 恢复初始状态，`detach` 保留数值但切断梯度；两者不可混用。不同样本、episode 和层的状态默认隔离。
- `evo1_groot` 保持独立的无记忆 baseline，不为新增记忆模型改写其既有训练与推理行为。记忆模型应提供明确的关闭记忆对照路径；近零门控不等于严格关闭。如果某变体同时改造了其他组件，需提供对应的无记忆消融，不能将相对 `evo1_groot` 的全部收益归因于记忆。
- 优先复用现有 dataset、processor、checkpoint 和评测基础设施；序列训练如需扩展 trainer，显式记录扩展边界，不假定原默认循环天然满足 TBPTT。

## 后续实施顺序与验收关注点

1. 将待决策事项收敛为新基座与首个 RoboTTT 组合的实施规格；保持原生/本地 Evo-1 对照线独立运行。
2. 实现独立的 `evo1_groot` 无记忆 baseline，验证模型注册、初始化、训练、checkpoint 和闭环执行。
3. 实现可读的 fast-model/KV 更新与状态操作，检查内外循环梯度、批内隔离、reset/detach 和 mask。
4. 在 `models/robottt/` 实现 RoboTTT-style，组合公共 TTT 组件并按需复用 `evo1_groot` 组件，固定 token 布局、时间尺度及去噪状态调度；验证关闭记忆后的基线行为。
5. 完成 sequence action forcing 和真正释放片段计算图的 TBPTT，再进行小规模真实训练与机制对照。
6. 在公共 `ttt/updates/` 增加动量/遗忘，在并列的 `models/mac/`、`models/mag/`、`models/mal/` 中实现对应 VLA 变体；分别验证更新器与结构，再扩展组合矩阵。

机制验收需包括未来输入扰动不影响过去输出、分段/流式前向一致性、状态重置、fast weights 确实更新、部署 slow weights 不变，以及参考后端与优化后端的输出/状态/梯度对照。性能测量需注明更新频率、预测与执行 horizon、去噪步数、TBPTT 长度和精度。

此记录没有宣称 RoboTTT/Titans 已实现、社区仓库完整测试已通过或新基座已有任务成绩。正式开发建议使用独立 worktree 和输出目录；共享数据资产可以复用，GPU 使用需与正在进行的 baseline 实验协调。
