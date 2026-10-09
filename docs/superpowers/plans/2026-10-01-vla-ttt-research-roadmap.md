# VLA-TTT 开放研究平台总路线图

**记录日期：** 2026-10-01

**进度更新：** 2026-10-08。LIBERO 两条本地基线和同权重原生/本地完整对照已验收；Titans-style VLA+TTT 代码实现与 CPU 验证已完成，正式任务验证未完成。

**2026-10-09 开发分支补充：** `.worktrees/observation-memory` 中新增 AQMem 与 T²Mem-style，保持 Evo1-GR00T 主干，使用多层 VL interface 特征、先读后写和半执行周期观测更新；支持三种更新规则及 MAC/MAG/MAL 的可选同协议对照。窗口训练与 CPU 管线已实现，真实任务实验尚未开展，未合入 master。以[新设计](../specs/2026-10-09-observation-memory.md)、[使用说明](../../observation-memory.md)和[验证记录](../../experiments/2026-10-09-observation-memory-implementation.md)为准；原 RoboTTT 保留为独立对照。

**目标：** 以本地 Evo-1 为研究基座，复现 RoboTTT 与 Titans 的测试时学习机制，建立可复现的记忆任务实验、消融和性能分析，支持后续论文研究。

**文档层级：** 全局路线图。各阶段开始前再制定具体实现计划；本文件不预先锁定尚未验证的算法接口或实验超参数。

**依据：** 用户已确认的架构、研究顺序和 benchmark 分工。

**2026-10-01 设计补充：** 原生与本地 Evo-1 的阶段 A 对照继续用于验证代码和管线。另行实现独立的 `evo1_groot` 无记忆 baseline，复用 Evo-1 组件搭建轻量 GR00T-style VLA，作为后续记忆研究的新基座，不替换既有 Evo-1 baseline。记忆研究按 RoboTTT-style / MAC / MAG / MAL 四种接入结构与普通梯度 / Titans-style 动量和遗忘两种更新规则分别配置。先推进新基座与首个 RoboTTT 组合，其他组合逐步扩展。下文阶段 B–D 的具体实施计划需据此细化；这项设计补充不代表实现或实验已完成。参考代码、待决策项和建议布局见[TTT 研究设计记录](../specs/2026-10-01-ttt-research-architecture-notes.md)。

## 一、已确定的总体安排

1. 建立原生 LeRobot Evo-1 baseline。
2. 建立本地 Evo-1 baseline，并验证与原生实现等价。
3. 实现本地 Evo-1 + RoboTTT。
4. 比较成功率、长序列退化、适应收益、显存和延迟。
5. 使用 RoboMME、LIBERO、RMBench 进行评测。
6. 实现本地 Evo-1 + Titans，纳入同一评测与消融体系。

以上是工作范围；训练、机制验证和评测将在各阶段交替进行。
研究验证仍优先推进 evo1_groot + RoboTTT；Titans-style MAC/MAG/MAL 已并行完成代码实现，正式实验另行推进，不预先构建面向多种 VLA 的复杂抽象。

## 二、平台边界与长期约束

- 本地 Evo-1 来源为 `lerobot==0.6.1` 的固定快照，全部 Evo-1 专属源码进入正式包目录。
- 保留原始文件名、模型层级、参数 key、配置字段与版权头；来源见 [快照记录](../../evo1-snapshot.md)。
- 一次性记录来源版本、文件映射和哈希；不主动同步、不持续追踪上游 Evo-1 差异。
- LeRobot 通用基础设施继续负责 dataset、processor pipeline、训练与评测框架等。
- 原生 `policy.type=evo1` 与本地 `policy.type=vla_ttt` 并存；本地 baseline 必须能够独立执行本地模型源码。
- TTT 机制在本地源码中显式实现，研究者能直接查看插入位置、状态更新和梯度路径。
- 数据、权重、仿真资产不提交到 Git。下载前告知用户来源、固定 revision、缓存位置与预计体积。
- 主要硬件为单张 RTX PRO 5000 72GB。训练上下文、截断长度、精度和 batch size 根据实际显存测量确定。
- server-client 评测已按本轮要求接入，参考 StarVLA 的 WebSocket + MessagePack；policy、LIBERO、RoboMME 使用三个环境。不是之前讨论的 gRPC 草案。
- 复用框架不等于直接使用默认训练/评测语义：序列采样、TBPTT、memory reset 和推理梯度上下文必须按 TTT 需要适配。

## 三、当前状态

| 工作 | 状态 | 证据或边界 |
| --- | --- | --- |
| LeRobot policy plugin、严格 checkpoint 转换 | 已实现 | 已有配置、processor、discovery 和 checkpoint 测试 |
| 本地 Evo-1 固定快照 | 已实现 | 提交 `6294fea`，位于 `feat/evo1-plugin-foundation` worktree |
| 原生/本地数值等价 | 已完成短程验证 | 真实 LIBERO batch 的动作、loss、全梯度一致；Stage 1 100步 + Stage 2 20步的逐步 loss、梯度范数、最终权重哈希一致，不代表独立完整长训练已完成 |
| 本地 CLI 兼容入口 | 已实现 | `vla-ttt-train/eval/rollout` 复用 LeRobot runner，补充本地 processor 恢复 |
| LIBERO 本地 Evo-1 正式训练 | 已完成 | 从通用预训练 VLM + 随机动作头启动，Stage 1 5000步/batch 8，Stage 2 80000步/batch 4；训练日志及 checkpoint 已审计 |
| LIBERO 本地 Evo-1 回归 | 已完成单训练 seed 评测 | 固定80k checkpoint，4套件×10任务×20初态：520/800（65.0%），基础设施错误0；已有逐任务成功率与区间 |
| 原生 Evo-1 完整长训练 | 暂缓重复执行 | 同一80k权重的原生/本地完整评测已完成；这是实现对照，不是独立训练的官方 baseline |
| RoboMME / RMBench 正式结果 | 未完成 | RoboMME 已有真实数据短程训练及 RouteStick smoke；RMBench 尚待后续接入，不能以 LIBERO 结果代替记忆评测 |
| 独立 `evo1_groot` baseline | 正式训练与LIBERO评测已验收 | 5000+80000步；539/800（67.375%），错误0；基线已落到本地master |
| RoboTTT / Titans | 首版代码与CPU机制验证已完成 | 位于foundation工作树，未随基线进入master；正式训练、长程稳定性与记忆任务成绩待验证 |
| server-client 评测 | LIBERO完整实现对照已完成并接受 | 本地520/800、原生515/800；795/800成功标志一致，788/800成功标志及步数一致；RoboMME仍为smoke验证 |

历史插件计划中的“不复制 Evo-1”约束已由本路线图与固定快照决策取代。
已验收的 Evo-1/evo1_groot 基线已落到本地 master（4877198）；foundation 的记忆机制开发未随之合并。

## 四、评测环境分工

| 环境 | 定位 | 要回答的问题 |
| --- | --- | --- |
| RoboMME | 第一主 benchmark | TTT 对不同类型记忆任务是否有效，收益来自什么机制？ |
| LIBERO | 基线与控制能力回归 | 原生/本地实现是否一致，加入 TTT 后基础控制能力是否退化？ |
| RMBench | 第二主 benchmark | 记忆机制能否推广到另一套任务和机器人操作环境？ |

优先在 RoboMME 少量代表性任务上完成端到端闭环，再扩展正式评测覆盖。
LIBERO 的长任务成功率不单独作为长期记忆能力的证据。
RMBench 接入在首个 RoboMME + RoboTTT 闭环后推进，正式机制比较尽可能覆盖两套记忆基准。
最终任务集合、训练/验证/测试划分及官方评测协议在对应子计划中固定。

## 五、阶段与验收条件

### 阶段 A：原生与本地 Evo-1 baseline

- [x] 保留可运行的原生 Evo-1 路径。
- [x] 完成本地固定快照和 checkpoint/state-dict 兼容。
- [x] 验证动作输出、训练 loss、action-head 梯度及 processor 行为等价。
- [x] 建立 RoboMME 首个任务的数据、动作语义、训练和闭环评测路径（RouteStick，真实数据短程训练与test episode smoke）。
- [x] 接入隔离环境 server-client，并验证同checkpoint的 LIBERO native/local 单episode一致性。
- [x] 完成原生/本地 Evo-1 的 LIBERO 两阶段短训练数值对照（100 + 20步）。
- [x] 完成本地 Evo-1 的 LIBERO 正式两阶段训练（5000 + 80000步），审计日志和 checkpoint。
- [x] 建立本地 Evo-1 的 LIBERO 回归结果（40任务、每任务20初态、520/800成功），记录逐任务结果和评测协议。
- [x] 用一致的任务配置、80k权重、种子与评测设置完成原生和本地 baseline 的完整 LIBERO 对照，差异经用户接受。
- [ ] 完成本地 Evo-1 与 evo1_groot 的 RoboMME 正式训练与足量闭环评测，固定任务/数据划分与训练预算。
- [x] 接入独立 `evo1_groot` 无记忆 baseline，验证训练/推理接口并对齐 LIBERO 训练设置。
- [x] 验收 `evo1_groot` 正式两阶段训练及40任务×20初态评测：539/800，错误0。

**原生对照决策：** 不重复原生完整长训练。已用本地80k权重的独立原生兼容配置完成800次评测，权重和归一化张量哈希相同，未覆盖原检查点。不将同权重实现对照表述为独立训练的官方baseline。

**验收：** baseline 可以从配置与固定资产复现，原生/本地差异有解释，得到真实任务结果。
现有 LIBERO checkpoint 仅是起点，不假定其可直接解决 RoboMME/RMBench。
**当前判定：阶段 A 的 LIBERO 部分完成并验收，RoboMME 正式结果仍待完成。** LIBERO结果来自单训练seed，不代表多seed结论或记忆能力验证。
实施与证据见 [阶段A计划](2026-10-01-stage-a-baselines.md) 和 [操作指南](../../baselines.md)。

更新证据：

- [原生/本地两阶段短训练与数值等价](../../experiments/2026-10-01-libero-scratch-preflight.md)。
- [本地 Evo-1 正式训练审计与初评](../../experiments/2026-10-05-libero-training-and-preliminary-eval.md)。
- [LIBERO 每任务20次评测与逐任务结果](../../experiments/2026-10-05-libero-20-episodes-per-task.md)。
- [`evo1_groot` 训练管线对齐与离线正式命令](../../experiments/2026-10-05-evo1-groot-libero-alignment.md)。


**2026-10-08 LIBERO 对照验收：** 相同权重、40任务×20初态，原生515/800（64.375%）、本地520/800（65.0%），差0.625个百分点；795/800成功标志一致，788/800成功标志与终止步数均一致，基础设施错误0。用户确认差异在接受范围内，原生/本地评测对照待办关闭，不再把逐episode完全一致作为此项验收前提。差异具体来源尚未定位，不宣称严格确定性等价或已证明由随机性导致。证据：[原生汇总](../../../../../outputs/eval/libero-native-stage2-080000-20ep-20261008/summary.json)、[逐episode对照](../../../../../outputs/eval/libero-native-stage2-080000-20ep-20261008/native-local-comparison.json)。

### 阶段 B：本地 Evo-1 + RoboTTT

当前具体研究基座为独立 `evo1_groot`；原生/本地 Evo-1 对照线继续保留。

- [x] 对照原论文与固定版本社区参考实现，形成机制清单与 Evo1-GR00T 对应关系（不声称取得官方实现）。
- [x] 明确 TTT 插入层、fast-weight 初始化、更新目标、读写顺序与门控。
- [x] 实现按轨迹顺序处理的训练数据路径、sequence action forcing 和 TBPTT。
- [x] 明确 episode reset、截断处 detach、action chunk 与环境时间步之间的更新频率。
- [x] 分离决策计数与fast-weight更新计数；RoPE默认按decision推进，保留update模式，逐去噪累计更新不变。
- [x] 实现可选TBPTT按段输入搬运；默认整窗口驻留GPU，使用 `--segment-transfer` 开启分段搬运，并记录运行配置。
- [ ] 在GPU空闲后验证真实VLM序列训练的峰值显存、吞吐与梯度；CPU机制测试不能替代长上下文实测。
- [ ] 在小规模任务上验证训练可收敛、历史能影响决策，并完成因果性与重叠标签信息路径诊断；当前完整 chunk 配方不宣称已排除重叠监督路径。
- [ ] 完成 RoboMME 上的本地 baseline 与 RoboTTT 闭环对比。

**2026-10-05 实施：** `models/robottt/`、公共 `ttt/`、序列数据/TBPTT 与评测入口已落地。CPU 机制、真实小型 LeRobot 数据集训练/保存/加载已验证；正式任务收敛、RoboMME 闭环及 demo 上下文协议仍待验证。见[实现说明](../../robottt.md)与[CPU 验证记录](../../experiments/2026-10-05-robottt-implementation-smoke.md)。

**审查后约定：** 两种输入驻留模式都保留分段backward和状态detach；关闭分段搬运不等于关闭TBPTT。用户反馈无记忆 `evo1_groot` 显存约18 GiB，不能据此推断RoboTTT长序列的显存上限。默认小窗口先整段驻留，扩展上下文时测量后选择搬运模式。RoPE时钟调整不消除训练1次/推理多次写入的差异；DAgger与重叠chunk改造暂缓，保留风险记录，不声称无泄漏。实现细节与显存估算见[修订设计](../specs/2026-10-05-evo1-groot-robottt-design.md)。阶段B仍未完成真实任务验收。

**验收：** 机制实现、状态生命周期和梯度路径有测试；关闭 TTT 的基线行为可验证；有可复现的任务对照结果。
在 Evo-1 上迁移 RoboTTT 机制与复现原论文 GR00T/机器人实验结果应分别表述。

### 阶段 C：RoboTTT 系统评测与机制分析

- [ ] 扩展 RoboMME 任务覆盖并完成 LIBERO 回归。
- [ ] 接入 RMBench，记录机器人状态/动作、相机、控制频率和训练数据的适配。
- [ ] 比较单帧 baseline、短历史 baseline 与 TTT；必要时加入同 backbone 的非 TTT 记忆对照。
- [ ] 完成上下文长度、memory reset、历史截断/置换等受控消融。
- [ ] 汇总成功率、长序列退化、适应收益、显存和延迟。

**验收：** 结论具有重复实验支持，并能区分记忆收益、额外参数、训练预算和低层控制差异。
他人不同 backbone 的公开分数可作为背景参考，不能替代同条件机制比较。

### 阶段 D：evo1_groot + Titans-style VLA+TTT

- [x] 实现MAC、MAG、MAL三种VLM内记忆连接，复用本地evo1_groot的VLM组件与GrootActionHead。
- [x] 提炼共享memory读写与状态接口，实现独立可选的普通GD和数据依赖动量/遗忘更新器。
- [x] 完成CPU机制、掩码、外层梯度、checkpoint重算、reset/detach、严格重载、动作队列及小型数据训练闭环验证。
- [ ] 验证真实预训练全尺寸VLM的GPU梯度、初始化扰动、显存/吞吐及短程训练收敛。
- [ ] 接通RoboMME记忆任务的演示/上下文输入：允许写入memory，但不施加执行动作监督；现有丢弃demo的无记忆基线准备流程不能直接替代。
- [ ] 在相同evo1_groot基座、数据划分和记录的计算预算下正式比较baseline、RoboTTT和MAC/MAG/MAL。
- [ ] 分析不同记忆任务、上下文长度和干扰条件，验证窗口训练与整episode推理的长程稳定性。

**当前判定：代码实现与CPU验证已完成，正式训练与任务评测尚未完成。** 工作位于foundation，未随稳定基线落入master。实现记录为154 passed、7 skipped；本会话独立复核同样为154 passed、7 skipped，跳过资产相关集成测试，不代表真实预训练模型GPU验证。见[实现说明](../../titans.md)和[验证记录](../../experiments/2026-10-08-titans-vlm-implementation.md)。

**命名与适配约定：** 保留MAC/MAG/MAL名称，表述为Titans-style VLA适配，不宣称完整复现论文。MAL的持久token绕过神经记忆、直接进入attention K/V前缀，可作为后续对比设计；当前没有声称已完成另一条路径的消融。三者默认观测内因果attention、观测级更新及普通GD，启用momentum_decay才使用动量/遗忘规则。memory关闭时的基线一致性不代表开启后的初始化行为无扰动。

**验收：** 可追溯实现和CPU测试已经具备；阶段D整体验收仍需真实模型训练与记忆任务实验，不以代码完成替代能力验证。

### 阶段 E：开放研究交付与改进

- [ ] 发布可复现的训练/评测配置、版本记录、模型说明和分析脚本。
- [ ] 提供 baseline → RoboTTT → Titans 的源码阅读与实验指南。
- [ ] 根据实验中的明确瓶颈开展改进、组合机制或新的消融。

**验收：** 其他研究者能运行基线、理解 TTT 插入位置、替换机制并复现实验。
每阶段及时补齐文档与复现材料；不等到最后才记录实验。

## 六、统一实验记录与指标

- **成功率：** 遵循 benchmark 官方成功判定，记录任务级结果、样本数、种子与不确定性。
- **长序列退化：** 区分轨迹时长、记忆延迟、上下文长度和干扰量；比较受控曲线，避免将任务难度变化误作遗忘。
- **适应收益：** 区分使用历史 memory 的收益与从失败/纠正中改进的收益；后者需要相应任务和数据，不能从成功率提升直接推断。
- **显存：** 分别记录训练与推理的峰值、常驻 memory state，并注明精度、batch、上下文和 TBPTT 长度。
- **延迟：** 分开记录预处理、memory update、动作生成与端到端时间；报告 action chunk 长度、执行频率、预热和 CUDA 同步测量方法。
- **公平性：** 记录 backbone/checkpoint、可训练参数、数据量、训练步数与算力预算；无法匹配的条件在结果中明确列出。
- **状态隔离：** 默认每 episode 恢复初始 memory；跨 episode 保留属于单独实验，不与标准评测混报。
- **可追溯性：** 每次正式实验保存代码 commit、依赖与资产 revision、配置、日志、checkpoint 和逐 episode 结果。

## 七、后续推进方式

LIBERO本地Evo-1、evo1_groot及同权重原生对照已完成，差异经用户接受，不再作为阶段A阻塞。下一步固定RoboMME的任务清单、数据划分、训练预算及checkpoint选择规则，准备两种无记忆baseline的离线正式训练与评测；本次更新不启动训练。

RoboMME原始数据已位于 `/home/hello/lizk/datasets/robomme`，元数据及行数核验为1600 episodes、768897帧，其中292040帧为demo，引用视频文件均存在；不将文件存在视为完整解码验证。无记忆baseline的数据准备与TTT的demo上下文协议分别设计，避免删除记忆任务所必需的信息。扩展下载仍须提前告知来源、revision和体积。

RoboTTT与Titans-style代码已在foundation实现，后续重点是真实模型、长程训练与任务能力验证；独立验收后再合并到master，不影响稳定基线实验。RMBench仍按阶段C推进。

各子阶段沿用Native方式逐项实施、测试和审查；研究范围或目标变化先更新路线图，一般实现细节由子计划记录。
