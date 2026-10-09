# VLA-TTT 开放研究平台总路线图

**记录日期：** 2026-10-01

**2026-10-08 稳定基线落地：** 本地 master 纳入已验收的 Evo-1 / evo1_groot 基线与隔离评测管线。foundation 工作树的 RoboTTT/Titans 开发不随本次合并进入 master。详见[验收记录](../../experiments/2026-10-08-baseline-acceptance.md)。

**目标：** 以本地 Evo-1 为研究基座，复现 RoboTTT 与 Titans 的测试时学习机制，建立可复现的记忆任务实验、消融和性能分析，支持后续论文研究。

**文档层级：** 全局路线图。各阶段开始前再制定具体实现计划；本文件不预先锁定尚未验证的算法接口或实验超参数。

**依据：** 用户已确认的架构、研究顺序和 benchmark 分工。

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
| 原生/本地数值等价 | 已验证 | 已缓存真实 checkpoint 上的合成 batch：动作、训练 loss、action-head 梯度、checkpoint round-trip/reset |
| 本地 CLI 兼容入口 | 已实现 | `vla-ttt-train/eval/rollout` 复用 LeRobot runner，补充本地 processor 恢复 |
| LIBERO 实际成功率 | 两种本地基线已完成 | 各40任务×20初态；Evo-1 520/800，evo1_groot 539/800；单训练seed，非记忆能力验证 |
| 真实数据训练 baseline | LIBERO 正式训练已完成 | 两种基线均5000+80000步；RoboMME仍只有短程验证，正式结果待补 |
| RoboTTT / Titans | foundation已完成首版代码与CPU机制验证 | 本稳定分支尚不包含；正式任务验证与代码合并分别推进 |
| server-client 评测 | LIBERO完整实现对照已完成并接受 | 本地520/800、原生515/800；795/800成功标志一致，788/800成功标志及步数一致；RoboMME仍为smoke验证 |

历史插件计划中的“不复制 Evo-1”约束已由本路线图与固定快照决策取代。
已完成的基线工作已落到本地 master；记忆机制代码与CPU验证已完成，仍保留在foundation工作树，尚未合并或完成正式任务验证。

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
- [x] 用一致的任务配置、80k权重、种子与评测设置完成原生/本地 LIBERO 完整对照，差异经用户接受。
- [x] 建立本地 Evo-1 和 evo1_groot 的 LIBERO 两阶段正式训练及40任务×20初态回归结果。
- [ ] 建立两种本地基线的 RoboMME 正式训练与评测结果，固定任务划分与训练预算。

**验收：** baseline 可以从配置与固定资产复现，原生/本地差异有解释，得到真实任务结果。
现有 LIBERO checkpoint 仅是起点，不假定其可直接解决 RoboMME/RMBench。
阶段A的LIBERO部分已完成并验收，两种本地基线的RoboMME正式结果仍待完成。不重复原生完整长训练；已完成的原生评测属于同权重实现对照，不是独立训练的官方baseline。
实施与证据见 [阶段A计划](2026-10-01-stage-a-baselines.md) 和 [操作指南](../../baselines.md)。


**2026-10-08 LIBERO 对照验收：** 相同权重、40任务×20初态，原生515/800（64.375%）、本地520/800（65.0%），差0.625个百分点；795/800成功标志一致，788/800成功标志与终止步数均一致，基础设施错误0。用户确认差异在接受范围内，原生/本地评测对照待办关闭，不再把逐episode完全一致作为此项验收前提。差异具体来源尚未定位，不宣称严格确定性等价或已证明由随机性导致。证据：[原生汇总](../../../outputs/eval/libero-native-stage2-080000-20ep-20261008/summary.json)、[逐episode对照](../../../outputs/eval/libero-native-stage2-080000-20ep-20261008/native-local-comparison.json)。

### 阶段 B：本地 Evo-1 + RoboTTT

- [ ] 对照原论文及可用官方实现，形成机制清单与 Evo-1 对应关系。
- [ ] 明确 TTT 插入层、fast-weight 初始化、更新目标、读写顺序与门控。
- [ ] 实现按轨迹顺序处理的训练数据路径、sequence action forcing 和 TBPTT。
- [ ] 明确 episode reset、截断处 detach、action chunk 与环境时间步之间的更新频率。
- [ ] 在小规模任务上验证训练可收敛、历史能影响决策，且不存在未来信息泄漏。
- [ ] 完成 RoboMME 上的本地 baseline 与 RoboTTT 闭环对比。

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

**当前判定：代码实现与CPU验证已完成，正式训练与任务评测尚未完成。** 工作位于foundation，未随稳定基线落入master。实现记录为154 passed、7 skipped；本会话独立复核同样为154 passed、7 skipped，跳过资产相关集成测试，不代表真实预训练模型GPU验证。见[实现说明](../../../.worktrees/evo1-plugin-foundation/docs/titans.md)和[验证记录](../../../.worktrees/evo1-plugin-foundation/docs/experiments/2026-10-08-titans-vlm-implementation.md)。

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

### 2026-10-08 RoboMME Evo1-GROOT正式准备

- [x] master独立RoboMME环境离线建立；16任务各test0真实reset/render/step通过。
- [x] 核验48组task/split元数据及官方seed；固定MME-VLA名义1300步协议及wrapper计数偏移。
- [x] 全量1600episode/768897帧数值和边界检查；476857执行帧作为动作监督，demo原文件保留。
- [x] 复用原LeRobot视频，执行段索引视图、20步action查询和padding mask、执行段统计量接线完成。
- [x] 本地InternVL3+随机GROOT头；Stage1 batch32、Stage2 batch16各3步通过，显存18.46/26.47 GiB。
- [x] 独立server-client四套件代表任务smoke通过，无基础设施错误。
- [ ] 运行已同意的正式5000+80000步训练，再进行val选模和16×50 test评测；本轮没有启动。

可执行脚本`bash scripts/train_robomme_evo1_groot.sh --train`；默认不传参数只跑smoke。
细节与证据见[训练前检查报告](../../experiments/2026-10-08-robomme-baseline-preflight.md)。
本地Evo-1的RoboMME正式结果仍未完成；不将本次Evo1-GROOT准备标作整个阶段A完成。
