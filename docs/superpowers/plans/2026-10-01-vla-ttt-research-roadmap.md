# VLA-TTT 开放研究平台总路线图

**记录日期：** 2026-10-01  
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
先把 Evo-1 + RoboTTT 做深，再开展 Titans，不预先构建面向多种 VLA 的复杂抽象。

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
| 三套 benchmark 的实际成功率 | 未完成 | 尚不能用数值等价测试代替任务评测 |
| 真实数据训练 baseline | 短程闭环已验证，正式训练未完成 | RoboMME/LIBERO 各自 native/local 真实数据2步训练、保存与数值一致性；非收敛结果 |
| RoboTTT / Titans | 未实现 | 后续阶段工作 |
| server-client 评测 | 已接入并真实运行 | LIBERO同初态native/local均79步成功；RoboMME RouteStick/test0均300步超时；仅单episode smoke |

历史插件计划中的“不复制 Evo-1”约束已由本路线图与固定快照决策取代。
已完成工作予以保留，不重复搭建；分支尚未合并到主分支。

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
- [ ] 用一致的任务配置、checkpoint、种子与评测设置比较原生和本地 baseline。
- [ ] 建立 LIBERO 回归结果；按 benchmark 分别记录模型适配和训练过程。

**验收：** baseline 可以从配置与固定资产复现，原生/本地差异有解释，得到真实任务结果。
现有 LIBERO checkpoint 仅是起点，不假定其可直接解决 RoboMME/RMBench。
当前没有全量训练与正式回归矩阵，不能把两步训练和单episode smoke视为阶段A全部完成。
实施与证据见 [阶段A计划](2026-10-01-stage-a-baselines.md) 和 [操作指南](../../baselines.md)。

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

### 阶段 D：本地 Evo-1 + Titans

- [ ] 选定 Titans 的具体变体，对照论文明确 memory 更新、遗忘、动量及检索机制。
- [ ] 在 RoboTTT 实现经验基础上，仅提炼两者实际需要共享的接口。
- [ ] 完成独立机制测试、训练闭环及 checkpoint/state reset 验证。
- [ ] 在相同 Evo-1 基座、数据划分和明确记录的计算预算下比较 baseline、RoboTTT、Titans。
- [ ] 分析不同记忆任务、上下文长度和干扰条件下的表现。

**验收：** Titans 有可追溯的机制实现与独立实验结果；不以任意 memory 模块替代论文机制后直接冠名 Titans。

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

下一步固定阶段A正式训练/评测任务清单、预算和checkpoint选择规则，完成足量训练与多episode结果。
RoboMME/LIBERO的格式、checkpoint适配和隔离评测已接通；只下载了最小训练分片与LIBERO仿真资产。
扩展数据下载仍需提前告知来源、revision和体积，不提前实施 RoboTTT 或 Titans。

各子阶段沿用 Native 方式逐项实施、测试和审查。新证据若要求改变研究范围或实验目标，
先更新本路线图；一般实现细节由对应子计划记录。
