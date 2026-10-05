# Evo1-GR00T + RoboTTT 实施设计草案

状态：用户已确认首版按论文描述使用完整 register/state/action 的 attention 隐藏特征，并沿用社区方案在每次去噪时累计更新记忆；相关风险独立记录，不先引入动作前缀或安全写入改造。首版模型、公共记忆、序列训练和评测接口已实现，CPU 验证见[实现记录](../../experiments/2026-10-05-robottt-implementation-smoke.md)；正式任务结果待验证。依据 [架构约定](2026-10-01-ttt-research-architecture-notes.md) 和 [路线图阶段 B](../plans/2026-10-01-vla-ttt-research-roadmap.md)。更新时机与信息路径详见[专项记录](2026-10-05-robottt-update-timing-and-hidden-features.md)。

## 目标与边界

在同一 foundation 项目内新增 `models/robottt/`，组合公共 `ttt/` 实现 Evo1-GR00T + RoboTTT，提供初始化、连续轨迹训练、Sequence Action Forcing、TBPTT 与有状态评测接口。保持 `evo1_groot` 已启动的 LIBERO 训练、配置与 checkpoint 不变。

本轮优先普通梯度更新，不同时实现 Titans、MAC/MAG/MAL。公共更新器接口为这些后续扩展保留职责边界，不提前添加空实现。原生/本地 Evo-1 仍作为独立参照。

代码正确性、真实任务训练收敛、RoboMME 闭环比较分别验收；机制测试通过不能替代阶段 B 的任务结果。当前 GPU 正用于 baseline 训练，先做 CPU 机制与小模型管线验证，不抢占正在进行的正式实验。

## 论文、社区代码与基座的对应

本次重新读取此前保存的 RoboTTT 论文第 2、3.1、3.2 节及附录 A.1，以及社区固定版本 `98eb8837f66ec28a0d131fc8bfabb8d6c7bb017e` 的 `MemoryKeyValueBind`、`TTTWrapper`、`_forward_with_ttt`、`_sample`。来源链接与版本见架构记录。

| 机制 | 已明确的依据 | 实施方向 |
| --- | --- | --- |
| 插入位置 | 论文在 self/cross attention 后加入 TTT | 新动作头在每个 attention 子层后、FFN 前接入；不将整段 DiT 输出简单再加一次残差 |
| 时间聚合 | attention 处理单个轨迹位置，TTT 沿轨迹递推 | 显式区分 batch、轨迹位置、chunk 内 token；不把去噪步当成环境时间 |
| 输入 | register/state/noisy action tokens；VL tokens 不直接输入 TTT | 复用基座编码组件；默认新增 16 个 register tokens；不限制 fast model 的输入仅为 registers |
| fast model | 两层 GeLU MLP；W0 与 Q/K/V 投影由外层任务学习 | 公共函数式 MLP 与显式逐样本、逐层状态；宽度可配置，轻量尺寸不宣称等于论文 |
| 更新/读取 | MSE KV 目标、普通 GD、update then apply | 在 Wt-1 上求梯度，得到 Wt 后读取 Q；验证外层梯度经过内循环回到 W0 |
| 学习率 | 论文提到 0.1 基础学习率与可学习项 | 明确正值参数化与损失归约；不直接沿用社区 `softplus(0.01)` 作为论文数值 |
| 位置与门控 | RoPE theta=10000；逐维 tanh 门控，alpha 初值 0.001 | 时间索引属于轨迹状态；padding/reset 不误推进；避免社区 hook 的重复残差 |
| Sequence Action Forcing | 每个 batch×轨迹位置独立采样 flow 时间 | 保留 Gaussian 噪声和 `0.999*(1-Beta(1.5,1))`；与时间序列 mask 对齐 |
| TBPTT | 数值跨段，梯度在段边界截断 | 每段计算与 backward 后释放图；按全序列有效监督量归一化；首段 W0 接收梯度 |

新增 registers 本身就会改变 self-attention。因此“严格关闭记忆”回归需要同时旁路新增 registers，才能与来源 baseline 对齐；保留 registers、仅关闭记忆的实验应另记为 register-only 消融。近零门控不能替代严格关闭。

## 时间协议：首版选择与研究限制

### 去噪阶段的状态提交

论文描述每个观测更新并传递 fast weights，同时每个动作 chunk 进行多步去噪，但没有明确这些去噪调用怎样共享和提交状态。

备选方案（首版不采用）：一次动作决策的每个去噪步骤从同一历史快照计算临时更新，决策边界提交一次。优点是环境时间与去噪次数分离；仍需明确提交哪次候选或是否独立执行一次写入，以及训练历史状态如何与之对应。这属于工程协议，不能称为已核实的官方实现。

**用户确认的首版选择：沿用社区实现，每次去噪都累计更新记忆。** 当前去噪步骤使用上一去噪步骤返回的 fast weights，最后的状态继续传给后续动作决策，不在各去噪步骤间恢复同一历史快照。调整去噪步数同时改变记忆更新次数，因此分别记录环境步、决策步和内循环更新次数。论文所述 Sequence Action Forcing 和所读社区训练入口在每个轨迹位置采样一个噪声等级，不自动执行与部署相同次数的去噪；这项训练/部署差异需记录和验证。

该选择确定更新的累积方式，不决定写入信息的可用性；不能据此认定重叠 chunk 已排除未来标签泄漏。

### 重叠动作 chunk 的监督泄漏

现有 LIBERO 基座预测 50 步、执行 10 步。训练若在环境时刻 0 用真实动作 0..49 的加噪版本更新记忆，并在时刻 10 直接继承该状态，历史可能已经包含时刻 10..49 的动作监督。仅对动作 loss 加 mask，或只挑 register tokens 写入，都不能排除这种路径：self-attention 已经混合动作标签信息。

需要区分训练与推理：推理从噪声及当时可用观测生成的 50 步预测动作，是模型自己的计划；即使后 40 步没有执行，保留其信息也不等于获得未来真实动作标签。风险来自训练把真实重叠 chunk 的标签信息注入可供后续预测读取的历史状态。加噪、增加更新次数或 TBPTT detach 均不保证消除这些信息。这是针对特定重叠采样/状态传递组合的因果性分析，不是断言论文或社区所有使用方式都存在泄漏。

**用户确认首版先按论文的完整 chunk 特征路径实现，不因上述疑虑改为只写入已执行动作。** 逐去噪累计更新继续采用已选社区协议；不能声称该选择已经排除重叠标签的信息路径。以下作为后续消融方向，不是首版实施前置条件：

- 非重叠序列：训练 stride 至少覆盖预测 horizon，同时评测使用匹配的决策/执行频率，并重新建立相同频率的无记忆对照。简单，但不能直接与当前执行 10 步的 baseline 做唯一变量比较。
- 保留 50/10：把当前动作去噪与提交到历史的输入分开；历史提交不能接触下一决策时刻之后的真实动作标签。可能需要额外的安全写入前向，例如仅使用已执行前缀及非标签填充，或使用模型预测动作。需逐项说明与论文配方的差异、计算开销和训练/部署对应关系。

不以“时间顺序采样”代替无泄漏保证。对历史 chunk 中未执行真实标签的扰动应作为信息路径诊断，记录状态/输出是否变化，而不是要求首版完整特征路径必然通过无变化测试。更晚序列位置不得影响更早输出的基本因果性测试仍需通过。两者区别及后续消融见[专项记录](2026-10-05-robottt-update-timing-and-hidden-features.md)。

## 公共职责与验证清单

- `ttt/functional.py`、`fast_models.py`、`state.py`、`memory.py`、`updates/gradient_descent.py`：函数式前向、KV 学习、纯更新与状态操作；状态和 slow parameters 分离。
- `models/robottt/`：独立配置/策略身份、DiT 连接、registers、去噪协议、动作队列和记忆状态生命周期；显式初始化来源与参数映射报告。
- `training/sequence.py`、`flow_matching.py`、`tbptt.py`：连续 episode/subtrajectory、独立 flow 时间、有效/写入/监督/reset mask、分段反向与计量。
- `evaluation/`：针对新策略提供允许 fast-weight 求梯度的入口，保留普通 baseline 的 `inference_mode` 行为；每个 episode reset，slow weights 不被更新。
- 测试：手算 GD、外层二阶梯度、批内/层间隔离、reset/detach 区别、padding 不写入、因果扰动、流式/分段一致性、严格关闭对齐、checkpoint 不保存 episode 状态、实际训练器梯度上下文。

现有 RoboMME baseline 的 canonical 数据准备会过滤 demo 帧。新序列路径不能直接把该过滤结果当作已经支持视频上下文的完整配方；需要保留独立的 context-only 输入与监督 mask，并核实 benchmark demo 输入协议。不得把 privileged labels 引入模型输入。

## 接下来

按已确认的完整隐藏特征与逐去噪累计协议，固化训练/推理状态转移、配置默认值和实施计划；潜在问题保留在专项记录中持续验证。当前草案不改变正在运行的 baseline，不代表阶段 B 已完成。
