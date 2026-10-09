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

## 2026-10-05 审查后修订：分段搬运与独立位置时钟

本节更新首版工程协议；保留完整 register/state/action 特征和逐去噪累计 fast-weight
更新，不处理 DAgger 或重叠 chunk 的标签路径，也不修改 `evo1_groot` baseline。

### TBPTT 输入按段搬运

**后续用户决策：按段搬运为可选项，默认关闭。** 默认对完整窗口执行preprocessor，
输入驻留GPU；追加 `--segment-transfer` 才启用按段搬运。该选项不是将模型参数
或fast weights卸载到CPU，也不是关闭训练必需的CPU→GPU数据传输。
`run.json` 和每步metrics记录 `segment_transfer`，便于复现实验。

开启时，`run_training` 将 CPU frames 和
preprocessor 交给 `train_sequence`。后者每段只预处理当前 segment，完成该段
backward、detach 后释放处理后的输入引用，再搬运下一段。有效监督量仍在整个窗口
上统一归一化；计数预扫描只处理动作和 mask，不搬运图像。

用户反馈当前无记忆 `evo1_groot` 占用约18 GiB，先选择默认整窗口驻留；该数字
不是RoboTTT真实长序列显存实测，扩展上下文时仍需测量峰值并酌情启用此选项。

本修改约束的是设备上的输入与计算图；CPU Dataset/DataLoader 仍保留整段窗口，
不宣称主存使用与窗口长度无关。CUDA allocator 的 reserved memory 也不一定立即下降。

按两相机、256×256、RGB、float32、batch=2估算：每个位置的图像为
`2 × 2 × 3 × 256 × 256 × 4 = 3 MiB`。窗口长度T、段长S时，旧方案
图像驻留约 `3T MiB`，新方案约 `3 min(S,T) MiB`，差值约
`3 max(T-S,0) MiB`。仅估算图像张量，不含模型、优化器、激活、临时resize和碎片。

| 窗口T | 整段图像驻留 | 段长S=4时图像驻留 | 相对额外开销 |
| --- | ---: | ---: | ---: |
| 8（当前小窗口默认） | 24 MiB | 12 MiB | 12 MiB |
| 128 | 384 MiB | 12 MiB | 372 MiB |
| 1024 | 3 GiB | 12 MiB | 约2.99 GiB |
| 4096 | 12 GiB | 12 MiB | 约11.99 GiB |
| 8192 | 24 GiB | 12 MiB | 约23.99 GiB |

不同batch、相机数、分辨率和实际输入dtype按比例缩放。VLM使用bf16不意味着
预处理后的图像张量也是bf16。这是内存体积计算，不是GPU实测峰值。

### RoPE位置与更新次数分离

新增配置 `rope_position_mode`：新模型默认 `decision`；`update` 保留原社区式时钟
供消融。两种模式都继续逐去噪累计更新记忆。

- 状态 `decision_index`：完成一次动作决策后加1。
- 状态 `update_count`：每次实际写入fast weights后加1。
- 状态 `position`：下一次调用所用RoPE位置，按选定模式推进。
- `decision`模式：一个决策的32次去噪共用一个位置，最后一次完成后推进位置；
  `update`模式每次写入都推进位置。训练每个轨迹位置只有一次前向，两种模式相同。
- 队列执行不增加上述计数；reset恢复初始状态和零计数；detach保留所有计数值。
  padding/禁止写入的行不推进计数。采样失败不提交部分状态。
- 训练CLI接受 `--rope-position-mode decision|update`；不传时新初始化采用decision，
  从新格式robottt checkpoint加载时继承其配置。日志分别记录更新数、决策数与RoPE位置。
- 模式保存在checkpoint配置中，严格加载不允许静默改变。早期缺失此字段的
  robottt配置会采用新默认decision；要复现实验的旧update语义，应先在独立配置副本中
  显式补上 `rope_position_mode="update"`，不得覆盖原始实验记录。

这只修正位置尺度，不消除训练每位置1次写入、推理每决策32次写入的分布差异；
不是已核实的官方协议，也不预先承诺提升成功率。未来分别用两种模式训练并配对评测。

### 验证边界

CPU回归覆盖实际Q/K投影收到的位置、两种时钟、累计权重变化、队列/reset、mask与
detach计数，以及分段预处理后的图像引用释放和loss/梯度一致性。未抢占正在训练的
GPU，不将CPU生命周期测试表述为真实VLM峰值显存测量。

本次最终验证：关闭CUDA、离线运行完整 `pytest tests -q`，90 passed、7 skipped
（资产opt-in集成测试未开启）；新增功能测试先复现缺失接口再通过。
RoboTTT相关代码和测试的ruff lint/format检查、`git diff --check`通过。
未下载资产、未启动正式训练、未修改已有checkpoint，也未提交或推送。
