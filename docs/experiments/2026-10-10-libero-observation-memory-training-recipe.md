# LIBERO 观测记忆训练 recipe 与资源预检

记录日期：2026-10-10。工作区：`.worktrees/observation-memory`，分支：`feat/observation-memory`。

本文汇总讨论中已经确定的研究协议、首轮推荐超参数，以及 2026-10-10 的真实 checkpoint / GPU 预检结果。**这是待落实的正式训练 recipe，不代表现有 trainer 已完整支持，也不代表已启动正式训练。**

状态约定：

- **已确定**：用户确认的训练方向与实验语义。
- **首轮推荐**：采用明确默认值推进实现；尚未通过正式训练证明最优。
- **实测**：本次探测直接验证的结果，需同时阅读其资源限制。

本记录更新此前 recipe v1 的两项建议：初始化优先采用共同的 **Stage1 5k**，而非 Stage2 80k 或挑选中间点；有效 batch 从固定 32 改为**暂建议 4、由实际训练检查调整**。20 GiB 是并行预检的保护上限，不是模型的正式显存要求。

## 1. 研究范围与共同初始化

### 已确定的方向

- 首轮使用与 Evo1-GR00T baseline 相同的 LIBERO 数据与 40 个任务：Spatial、Object、Goal、LIBERO-10。
- 首个实现与训练对象推荐 T²Mem-style，随后用 MAG 做结构对照；后续 AQMem / MAC / MAG / MAL 采用同一主要训练协议。
- 保留 Evo1-GR00T 的 VLM 和 Action Expert 主干；不因新增记忆结构而重新随机初始化已有 Action Expert。
- Policy 阶段全量训练视觉编码器、视觉到语言投影、语言模型、Action Expert，不使用 LoRA。
- 不额外切分训练示范验证集；预先固定预算、评测点和报告 checkpoint，避免根据测试结果反复挑选最优 checkpoint。

### 共同起点：Stage1 5k

优先使用已经完成 Stage1 的共同权重，让无记忆分支和记忆分支从同一处开始后续训练：

```text
/home/hello/lizk/vlas/vla-ttt/.worktrees/evo1-plugin-foundation/
outputs/libero-evo1-groot-scratch-seed1000-stage1/checkpoints/005000/pretrained_model
```

上面分行为展示，实际路径连续。该选择符合“不为节省少量训练而复杂化对照”的原则。已有 Stage1 可以复用；若以后重新训练 Stage1，也应让各分支复用同一个产物。

各分支为新阶段建立新的优化器和调度器，不直接续接 Stage1 已衰减至零的学习率。这是**共同权重初始化后开始新阶段**，不是声称精确恢复旧训练。原 Stage2 80k 的已评测结果保留为历史参考，不作为唯一对照。

已检查可用 checkpoint：Stage1 2.5k / 5k；Stage2 10k 至 80k、每 10k 保存一次。均发现模型、processor、训练配置、优化器张量及参数组、scheduler、RNG 与 training step 文件；本次没有实际执行精确恢复。

## 2. 数据、采样与时序

| 项目 | 配置与状态 |
| --- | --- |
| 数据集 | `HuggingFaceVLA/libero`，与 baseline 一致 |
| Revision | `86958911c0f959db2bbbdb107eb3e17c5f9c798e` |
| 本地数据根目录 | `/home/hello/lizk/datasets/libero` |
| 规模（实测元数据） | 1,693 episodes、273,465 frames、40 tasks，10 FPS |
| 任务采样 | 均匀选择任务 |
| Episode 采样 | 在选中任务内均匀选择 episode；不因长度增加被选概率 |
| 有效 batch | 同一任务；梯度累积的 microbatch 也来自该任务 |
| 序列 | 首轮完整 episode，内部时间顺序固定；各 episode fast state 独立 |
| 记忆初始化 | 每条新 episode 从当前可学习的 W0 开始 |
| Episode 边界 | 清空 fast weights、辅助状态、时钟与动作缓存 |
| 子轨迹 | 首轮不默认使用；若以后引入，须明确冷启动或历史前缀策略 |

LIBERO 本轮不设置任务 demonstration 前缀。完整 episode 从起点自然积累历史，无需额外预热。若以后随机截取 episode 中段，“没有任务 demonstration”不等于“无需恢复中段之前的历史”。

| 时间相关量 | 已确定设置 |
| --- | --- |
| 预测长度 H | 50 个动作 |
| 执行长度 E | 10 个动作 |
| 写入间隔 | E/2 = 5 个物理环境步 |
| 决策位置 | 0、10、20……；计算动作监督 |
| 中点位置 | 5、15、25……；仅观测写入，不产生动作 loss 或采样动作噪声 |
| 推理去噪次数 | 32，沿用 baseline |
| 读写顺序 | 当前决策全过程读取旧记忆，完成动作预测后写入当前观测一次 |

环境步、观测事件、去噪迭代、外层 optimizer 更新分别计数。推理不能漏掉应处理的中点观测；训练时动作标签不进入记忆写入路径。

## 3. 外层目标与梯度语义

保留完整 H 步的 flow-matching 动作监督，屏蔽超出 episode 的动作与无效动作维度。每个决策位置、每个样本独立采样噪声和 flow time，沿用 baseline 的 Beta(1.5, 1)、时间缩放 0.999 及其流方向约定，不混用其他实现的时间/目标符号。

推荐采用三级平均，保证均匀 episode 采样与梯度贡献一致：

1. 每个动作 chunk 按有效动作元素平均。
2. 每条 episode 按有效监督决策数平均。
3. 有效 batch 内按 episode 平均。

中点不进入监督决策数，但其写入可以被后续动作损失监督。暂不增加记忆内容标签或辅助任务损失。

TBPTT 段不是独立 episode：跨段保留 fast weights 的数值，仅 detach 梯度。整个有效 batch 的所有段和累积 microbatch 完成 backward 后，才执行一次外层更新；期间 slow weights 不变。每段使用相同的全 episode / 有效 batch 归一化分母，不能各自平均后造成权重偏移。

## 4. 交替训练与参数组

从共同 Stage1 权重开始，新增记忆组件初始化后，先 Memory，再 Policy，重复交替。

| 阶段 | 更新的慢参数 | 冻结的慢参数 | Episode fast weights |
| --- | --- | --- | --- |
| Memory | Interface、记忆读写/输出投影、W0、内层步长参数、融合组件 | 完整 VLM、Action Expert | 正常读取和更新 |
| Policy | 完整 VLM、Action Expert | 新增记忆组件 | 正常读取和更新 |

冻结慢参数不等于对整个模块使用 `no_grad`：必须保留通往当前活动参数组的必要输入梯度。新 episode 开始时重新从 W0 构建状态，不跨独立训练样本携带 fast weights。

### 首轮推荐外层超参数

| 参数组 | 峰值 LR | Weight decay |
| --- | ---: | ---: |
| 完整 VLM：视觉、投影、语言 | 1e-5 | 1e-3 |
| Action Expert | 1e-5 | 1e-3 |
| Interface、记忆 Q/K/V 和输出投影等新增矩阵参数 | 7.5e-5 | 1e-5 |
| 可学习 fast-weight 初始化 W0 | 7.5e-5 | 0 |
| T²Mem 融合 gate | 5e-4 | 0 |
| 自适应内层步长参数 | 7.5e-5 | 0 |

统一 AdamW：betas=(0.9, 0.999)、eps=1e-8。Bias、normalization 参数不施加 weight decay；累积结束后对当前活动参数统一裁剪，阈值 1.0。

VLM / Action Expert 采用已经跑通的 Evo1-GR00T 设置；记忆与 gate LR 参考 T²Mem，新增矩阵 decay 参考 RoboTTT。W0、gate、步长参数不施加 decay 是本平台选择。MAG 等不同融合网络不直接整体套用 T²Mem gate 的高 LR，默认归入普通记忆组。

内层规则选 `t2mem_adaptive`，保留现有曲率/残差/对齐度校准与可学习乘子；上表为外层 LR，不额外叠加 RoboTTT 的固定内层 0.1。校准量 stop-gradient，不将整个 fast-weight 递推替换为一阶近似。

| 调度项目 | 首轮推荐 |
| --- | --- |
| 交替周期 | Memory 500 次外层更新 → Policy 500 次外层更新 |
| 首轮预算 | 两轮交替，共 2,000 次外层更新，每组各 1,000 次 |
| Warmup | 各组首次 50 次实际更新 |
| 后续调度 | Cosine，衰减至该组峰值 LR 的 10% |
| 切换阶段 | 不重置 optimizer moments 或 scheduler；各组按自身实际更新次数计数 |
| 种子 | 首轮 1000 |

2,000 次是固定首轮预算，不是已验证的收敛步数。因有效 batch 推荐值已降低，该预算也不再代表与论文相同的数据暴露量。必须同时记录 episode、观测事件、监督决策数和墙钟时间；不能直接与 baseline 帧级训练步数比较。

## 5. 精度、增强与显存设置

实际 Stage2 80k 配置使用 FP32 VLM 权重、BF16 AMP；Stage1 5k 保存的 VLM dtype 为 BF16。后续全量训练将加载的权重转为 FP32，使用 BF16 AMP；这不恢复 BF16 checkpoint 已丢失的精度。Fast weights 与内层更新计算保持 FP32。沿用 dropout=0.2，冻结的 backbone 模块使用 eval 模式，采用非重入 gradient checkpointing。

为数值对照统一 attention 后端；本次测试确认统一 MATH 后端后 baseline bypass 完全一致。该一致性只覆盖测试输入，不替代真实 rollout 验证。

图像增强沿用 baseline 的变换类型、幅度与选择设置，**推荐在同一 episode、同一相机内固定变换随机参数**，避免逐帧随机变化污染时序信号。训练与评测使用相同的 processor / 动作归一化来源，评测不启用训练增强。

| 资源设置 | 当前结论 |
| --- | --- |
| Microbatch | 当前并行训练环境下推荐 1 条完整轨迹 |
| 有效 batch | **暂建议 4**，通过梯度累积实现；不是已测得的收敛最优值 |
| BPTT | 原则上先测试完整 BPTT；本次并行条件下优先采用 TBPTT=8 |
| TBPTT 单位 | 8 次观测事件，即 stride=5 下连续 8 个采样位置；不是 8 个动作或去噪迭代 |
| 记忆跨度 | 仍为完整 episode，不因 TBPTT 截断而重置 |

不能用梯度累积解决单条完整轨迹计算图过大。若资源不足，先调整 microbatch、checkpointing 与分段搬运，再按实测调整 TBPTT；不静默删除长轨迹或缩短实际记忆历史。各架构的正式对照应记录并尽量统一截断协议。

## 6. 公平对照、保存与评测

无记忆继续训练分支与 T²Mem 分支从同一 Stage1 5k 权重出发。首轮优先匹配 Policy 实际更新次数、Policy 阶段 episode / 监督决策预算、数据顺序、增强、有效 batch 与调度。T²Mem 额外 Memory 阶段的数据与计算单独报告。

该协议衡量整个记忆训练方案相对匹配 Policy 学习预算的收益，**不宣称总计算预算完全相等**。如需回答等计算预算问题，另设对应 baseline。

至少报告：

1. 原有 Evo1-GR00T Stage2 80k 历史结果。
2. 从共同 Stage1 起点、按匹配预算继续训练的无记忆对照。
3. T²Mem 正常写入。
4. 同一个训练后 T²Mem 禁用写入、始终读取 W0，但保留新增结构与其余推理流程。

“禁用写入”与“关闭整个新增分支”是不同消融。特别是 T²Mem-style，memory gate=0 仍保留 interface token 路径，不等于原 baseline。

推荐每 250 次外层更新保存 checkpoint，在 1,000、2,000 次完成完整评测；正式报告预先固定使用 2,000 次 checkpoint。沿用已有 baseline 的四套任务、每任务 20 个初始状态，共 800 次 rollout，匹配原 seed 与环境配置。不能仅报告测试集上事后选择的最好 checkpoint。

训练状态保存应覆盖当前交替阶段、两组 optimizer / scheduler、更新计数、随机状态和采样状态。序列边界保存时不需要保存该 episode 的 fast state；若支持中途保存，则必须单独定义恢复语义。现有序列 trainer 仍是 warm start，完整恢复能力尚待实现。

## 7. 2026-10-10 实测结果

### 环境与限制

- GPU：RTX PRO 5000 72GB Blackwell；PyTorch 查询总量约 71.10 GiB。
- 另有训练持续运行。预检开始和结束时 CUDA 查询剩余显存均约 40.62 GiB。
- `nvidia-smi` / NVML 不可用，但 PyTorch CUDA 可用。
- 每个探测进程设置 PyTorch allocator 上限 20 GiB，并要求启动前剩余显存至少 30 GiB。CUDA 上下文等非 allocator 分配不包含在该上限中。
- 不停止、重启或修改原训练进程；未修复驱动，未保存新模型权重，未启动正式训练。
- 轨迹长度：最短 75 帧，中位数 140 帧，P90=265 帧，最长 505 帧。中位样本为 episode 469，最长为 episode 306；对应 28 / 101 次观测事件。

### Checkpoint 与输出一致性

Stage1 5k 真实权重已成功初始化 T²Mem。关闭记忆后，在相同随机种子下比较形状 `[1, 50, 24]` 的动作输出：

| 比较 | 最大绝对差异 |
| --- | ---: |
| 默认 BF16 attention 后端配置 | 0.002685546875 |
| 统一 MATH attention 后端 | **0.0** |

对齐后未放宽原比较容差。该探测的峰值分配约 6.12 GiB。

### 训练资源探测

下表为**本探测进程的 PyTorch 峰值显存**，不是整卡占用。耗时不含模型初始化，但包含被测序列处理、反向与更新；是在另一个训练并行运行时测得。

| 阶段 | 观测事件 / microbatch / 截断 | 更新数 | 峰值 allocated / reserved（GiB） | 耗时 | 结果 |
| --- | --- | ---: | --- | --- | --- |
| Memory | 28 / 1 / 完整 BPTT | 1 | 8.73 / 9.30 | 10.41 s | 通过 |
| Policy | 28 / 1 / 完整 BPTT | 未完成 | 失败前 allocated 峰值接近 19.81 | — | 接近限制后报 NVML allocator 错误 |
| Policy | 101 / 1 / TBPTT=8 | 1 | 13.62 / 14.32 | 67.26 s | 最长轨迹通过 |
| Policy | 28 / 1 / TBPTT=8 | 连续 2 次 | 17.99 / 19.67 | 合计 37.48 s | 含 AdamW 常驻状态通过 |
| Policy | 8 / 2 / TBPTT=8 | 第二次未完成 | — | — | 第二次更新过程中报 NVML allocator 错误 |

失败的实际异常是 `nvmlInit_v2_()` 的内部断言，不能把它改写成已经捕获的标准 CUDA OOM。结合设置的 allocator 上限与显存轨迹，只能认为本次受限配置未通过；不能据此判断独占整卡时完整 BPTT 不可行。

Memory / Policy 的 optimizer 状态张量分别约 0.48 / 5.31 GiB。Policy 连续两次更新的峰值明显高于首步，说明只测首次 forward / backward 会低估实际训练开销。

### 不能从本次测试推导的结论

- 没有验证有效 batch=4 的正式梯度累积和收敛，也没有完成 4/8/16/32 的有效 batch 对比。
- 两个阶段单独探测；尚未测量交替时双方 optimizer 状态同时常驻的完整峰值。
- 最长轨迹只完成一次更新；中位轨迹完成两次，不是整个数据集的稳态压力测试。
- microbatch=2 使用相同真实样本复制来测量张量形状开销，不是两个独立 episode 的训练质量实验。
- 未启用生产级时序增强或完整分组 LR / scheduler。资源探测使用简化优化器，不是上述 recipe 的效果验证。
- 完整 BPTT 的 Memory 测试只覆盖中位轨迹；未证明所有长轨迹都能使用完整 BPTT。
- 没有新的 LIBERO 成功率、训练收敛或独占 GPU 吞吐结果。

## 8. 实现状态与后续验收

已具备：新模型、三种更新规则、观测时钟、TBPTT 基础机制、基础交替训练、checkpoint 权重加载，以及上述真实 GPU 探测。

落实本文 recipe 仍需补齐：

- 同任务的任务/episode 分层采样与完整变长 episode batch。
- 三级损失归一化与有效 batch 梯度累积。
- 独立参数组、交替更新计数、warmup/cosine 调度。
- 时序一致图像增强及公平对照的统一 attention 后端。
- 训练状态完整保存/恢复与可审计运行配置。
- 双方 optimizer 常驻的交替资源测试，再确认最终 microbatch、TBPTT 与有效 batch。

在完成这些实现和短程验收前，不把现有 `train_robottt` 的示例命令当作本文完整 recipe 的启动命令。常规工程默认值按本文推进，不再逐项询问；改变实验含义的调整需明确记录。

## 9. 证据与参考

- [原始预检报告](../../artifacts/preflight-20261010/report.json)
- [资源探测脚本](../../artifacts/preflight-20261010/probe.py) / [输出对齐脚本](../../artifacts/preflight-20261010/bypass.py)
- [Memory 完整 BPTT](../../artifacts/preflight-20261010/memory-28.log)
- [Policy 最长轨迹](../../artifacts/preflight-20261010/policy-101-segment8.log)
- [Policy 连续两次更新](../../artifacts/preflight-20261010/policy-28-segment8-two-updates.log)
- [Policy 完整 BPTT 失败](../../artifacts/preflight-20261010/policy-28.log) / [microbatch=2 失败](../../artifacts/preflight-20261010/policy-batch2-segment8.log)
- [统一 attention 后端结果](../../artifacts/preflight-20261010/bypass-matched.log)
- [架构设计](../superpowers/specs/2026-10-09-observation-memory.md) / [实现说明](../observation-memory.md)
- [RoboTTT](https://arxiv.org/html/2607.15275)：§3.2 序列训练与 TBPTT、附录 A.2 优化器与训练设置。
- [T²Mem](https://arxiv.org/html/2609.36720v1)：§3.4 交替学习、附录 A 自适应写入、附录 B 参考 recipe。

`artifacts/` 为本地忽略目录；表格中的关键结果已在本文留档，但原始日志和探测脚本不会自动随 Git 文档分发。报告中的脚本哈希对应最终文件版本，早期探测没有 `repeats` 字段时实际只进行一次更新。

## 2026-10-12 补充：独占 GPU 验收

驱动更新后已重新测试真实最长轨迹，完整 BPTT 的第二轮 Policy 在未设置 allocator 上限时仍然 OOM。新的 episode-parallel trainer 已具备本文此前列为待实现的分层采样、归一化、累积、参数组调度、时序增强与完整训练状态恢复；其 GPU 资源测试记录见 [BPTT / TBPTT 验收](2026-10-12-bptt-resource-validation.md)。上文第 7、8 节保留为 10 月 10 日的历史状态。

本机资源候选 **TBPTT=64、microbatch=1、有效 batch=4** 已通过实测，T²Mem 两轮交替峰值 allocated / reserved 为 **51.06 / 54.04 GiB**，恢复后最终模型和优化器状态对齐。候选配置见 [72GB recipe](../../configs/ttt/t2mem/libero/recipe-72gb.json)；适用范围、增强顺序修复与目标服务器待验证项见上述新报告。

## 合并前协议澄清

后续讨论保持 **TBPTT=8、microbatch=1、有效 batch=4** 为当前方案；64 是已通过 GPU 压力测试的候选配置，尚未用训练效果证明优于 8。2000 次外层更新仍是探索性预算，不代表与已有 80k-step baseline 或论文 batch=32 等训练量。正式训练量暂不在本次代码合并中调整。8 在当前完整入口下的同条件 GPU 资源对照尚未补测；此前 8 的测试条件见第 7 节。
