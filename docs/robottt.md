# Evo1-GR00T + RoboTTT

首版实现位于 foundation 的 `src/lerobot_policy_vla_ttt/models/robottt/`，独立
注册为 `robottt`。`evo1_groot` 仍是无记忆 baseline。GPU 训练收敛与 RoboMME
闭环对照尚待运行；CPU 机制测试不能代替这些实验。

## 代码入口

| 位置 | 职责 |
| --- | --- |
| `ttt/functional.py`, `fast_models.py` | 函数式两层 GeLU MLP、时间 RoPE、外层学习的 W0 |
| `ttt/state.py` | 每个样本独立的 fast weights 和更新位置；显式 detach |
| `ttt/updates/gradient_descent.py` | 普通 GD，正值可学习学习率 |
| `ttt/memory.py` | Q/K/V、KV MSE、update-then-read、逐维 tanh gate、按行 reset |
| `models/robottt/action_head.py` | attention → memory → FFN，序列状态与逐去噪累计 |
| `models/robottt/modeling_robottt.py` | 独立 policy、训练状态、初始化审计、严格 checkpoint |
| `training/sequence.py` | episode 内连续窗口、决策 stride、padding mask |
| `training/flow_matching.py` | 每个样本和轨迹位置独立采样噪声等级 |
| `training/tbptt.py` | 分段 backward，跨段只保留 detached 数值 |
| `training/train_robottt.py` | 本地 LeRobot 数据集序列训练入口 |

复用基座的 token、DiT 组件和 I/O 适配方法；新模型自己管理记忆生命周期。
公共 TTT 不依赖 Evo1-GR00T。不创建 Titans/MAC/MAG/MAL 空实现。

## 首版协议

- 每个交替 cross/self-attention 子层后、FFN 前插入一次记忆残差，不重复添加输入残差。
- 写入全部 state、16 个新增 register、完整 noisy action chunk 的 attention 隐藏特征。
  VL tokens 只经 cross-attention 提供条件。token 排列沿用基座：state/register/action。
- KV loss 对单个样本内所有 token 和特征取均值；批内各行独立更新。
- W0、QKV、输出投影、学习率与 gate 属于外层参数。学习率 `softplus(raw_lr)`
  初值恰为 0.1；逐维 `tanh(gate)`，gate 初值 0.001；RoPE theta=10000。
- `memory_dim=128`、`memory_hidden_dim=256` 是可调整的平台轻量默认值，
  不宣称为论文原模型尺寸。fast MLP 含 bias，外层 loss 对有效动作标量取均值。
- 每次去噪累计更新，最后状态带入下一动作决策。RoPE 位置按有效记忆更新推进，
  不等同于环境步；相同 chunk 内共享时间位置。去噪步数影响记忆更新次数。
- 队列中的执行步不更新记忆；`policy.reset()` 同时清空队列和所有层的状态。
  评测服务为单环境接口；训练支持 batch 行独立 reset。
- 训练每个轨迹位置采样一次 `t=0.999*(1-Beta(1.5,1))`，
  `x_t=(1-t)*noise+t*action`，目标 `action-noise`。这不是部署多次去噪的完全同构过程。
- 推理使用 `no_grad`，仅局部开启 fast-weight 求梯度；不修改 slow parameters。
  外部 `torch.inference_mode()` 会明确报错，避免静默失去内循环更新。
- checkpoint 仅保存 slow parameters/config/processors，不保存某个 episode 的 fast state。

严格关闭记忆的动作头路径同时去除新增 registers；可以与初始化来源 baseline
做数值回归。关闭记忆时不能选择 `memory_only` 训练；该组合会明确报错。仅将 gate 置零而保留 registers，不是严格 baseline 等价。

## 训练

必须使用序列入口；普通随机单帧 LeRobot trainer 调用 `forward()` 只训练独立
位置，不能替代 Sequence Action Forcing + TBPTT。`forward_step()` 是显式状态接口。

从已经训练完成的 `evo1_groot` checkpoint 初始化（路径替换为实际资产）：

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m lerobot_policy_vla_ttt.training.train_robottt \
  --checkpoint /path/to/evo1_groot/pretrained_model \
  --dataset-root /home/hello/lizk/datasets/libero \
  --dataset-repo local/libero --benchmark libero \
  --output /path/to/new/robottt-memory-only \
  --device cuda --memory-training memory_only \
  --steps 5000 --batch-size 2 --sequence-length 8 \
  --decision-stride 10 --tbptt-length 4 \
  --learning-rate 1e-4 --warmup-steps 500
```

这些数值是起始研究配方，尚未验证任务收敛。源 checkpoint 决定动作 horizon、
执行长度、去噪次数、归一化统计和 VLM 精度。初始化严格检查兼容配置和每个
基座张量，只允许新增 registers/memory；映射报告写入 `run.json`。
`memory_only` 冻结基座，仅训练 memory 与新增 registers。随后以输出的
`step-XXXXXXXX` 为 `--checkpoint`，用 `--memory-training joint` 进行联合训练。
联合阶段使用 stage2 的基座 finetune 设置。VLM 精度沿用来源配置；正式联合训练
前应对齐既有 baseline 的精度与显存配方。

每个 window 从 W0 开始；window 内保持时间顺序，window 间可 shuffle。
每段执行 backward 后释放图，fast-state 数值 detach 后传入下一段；W0 不会
在每个段边界重新接回计算图。外层 optimizer 每个完整 window batch 更新一次。
各段 loss 都按整批 window 的有效动作标量数归一化。

尾部 padding 不写入、不监督。底层接口支持独立的 write/supervision/reset mask；
context-only 行屏蔽动作输入及 loss，但仍写入观测特征。当前 CLI 读取 canonical
控制帧数据，不宣称已支持 RoboMME demo 视频完整上下文协议。

`run.json` 记录 stride/horizon/执行长度/去噪步数、初始化映射、数据元信息和重叠
标签路径。`metrics.jsonl` 记录 loss、有效监督量、段数及记忆更新次数。
保存点含 policy、processors 和 `optimizer.pt`。当前 `--checkpoint` 是权重和
processor warm start，**不会恢复 optimizer、随机数或数据迭代位置**；不是精确续训。

## 评测

使用已有评测服务器，选择 `--implementation robottt` 并传入上述 checkpoint。
服务为 RoboTTT 选择 `no_grad`，其他 baseline 保持原有 inference-mode 路径。
每个 episode 必须 reset；不能把前一 episode 的记忆带入标准指标。

## 研究限制和依据

完整动作隐藏特征与逐去噪累计是已确认的首版选择。训练 stride 小于预测 horizon
时，历史真实动作 chunk 的标签可能进入后续记忆，TBPTT detach 并不消除这条
数值信息路径。推理保留未执行的预测计划不等于获得未来真实标签。完整讨论见
[更新时机与隐藏特征记录](superpowers/specs/2026-10-05-robottt-update-timing-and-hidden-features.md)。
后续对重叠标签做扰动诊断；不能把该配方宣传为已证明无信息泄漏。

机制依据：[RoboTTT 论文](https://arxiv.org/html/2607.15275)；
累计去噪与时间位置参考[社区固定版本](https://github.com/lucidrains/robo_ttt/tree/98eb8837f66ec28a0d131fc8bfabb8d6c7bb017e)。
本实现未沿用社区 hook 重复残差、默认 forget 或不同学习率初值。
[TTT-Video 固定版本](https://github.com/test-time-training/ttt-video-dit/tree/e43707e9b8594ace5a73c4a4438f486d5d17729c)
仅作为结构参考，未移植视频双向机制或 CUDA kernels。
