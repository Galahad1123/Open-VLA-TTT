# LIBERO 正式训练日志审计与 server-client 初评

检查日期：2026-10-05。训练日志时间：2026-10-01 至 2026-10-02。

结论：未发现数值发散、阶段切换或 checkpoint 保存异常；训练后的模型已具备闭环操作能力。在固定最终 80k checkpoint、每任务初态 0 的 40 次初评中成功 27 次（67.5%）。**这是初评，不是完整 benchmark、多种子结果，也不意味着已经复现官方分数或排除所有实现问题。**

## 1. 实際训练配置与完整性

来源日志：

- `outputs/logs/libero-local-scratch-seed1000-stage1.log`
- `outputs/logs/libero-local-scratch-seed1000-stage2.log`

实际模型类型 `vla_ttt`：本地 Evo-1，未启用 TTT。Stage 1 无策略 pretrained_path，仅加载固定通用 InternVL VLM、随机初始化 action head。Stage 2 加载本次 Stage 1 的 `005000`，`resume=false`，VLM 解冻。

| 项目 | Stage 1 | Stage 2 |
| --- | ---: | ---: |
| 训练步数 | 5,000 | 80,000 |
| 单卡 batch | 8 | 4 |
| 优化器 | AdamW，lr=1e-5，weight_decay=1e-3 | 同左 |
| warmup | 1,000 | 1,000 |
| 实际 save_freq | 2,500 | 10,000 |
| 日志窗口 | 100 个，每 50 步 | 1,600 个，每 50 步 |
| 优化器 step 状态 | 114 个均为 5,000 | 726 个均为 80,000 |
| 最终模型张量 | 728 个，全部有限 | 728 个，全部有限 |
| 日志记录 allocated 显存 | 约 6.75 GiB | 约 14.49 GiB |
| 训练循环耗时（日志进度条） | 44 分 24 秒 | 22 小时 21 分 42 秒 |

两个日志均有 `End of training`，checkpoint 的 `training_step.json` 与预期一致，50 步日志窗口完整。所有解析出的 loss、梯度范数和 LR 都是有限值，没有发现 Traceback/OOM/NaN/Inf。日志窗口不是逐步遥测，但所有 AdamW step 达到目标值，未发现优化器更新丢失的迹象。

Stage 2 的三条 finetune flag override 警告是预期的阶段切换：Stage 1 保存的冻结标志被 Stage 2 默认值覆盖为 True，不是错误。

用户实际每 10,000 步保存 Stage 2，而非之前命令示例的 5,000 步；这只改变 checkpoint 保存频率，不改变训练步数或 batch。

## 2. Loss 与训练预算

下表使用日志中 50 步窗口均值再次求平均；各窗口等长，没有将末尾单个 batch 当成收敛结论。

| 指标 | Stage 1 | Stage 2 |
| --- | ---: | ---: |
| 最初 500 步平均 loss | 0.5462 | 0.3956 |
| 最后 500 步平均 loss | 0.3845 | 0.0503 |
| 日志窗口梯度范数范围 | 1.441–3.585 | 1.847–6.322 |

Stage 2 分段平均 loss：

| 步数区间 | 平均 loss |
| --- | ---: |
| 0–1k | 0.40235 |
| 1k–10k | 0.32199 |
| 10k–20k | 0.13551 |
| 20k–40k | 0.07860 |
| 40k–60k | 0.05939 |
| 60k–70k | 0.05204 |
| 70k–80k | 0.05012 |

Loss 总体明显下降，末段趋缓，同时 cosine LR 已接近 0。日志的 grdn 是裁剪前范数，数值大于 clip_norm=1 不表示裁剪失效；没有据此发现梯度爆炸。

数据为完整 `HuggingFaceVLA/libero`（revision `86958911c0f959db2bbbdb107eb3e17c5f9c798e`），1,693 episodes、273,465 frames、40 tasks。Stage 1 采样 40,000 个帧起点，约 0.146 epoch；Stage 2 采样 320,000 个，约 1.170 epoch。不能把单卡小 batch 的相同步数视为与大 global batch 训练等量。

`dataset.eval_split=0`，没有 held-out loss。因此不能仅凭训练曲线判定泛化、是否过拟合或能力已经收敛，也不应把所有 rollout 失败直接归因于模型记忆能力。

## 3. 固定 checkpoint 的闭环初评

评测前固定最终 checkpoint，没有用本轮测试成功率挑选 checkpoint：

`outputs/libero-local-scratch-seed1000-stage2/checkpoints/080000/pretrained_model`

`model.safetensors` SHA256：`94040fb0dd2c7974619839b86b4acb88df1f8a27280cc0b07ab4861cc1c6891e`。

协议：

- policy：主项目 `.venv`（Python 3.12）；环境：`environments/libero/.venv`（Python 3.11），独立进程，通过 localhost:8780 连接。
- 四个 suite，每个 task ID 0–9，每任务 1 episode、官方打包初态索引 0，policy/environment seed=1000。
- 每任务启动新的环境进程；每 episode 重置 policy/action queue。
- 图像采集 256×256，由模型处理到 448×448；chunk_size=50、执行 10 步后重算；32 个 flow 推理步；沿用 checkpoint 的连续 gripper，不新增二值化。
- horizons：Spatial/Object 280，Goal 300，LIBERO-10 520；每次 reset 先执行既有的 10 步稳定动作。
- 成功由环境 `check_success()` 判定；不以基础设施异常充当任务失败。
- 全程离线使用已有资产；不改 checkpoint、归一化统计、动作处理或训练数据。

| Suite | 成功 / 总数 | 初评成功率 | 未成功 task ID |
| --- | ---: | ---: | --- |
| LIBERO Spatial | 7 / 10 | 70% | 0, 5, 8 |
| LIBERO Object | 9 / 10 | 90% | 5 |
| LIBERO Goal | 6 / 10 | 60% | 0, 3, 5, 9 |
| LIBERO-10 | 5 / 10 | 50% | 2, 3, 4, 8, 9 |
| 合计 | 27 / 40 | 67.5% | — |

所有 40 个预声明 task/episode 均完成，40 份结果的模型 SHA256 相同，基础设施错误为 0。13 个未成功 episode 都达到各自步数上限。累计 8,809 个环境步，episode 运行时间合计约 274.8 秒，不含进程启动等开销。

LIBERO-10 初评较弱，但这不足以证明失败来自记忆缺陷：长任务中的控制误差累积、训练量、数据覆盖和执行设置都可能影响结果。本轮未录像，因此未逐个确定失败行为的原因。

只用了每任务一个初态，这个数值不能与每任务多初态、多 seed 或不同 action/gripper 协议的结果直接比较。

## 4. 产物与后续建议

全部路径相对于当前工作树：

- `artifacts/experiments/libero-trained-eval-20261005/run.sh`：固定 80k checkpoint 的 40 次初评命令。
- 同目录 `analyze_training.py`、`training-audit.json`、`training-log-windows.json`：日志和 checkpoint 审计。
- 同目录 `summarize_eval.py`：强制检查全部 40 个预声明任务、错误数和 checkpoint 身份一致后汇总。
- 同目录 `server.log`、40 份 task `.log`。
- `outputs/eval/libero-stage2-080000-preliminary-20261005/`：40 份原始 JSONL 与 `summary.json`。

上述实验脚本和原始产物位于 Git 忽略目录，需要单独归档；本报告保存在 docs。已有训练产物和用户其他未提交修改保持不动，没有修改模型或评测实现，没有提交或推送代码。

可以把最终 80k 固定为候选 baseline，下一步扩展初态覆盖（例如每任务 50 episodes，共 2,000），并单独安排失败行为录像诊断；录像不是现有 runner 的本轮产物。不要根据同一测试集反复挑选 checkpoint 或调整执行超参数后把分数当成独立测试结果。

本次评测服务已退出，GPU 计算进程检查为空。正式扩展评测未自动启动。
