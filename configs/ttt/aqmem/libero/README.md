# AQMem + LIBERO：Stage2 初始化、联合训练、混合任务

此实验先验证加入 memory 后能否保留通用任务能力，再将主要预算转向 RoboMME。
它不改变 RoboMME/T²Mem 的交替训练 recipe。训练 loss 不能替代 rollout 成功率。

- 初始化：Evo1-GR00T LIBERO Stage2 **20k**；30k 可作为独立初始化对照。
- H50/E10、TBPTT 8 个观测事件、global batch 8；每个 rank 顺序处理 episode，microbatch=1。
- task 按随机排列循环抽取，余数不放回；任务数至少 8 时每 batch 8 个不同任务。
  再在各任务内均匀有放回抽 episode，episode 长度不影响权重。
- VLM（视觉、投影、语言）、action expert、memory 每步联合更新，不冻结 VLM。
  一个 AdamW、一个全局梯度裁剪和一个 outer-step 计数。
- policy LR 1e-5；memory（含 AQMem action gates）LR 7.5e-5。
  W0/内层更新参数/bias/norm 不 decay，AQMem 不使用 T²Mem 的 gate 高 LR。
- gate 初值 0.001，adaptive writes；不是严格零扰动初始化。零 gate 保留 baseline
  的回归已单独覆盖。当前预算 2000 步、warmup 50，每 250 步保存。

仅支持 `HuggingFaceVLA/libero` revision
`86958911c0f959db2bbbdb107eb3e17c5f9c798e`，不是 `lerobot/libero`。
模型维度、归一化与 processors 继承源 checkpoint；增强配置继承其 `train_config.json`。

## 本机命令

在 observation-memory worktree 根目录执行。以下路径显式选择已存在的 Stage2 20k；
改成 `030000` 并换输出目录即可启动 30k 对照，不能作为同一 run 的 exact resume。
跨机器时替换路径或使用 [machine 配置](../../../../docs/portable-training.md)。

```bash
export PYTHONPATH=src
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PYTHON=/home/hello/lizk/vlas/vla-ttt/.venv/bin/python
CHECKPOINT=/home/hello/lizk/vlas/vla-ttt/.worktrees/evo1-plugin-foundation/outputs/libero-evo1-groot-scratch-seed1000-stage2/checkpoints/020000/pretrained_model
DATASET=/home/hello/lizk/datasets/libero
RECIPE=configs/ttt/aqmem/libero/recipe.json
MEMORY=configs/ttt/aqmem/libero/memory.json

# 配置检查：不加载权重，不创建输出，不使用 GPU。
CUDA_VISIBLE_DEVICES='' "$PYTHON" -m lerobot_policy_vla_ttt.training.train_episode_parallel \
  --architecture aqmem --checkpoint "$CHECKPOINT" --dataset-root "$DATASET" \
  --recipe "$RECIPE" --memory-config "$MEMORY" \
  --output outputs/aqmem-libero-stage2-20k --dry-run

# 单 GPU：先运行两步，保留完整 2000 步 scheduler，建立 AdamW 状态。
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m lerobot_policy_vla_ttt.training.train_episode_parallel \
  --architecture aqmem --checkpoint "$CHECKPOINT" --dataset-root "$DATASET" \
  --recipe "$RECIPE" --memory-config "$MEMORY" \
  --output outputs/aqmem-libero-stage2-20k-smoke --stop-after 2

# 从两步检查点继续，必须使用新的输出目录；不重复指定初始化 memory 配置。
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m lerobot_policy_vla_ttt.training.train_episode_parallel \
  --resume outputs/aqmem-libero-stage2-20k-smoke/step-00000002 \
  --dataset-root "$DATASET" --output outputs/aqmem-libero-stage2-20k-continued
```

两步只是启动检查，不足以覆盖最长轨迹或可靠估算训练时长。
正式运行前测量代表性长短轨迹、优化器状态建立后的峰值显存及每步时间；
预计剩余时间约为剩余步数 × 代表性平均 update_seconds，再计入保存开销。
尚无此 AQMem 联合 recipe 的 GPU 峰值/吞吐实测，不能套用 T²Mem 交替训练数据。

## 多 GPU 启动（待 CUDA/NCCL 验收）

保持上述变量，选择以下一组配置：

| 卡数 | CUDA_VISIBLE_DEVICES | NPROC | 每 rank episode/更新 |
|---|---|---|---|
| 2 | 0,1 | 2 | 4 |
| 4 | 0,1,2,3 | 4 | 2 |
| 8 | 0,1,2,3,4,5,6,7 | 8 | 1 |

```bash
export CUDA_VISIBLE_DEVICES=0,1
NPROC=2  # 4/8 卡时同时调整可见 GPU 列表
"$PYTHON" -m accelerate.commands.launch --multi_gpu --num_processes="$NPROC" \
  --mixed_precision=no -m lerobot_policy_vla_ttt.training.train_episode_parallel \
  --architecture aqmem --checkpoint "$CHECKPOINT" --dataset-root "$DATASET" \
  --recipe "$RECIPE" --memory-config "$MEMORY" \
  --output "outputs/aqmem-libero-${NPROC}gpu-smoke" --stop-after 2

"$PYTHON" -m accelerate.commands.launch --multi_gpu --num_processes="$NPROC" \
  --mixed_precision=no -m lerobot_policy_vla_ttt.training.train_episode_parallel \
  --resume "outputs/aqmem-libero-${NPROC}gpu-smoke/step-00000002" \
  --dataset-root "$DATASET" --output "outputs/aqmem-libero-${NPROC}gpu-continued"
```

全局 slot 决定 episode 和随机种子，rank 只负责分片。每卡保留完整模型和 optimizer，
显式 SUM 归约慢参数梯度；不平均 fast state，不采用 FSDP/DeepSpeed。
多卡不保证显存按卡数缩减，也不保证线性加速。policy 自身负责 CUDA BF16 autocast。
Exact resume 必须保持 world size、两种 mode、batch、其他 recipe 字段及资源身份一致。
旧 recipe 缺少 mode 字段时只补 `alternating` / `same_task`，不改旧采样行为。

`metrics.jsonl` 记录 `phase=joint`、`phase_updates.joint`、全局 task 数/计数、
带类别的 optimizer LR/decay、有效监督数和 update_seconds，便于核对训练协议。
CPU/Gloo 测试覆盖真实 tiny AQMem、变长跨任务 episode、单/双进程梯度与更新一致性、
同 world size 精确恢复；它不能替代真实权重、CUDA/NCCL、BF16 和吞吐验证。
