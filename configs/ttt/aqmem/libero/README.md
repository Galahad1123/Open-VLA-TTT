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

## 机器配置与单 GPU 命令

按 [跨机器训练指南](../../../../docs/portable-training.md) 安装锁定环境，在仓库根目录运行。
路径只写入被 Git 忽略的 `configs/machine.local.toml`，recipe/memory JSON 不保存机器路径。
在已有 TOML 中修改对应字段，勿重复创建段；示例绝对路径须替换为当前机器资源：

```toml
[assets]
initial_checkpoint = "/srv/models/libero-stage2-20k/pretrained_model"
vlm = "/srv/models/InternVL3-1B-hf"
libero_dataset = "/srv/datasets/HuggingFaceVLA-libero"

[storage]
runs_root = "/srv/experiments/vla-ttt"
hf_home = "/srv/cache/huggingface"
```

`initial_checkpoint` 沿用 master 的 LIBERO 字段，不增加 AQMem 专属路径字段。
必须复制完整 Stage2 `020000/pretrained_model/`，包括模型、config、processor/归一化文件
和 `train_config.json`。选择 30k 时改为对应的 `030000/pretrained_model/` 并使用新 run name。
T²Mem 的 Stage1 初始化实验仍使用其原检查点；若共用机器配置，CLI `--checkpoint` 可显式
覆盖默认值。RoboMME 继续使用独立的 `robomme_initial_checkpoint` / `robomme_dataset`。

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# 配置检查：不加载权重，不创建输出，不使用 GPU。
CUDA_VISIBLE_DEVICES='' uv run --frozen python \
  -m lerobot_policy_vla_ttt.training.train_episode_parallel \
  --machine configs/machine.local.toml --run-name aqmem-libero-stage2-20k-check \
  --architecture aqmem --recipe configs/ttt/aqmem/libero/recipe.json \
  --memory-config configs/ttt/aqmem/libero/memory.json --dry-run

# 单 GPU：先运行两步，保留完整 2000 步 scheduler，建立 AdamW 状态。
CUDA_VISIBLE_DEVICES=0 uv run --frozen python \
  -m lerobot_policy_vla_ttt.training.train_episode_parallel \
  --machine configs/machine.local.toml --run-name aqmem-libero-stage2-20k-smoke \
  --architecture aqmem --recipe configs/ttt/aqmem/libero/recipe.json \
  --memory-config configs/ttt/aqmem/libero/memory.json --stop-after 2

# 恢复时使用完整 step-* 目录及新 run name，不重复指定初始化 memory 配置。
CUDA_VISIBLE_DEVICES=0 uv run --frozen python \
  -m lerobot_policy_vla_ttt.training.train_episode_parallel \
  --machine configs/machine.local.toml --run-name aqmem-libero-stage2-20k-continued \
  --resume /srv/experiments/vla-ttt/aqmem-libero-stage2-20k-smoke/step-00000002
```

上面的恢复路径替换为实际 `storage.runs_root` 下的检查点。
正式从 Stage2 初始化训练时去掉 `--stop-after 2`，并使用新 run name。
按指南通过镜像安装的环境用 `uv run --no-sync` 替换 `uv run --frozen`，避免重新同步依赖。
每次启动（包括恢复）都使用新输出目录，产物遵循 `run.json`、`metrics.jsonl`、
`logs/rank-N.log`、`step-XXXXXXXX/` 约定。`hf_home` 在库导入前通过重新启动生效。

迁移时复制完整资源和完整 `step-*` 目录，只修改新机器 TOML。
恢复检查数据和 VLM 内容 SHA256，允许路径变化，拒绝内容变化；旧 checkpoint
没有内容身份时仍执行旧路径约束。`--resume` 忽略 TOML 的默认初始化 checkpoint。

两步只是启动检查，不足以覆盖最长轨迹或可靠估算训练时长；新硬件应运行代表性长短轨迹，
测量 AdamW 状态建立后的峰值显存和吞吐。

## 多 GPU 启动（待 CUDA/NCCL 验收）

沿用同一机器 TOML，选择以下一组配置：

| 卡数 | CUDA_VISIBLE_DEVICES | NPROC | 每 rank episode/更新 |
|---|---|---|---|
| 2 | 0,1 | 2 | 4 |
| 4 | 0,1,2,3 | 4 | 2 |
| 8 | 0,1,2,3,4,5,6,7 | 8 | 1 |

```bash
export CUDA_VISIBLE_DEVICES=0,1
NPROC=2  # 4/8 卡时同时调整可见 GPU 列表
uv run --frozen python -m accelerate.commands.launch --multi_gpu --num_processes="$NPROC" \
  --mixed_precision=no -m lerobot_policy_vla_ttt.training.train_episode_parallel \
  --machine configs/machine.local.toml --run-name "aqmem-libero-${NPROC}gpu-smoke" \
  --architecture aqmem --recipe configs/ttt/aqmem/libero/recipe.json \
  --memory-config configs/ttt/aqmem/libero/memory.json --stop-after 2

uv run --frozen python -m accelerate.commands.launch --multi_gpu --num_processes="$NPROC" \
  --mixed_precision=no -m lerobot_policy_vla_ttt.training.train_episode_parallel \
  --machine configs/machine.local.toml --run-name "aqmem-libero-${NPROC}gpu-continued" \
  --resume "/srv/experiments/vla-ttt/aqmem-libero-${NPROC}gpu-smoke/step-00000002"
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

## 2026-10-14 单 GPU 短测记录

实际 Stage2 20k 权重、原 recipe、全量 VLM、64 条 episode / 8 次联合更新通过，
检查点正常保存；无 OOM、NaN 或非有限梯度。没有启动完整 2000 步训练或 rollout 评测。

| 指标 | RTX PRO 5000 72GB 实测 |
|---|---|
| PyTorch peak allocated / reserved | 19.46 / 21.19 GiB |
| 整卡占用采样峰值（2 秒间隔） | 22.17 GiB |
| 短测总耗时（含启动与保存） | 988.9 秒 |
| 第 2–8 步平均耗时 | 117.8 秒/更新 |
| 2000 步预计耗时（采样长度校正） | 约 63 小时 |

以优化器状态建立后的第 2–8 步吞吐和 seed1000 的完整采样工作量估计，
并计入启动与 8 次保存开销；实测速率范围对应约 62–64 小时，不是置信区间或保证。
最长实测 episode 为 450 帧，数据集最长为 505 帧；不是最坏情况显存保证。
该结果不外推为 RTX 5090、多卡 CUDA/NCCL、长期稳定性或任务成功率验收。
原始 JSON、采样清单、探测脚本和日志保存在验证机器的
`artifacts/aqmem-libero-gpu-20261014/`，该目录不随 Git 分发。
