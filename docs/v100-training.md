# V100 32GB 单机训练

本分支基于 master `08fd511`，支持 evo1_groot、MAC、MAG、MAL、AQMem、T2Mem 的 episode trainer。原来的 BF16 默认行为保留；V100 启动时显式传入 `--precision fp16` 或 `--precision fp32`。这是精度适配，不改变训练 recipe。

## 独立环境

目标平台：Linux x86_64、GLIBC >= 2.28、Python 3.12；已知服务器驱动 570.195.03 可运行 cu126。`nvidia-smi` 的 CUDA 12.8 不要求安装 cu128。PyTorch 2.11 的 cu128 构建不包含 Volta，必须使用这里锁定的 cu126。

在本分支工作目录执行：

```bash
uv sync --project environments/v100 --frozen
uv run --project environments/v100 --frozen python -c 'import torch; print(torch.__version__, torch.version.cuda)'
```

预期 `2.11.0+cu126` / `12.6`。torchvision 固定 `0.26.0+cu126`。环境位于 `environments/v100/.venv`，锁位于同目录，根目录的 `uv.lock` 不变。不要在目标机直接执行根目录的 `uv sync`，也不要复用开发机 `.venv`。LeRobot 保持 0.6.1；与 master 共有的非 CUDA 依赖版本也对齐根锁文件，避免无关升级，其余依赖由独立锁文件固定。安装不需要升级驱动，使用 wheel 时也不要求另装系统 CUDA Toolkit。

参考：[PyTorch 版本安装列表](https://pytorch.org/get-started/previous-versions/)、[Volta 构建变更](https://dev-discuss.pytorch.org/t/dropping-volta-support-from-cuda-12-8-binaries-for-release-2-11/3290)、[CUDA 12.6 驱动要求](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-toolkit-release-notes/index.html)。

## 先验证目标机器

以下命令会使用 GPU，只在新服务器上执行。先检查拓扑，再做小矩阵 FP16 backward 和各卡梯度 SUM；这个检查不能替代真实模型验收。

```bash
nvidia-smi topo -m
uv run --project environments/v100 --frozen python scripts/check_v100.py --cuda
uv run --project environments/v100 --frozen python -m torch.distributed.run \
  --standalone --nproc-per-node=8 scripts/check_v100.py --cuda
```

随后按单卡 FP32 → 单卡 FP16 → 双卡 FP16 → 八卡 FP16 逐步验证。对照同一有效 batch 的 loss、梯度、成功更新次数及 checkpoint 恢复；FP16 与 FP32 允许数值误差，不要求 bitwise 相等。涵盖最长轨迹、交替训练的两个阶段和 joint 模式峰值显存。

## 训练命令

按 `docs/episode-parallel-training.md` 配置机器路径、初始化 checkpoint、数据与 augmentation。示例使用现有 LIBERO AQMem recipe（global batch 8），每卡一个 episode；路径中的 `machine.toml` 需换成该服务器自己的文件。

```bash
uv run --project environments/v100 --frozen python -m torch.distributed.run \
  --standalone --nproc-per-node=8 \
  -m lerobot_policy_vla_ttt.training.train_episode_parallel \
  --machine /path/to/machine.toml --run-name aqmem-v100-smoke \
  --architecture aqmem --precision fp16 \
  --recipe configs/ttt/aqmem/libero/recipe.json --stop-after 2
```

正式训练使用新的 run-name 并去掉 `--stop-after`，或用 `--resume` 继续 smoke checkpoint。RoboMME 使用 `--benchmark robomme` 及对应 machine/recipe 配置。global batch 必须整除进程数；原来 batch 4 的 recipe 使用四卡，不会自动改成 batch 8。可以在另外四卡跑独立实验。32GB × 8 不会合并成单模型可用的 256GB：每卡保留完整参数与优化器状态。

## 精度与恢复语义

- 参数、AdamW 状态、外层梯度 SUM、TTT fast weights 和内层 `autograd.grad` 更新保持 FP32。只在主模型前向中使用 FP16 autocast，动作损失归约保持 FP32。禁止把整个 policy `.half()`。
- 一个全局 batch 的所有 episode/TBPTT 段使用相同 loss scale，只缩放外层 loss。梯度先 unscale，再做跨卡 SUM 和 clipping。Accelerate 仍使用 `mixed_precision="no"`，不要额外启用其 FP16 scaler。
- 初始 loss scale 65536；连续 2000 次成功更新后翻倍（上限 2^24）。任一卡梯度溢出或 SUM 后溢出时，所有卡共同减半并重新计算同一批数据。最多连续 16 次溢出后报错，建议 FP32 排查。
- 溢出不推进 optimizer、LR schedule、phase、采样位置或数据计数。重试丢弃 episode fast state，并恢复随机数。前向 loss 非有限直接报错：降低 loss scale 不能修复前向溢出。
- `training.pt` 的 `control.precision` 保存缩放状态；metrics 记录 `loss_scale`（下一次更新使用值）和累计 `skipped_updates`。控制台记录每次溢出及重试。
- `--resume` 默认继承 checkpoint 精度，拒绝更改精度。旧 BF16/FP32 checkpoint 没有 scaler 状态仍可恢复；FP16 checkpoint 缺少此状态会拒绝恢复。要将 BF16 权重用于 FP16 新实验，使用 `--checkpoint .../pretrained_model --precision fp16`，这会新建优化器和训练进度。

本地 CPU/Gloo 测试覆盖控制流与数值缩放，不证明真实 V100 的 FP16 前向稳定性、显存峰值或八卡加速比。真实权重若在 FP16 前向出现溢出，应先用 `--precision fp32` 对照再定位敏感模块。

## 随机 chunk 与指定任务子集

RoboMME evo1-groot 随机 chunk 多卡训练使用独立入口 `training.train_task_subset`。
它支持原始 episode ID 子集、全局 batch 换算及 `--precision fp16`，并使用
Accelerate 自带的 GradScaler；不要套用上面 episode trainer 的自定义 scaler。
完整 recipe、四卡短测/正式训练命令及验证边界见
[随机 chunk 子集训练说明](robomme-subset-training.md)。
