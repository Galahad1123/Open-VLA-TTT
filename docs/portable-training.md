# 跨机器训练快速开始

适用于 LIBERO 的 T²Mem / AQMem / MAC / MAG / MAL episode-parallel 训练。
每台机器只改 `configs/machine.local.toml` 中的路径，所有机器使用同一代码版本和 recipe。

## 1. 安装并准备资源

检出团队约定的 Git commit/tag，在仓库根目录执行：

```bash
uv sync --frozen
cp configs/machine.example.toml configs/machine.local.toml
```

主环境使用 Python 3.12 或 3.13；依赖按 `uv.lock` 安装，不复制其他机器的 `.venv`。
当前主锁文件包含 Torch 2.11 / CUDA 13 相关运行库，目标 GPU 和驱动需要兼容。
可以先检查 CUDA，再进行下方短程训练验收：

```bash
uv run --frozen python -c 'import torch; print(torch.__version__, torch.version.cuda); print(torch.ones(1, device="cuda") + 1)'
```

提前复制或下载以下完整资源，资源本身不进 Git：

| 配置字段 | 内容 |
| --- | --- |
| `assets.initial_checkpoint` | 共同 Stage1 5k 的 `pretrained_model/`，包含权重、config、全部 processor/归一化文件及 `train_config.json` |
| `assets.vlm` | 完整 `OpenGVLab/InternVL3-1B-hf` 快照，revision `014c0583a0d4bedf29fbe2dbff4f865eb998e171`，包括 tokenizer、配置和权重 |
| `assets.libero_dataset` | 完整 `HuggingFaceVLA/libero`，revision `86958911c0f959db2bbbdb107eb3e17c5f9c798e`，包含 `meta/` 和全部数据分片 |
| `storage.runs_root` | 可写的训练产物根目录 |
| `storage.hf_home` | 本机 Hugging Face 缓存目录 |

模板全部使用绝对路径示例，替换为目标机器的实际路径。
复制 HF snapshot 时必须保留可用的 blobs 链接，或者复制链接所指的真实文件。
数据集不能使用早期 smoke 的部分下载；资源目录在训练期间保持只读，不在其中放日志。

## 2. 检查配置

```bash
uv run --frozen python -m lerobot_policy_vla_ttt.training.train_episode_parallel \
  --machine configs/machine.local.toml --run-name t2mem-seed1000-check \
  --architecture t2mem --recipe configs/ttt/t2mem/libero/recipe.json \
  --dry-run
```

这一步只读取配置和小型元数据，不加载模型、初始化 CUDA、创建运行目录或读取全部权重/数据。
输出 `asset_validation=metadata_only`，不代表资源完整或 GPU 训练已经通过。
`recipe.json` 显式保存当前约定的 TBPTT=8 默认值。
`recipe-72gb.json` 是已记录资源验收的 TBPTT=64 候选，需主动选择，不替代默认协议。
跨机器对照应显式使用同一 recipe，不根据显存自动改变实验协议。

## 3. 短程检查与正式训练

单 GPU 两步检查（保留完整训练预算与 scheduler）：

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run --frozen python -m lerobot_policy_vla_ttt.training.train_episode_parallel \
  --machine configs/machine.local.toml --run-name t2mem-seed1000-smoke \
  --architecture t2mem --recipe configs/ttt/t2mem/libero/recipe.json \
  --stop-after 2
```

正式训练使用新的 `--run-name` 并去掉 `--stop-after 2`。
两步检查不覆盖第 500 步的 Memory→Policy 切换；新硬件仍需检查双方 optimizer 状态常驻后的显存。

两张 GPU 用相同参数启动：

```bash
CUDA_VISIBLE_DEVICES=0,1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run --frozen python -m accelerate.commands.launch \
  --multi_gpu --num_processes=2 --mixed_precision=no \
  -m lerobot_policy_vla_ttt.training.train_episode_parallel \
  --machine configs/machine.local.toml --run-name t2mem-seed1000-two-gpus \
  --architecture t2mem --recipe configs/ttt/t2mem/libero/recipe.json
```

默认全局 batch=4，可用 1/2/4 个进程；每卡 microbatch=1，不自动扩大 batch 或 LR。
不要启用 FSDP/DeepSpeed；policy 自行使用 BF16 AMP。多节点运行时资源和输出目录必须对各 rank 可见，
本指南的多卡命令针对单机，不能据此视为完成多节点验收。

## 4. 产物位置与恢复

```text
<storage.runs_root>/<run-name>/
  run.json                  # recipe、资源身份、实际路径、代码和运行环境
  metrics.jsonl             # 主进程逐步指标
  logs/rank-0.log           # 各进程的控制台输出与运行异常
  logs/rank-1.log
  step-00000250/
    COMPLETE
    training.pt
    pretrained_model/
```

目录已存在就拒绝启动；每次启动（包括恢复）使用新的名称。进入运行阶段后才创建日志，
参数解析/资源校验错误直接报告到终端。日志同时显示在控制台，不需要额外 `tee`。

恢复时指定完整 `step-*` 目录，recipe 默认从 checkpoint 恢复：

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run --frozen python -m lerobot_policy_vla_ttt.training.train_episode_parallel \
  --machine configs/machine.local.toml --run-name t2mem-seed1000-continued \
  --resume /srv/experiments/vla-ttt/t2mem-seed1000-smoke/step-00000002
```

恢复必须保持 world size、recipe、结构和数据/增强语义相同。
本版新建运行按数据集、VLM 的完整内容 SHA256 校验，允许相同资源换目录；初始化来源也记录内容哈希。
正式启动会流式读取完整资源进行校验，每个 rank 都会校验，不需要 GPU，但可能耗时较长。
忽略隐藏缓存文件；不要往资源目录添加无关文件。完整哈希用于拒绝恢复时的资源变化，
首次运行仍需要操作者按上表准备正确版本，它不替代发布方的资源清单。

旧版 checkpoint 没有完整内容身份，继续使用原来的严格路径校验；不会自动放宽校验迁移。
若旧路径无法保留，可用 `--checkpoint .../pretrained_model` 开始新阶段，明确属于 warm start，
不恢复 optimizer/进度。新 checkpoint 的异构 GPU 恢复也不保证逐位数值一致。

## 配置规则与评测

- CLI 显式 `--checkpoint`、`--dataset-root`、`--vlm-path` 优先于 TOML；`--resume` 不使用默认初始化 checkpoint。
- `--output /absolute/path` 可代替 `--run-name`，两者不能同时指定。
- TOML 支持 `~`；相对路径相对于 TOML 文件目录，CLI 相对路径相对于当前目录。
- 未知字段、无效路径会报错；本机配置被 Git 忽略，recipe 和示例模板提交 Git。
- `hf_home` 在必要时通过一次进程重新启动于库导入前生效；不移动已有缓存。
- 当前机器配置用于新 episode-parallel 入口和 policy server，旧 baseline/序列训练入口保持原 CLI。

评测服务端也接受相同 VLM 配置，checkpoint 必须显式选择：

```bash
uv run --frozen vla-ttt-serve --machine configs/machine.local.toml \
  --checkpoint /srv/experiments/vla-ttt/t2mem-seed1000-continued/step-00002000/pretrained_model \
  --implementation t2mem --benchmark libero
```

仿真端另按 [baseline 指南](baselines.md) 安装 LIBERO 环境并启动 client。
训练语义、验收范围见 [episode-parallel 指南](episode-parallel-training.md)。
