# RoboMME observation-memory 训练适配与检查（2026-10-13）

实现位于 `.worktrees/observation-memory`，分支 `feat/observation-memory`。
初轮完成 CPU 与数据检查；随后经用户授权，在 baseline 评测并行运行时完成 GPU 短测（见下文）。未修改 master 或停止其他进程。

## 训练协议

复用 `training/train_episode_parallel.py` 与现有 Accelerate 外层梯度归约、交替优化、TBPTT 和 checkpoint 实现。增加 `--benchmark robomme` 与独立数据视图 `training/episode_parallel/robomme.py`，原 baseline 的 execution-only 数据视图保持不变。

- 从 **RoboMME Evo1-GROOT Stage1 5000-step** checkpoint 初始化；继承 VLM、action expert、双相机顺序、处理器及 execution-only 归一化统计。不是从 LIBERO 或 Stage2 80k 权重初始化。
- H=20、E=10、动作维度 8；每 5 个执行环境步写入一次，每 10 步有一次动作监督。去噪不写入记忆。
- 保留 demo 前缀的完整时间范围，默认每 5 帧采样一次视觉观测（不是逐帧全部输入）。`--demo-stride` 可显式改变该间隔；这是本平台设置。
- demo 只写记忆，不提供 action/action_mask/action_is_pad，不产生动作 loss、不采样动作噪声。proprioception 仍作为框架要求的观测字段存在，但不进入 VL memory interface。
- 执行段从 `exec_start_idx` 重新计算 E/2 和 E 的时钟；demo→执行不 reset，episode→episode 重新初始化 fast state。末尾监督按真实 episode 边界 mask。
- TBPTT=8 个观测事件，包含 demo 事件。跨段 detach 保留记忆数值，但执行段的梯度不会穿过已截断的早期 demo。长 demo 的纯写入段没有 loss；这不是完整 BPTT。
- 全局 batch=4 条 episode，每卡 microbatch=1，2000 次外层更新，每 500 次交替 memory-only/policy-only。policy 阶段全量训练 VLM+action expert，不使用 LoRA。其余 LR、优化器参数继承已确定的 LIBERO recipe。
- 沿用“均匀选语言指令，再均匀选该指令下 episode”的分层采样。本地数据有 **116 条语言指令**，不是 16 类任务均匀采样，也不是所有 episode 均匀采样。这里不声称已经对齐 baseline 80k 的训练量。
- demo stride、边界协议进入 checkpoint identity。恢复可自动识别 RoboMME；改变 stride 会拒绝 exact resume。已有 LIBERO checkpoint 的 identity 结构保持兼容。

## 启动方式

在本 worktree 中运行，显式设置 `PYTHONPATH=src`，避免共享 venv 的 editable 安装导入 master 源码。下面先做无权重、无 GPU 的配置检查：

```bash
cd /home/hello/lizk/vlas/vla-ttt/.worktrees/observation-memory
export PYTHONPATH="$PWD/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE="$PWD/artifacts/hf-datasets-cache"
/home/hello/lizk/vlas/vla-ttt/.venv/bin/python \
  -m lerobot_policy_vla_ttt.training.train_episode_parallel \
  --benchmark robomme --architecture t2mem \
  --checkpoint /home/hello/lizk/vlas/vla-ttt/outputs/robomme-evo1-groot-scratch-seed1000-stage1/checkpoints/005000/pretrained_model \
  --dataset-root /home/hello/lizk/datasets/robomme \
  --recipe configs/training/robomme-t2mem-episode.json \
  --output outputs/robomme-t2mem-smoke --dry-run
```

确认 GPU 余量并安排测试后，把 `--dry-run` 换成 `--stop-after 2` 可做真实权重短测；它保留完整 2000-step schedule，前两次均为 memory 阶段。要验证两种阶段的实际 GPU 负载，另建短测 recipe 设置 `phase_length=1`，不能把该短测产物当正式训练恢复点。

正式训练去掉 `--stop-after` 并使用新输出目录。多卡仍可用 Accelerate/torchrun 启动同一模块，全局 batch 必须整除 world size。迁移服务器时确认初始化 checkpoint 的本地 VLM 路径有效。恢复使用 `--resume .../step-XXXXXXXX` 替代 `--checkpoint`，使用新输出目录。

## 本轮证据与边界

本地检查产物在 `artifacts/robomme-memory-preflight-20261013/`：

- `resolved.json`：真实 RoboMME Stage1 checkpoint 的 dry-run 解析结果。
- `data-report.json`：检查 1600 条 episode，768897 帧，其中 demo 292040、执行 476857；1292 个数值 parquet、48 个引用视频文件；所有数值和 demo 边界检查通过。
- 选取 episode 0、800、1599，各抽查开头、demo 末帧、执行首帧、episode 尾帧，共 12 个双相机样本；图像为 3×256×256，动作 20×8，尾帧有 19 个 padding。另完整遍历 episode 0 的 20 个采样事件并检查监督标志。
- 当前配置对全体 episode 共形成 48376 个监督决策点；实际训练有放回抽样，不代表一轮训练遍历了全部数据。
- CPU tiny 真实模型（替代大 VLM 资产）完成 memory/policy 两阶段更新与断点恢复；测试覆盖 demo 不监督、执行时钟重启、尾部 mask、非法边界和改变 demo stride 的恢复拒绝。
- 初轮只读 GPU 查询显示占用 7359 MiB、利用率 50%，当时未启动 CUDA 训练；随后完成的真实权重 GPU 验证见下文。

RoboMME 记忆版闭环评测的 demo 注入尚未在本次接入；不能直接把 memoryless baseline 的评测流程视为记忆模型评测已完成。

### 回归测试结果

完整 `pytest -q`：**218 passed、7 skipped**，70.33 秒；GPU 测试禁用，含 CPU/Gloo 分布式验证。18 条 warning 为上游 `torch.jit.script` 弃用提示。首次运行有 4 个 CLI 查找失败（`lerobot-rollout` 和三个 `vla-ttt-*` 命令），补上已有 venv 的 `PATH` 后完整重跑通过；未安装或修改依赖。修改文件的 Ruff 检查与 `git diff --check` 通过。


## 真实权重 GPU 短测（同日追加）

用户确认剩余显存可用于并行短测后，使用 NVIDIA RTX PRO 5000 72GB Blackwell 运行了生产训练入口。初始化为上文 RoboMME Stage1 5000-step 权重；采用真实数据、双相机、BF16 autocast、非重入 checkpointing、Accelerate 单进程。没有缩短 episode 或缩小模型。

全局 batch=4、microbatch=1、H=20、E=10、demo stride=5、TBPTT=8 均保持不变。仅将短测 `phase_length` 改为 1，使用 `--stop-after 2`，让 memory-only 和 policy-only 各更新一次；总 schedule 仍为 2000，warmup=50。测试设置本进程 PyTorch 显存上限为 GPU 总量的 70%（约 49.8 GiB）。这是分配器上限，不包含全部 CUDA 驱动开销，也不保证其他任务的速度不受共享 GPU 影响。

| 阶段 | episode 数 | 观测事件 | 监督决策 | 更新耗时 | 峰值 allocated | 峰值 reserved | loss | 裁剪前梯度范数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| memory-only | 4 | 787 | 91 | 99.51 s | 4.97 GiB | 5.44 GiB | 1.24770 | 56.63 |
| policy-only | 4 | 301 | 151 | 99.71 s | 13.39 GiB | 14.86 GiB | 1.24587 | 37.46 |

以上显存是训练进程的 PyTorch 统计，不包括其他评测进程。两次更新合计 199.22 秒，不含模型初始化和最终 checkpoint 保存；因抽到的任务和轨迹长度不同，不能直接按此估算完整 2000-step 时长。memory batch 包含大量 demo 事件；本次 policy batch 的事件数接近两倍监督决策数，未覆盖长 demo 下的 policy-phase 最坏情况。

验收结果：进程退出码 0，无 OOM；两阶段 loss 和梯度有限，optimizer 均完成一次更新。最终 `step-00000002/COMPLETE` 存在，保存的 optimizer 计数为 memory=1、policy=1，分别有 176 和 732 个参数的 optimizer state，模型保存了 937 个 tensor。检查了保存产物，本轮没有另做 CUDA resume。训练退出后只读查询显存回到 7357 MiB，原有 GPU 任务仍有利用率。

产物在 `artifacts/robomme-memory-gpu-20261013/`：

- `probe.py`：仅增加进度、耗时和显存统计的包装入口，调用原生产 CLI 和训练循环。
- `recipe.json`：阶段长度为 1 的专用短测配置。
- `run.log`、`run/metrics.jsonl`、`summary.json`：原始日志与汇总。
- `run/step-00000002/`：测试 checkpoint。

结论：**当前 batch=4、TBPTT=8 配置已通过真实权重、真实 RoboMME 完整 episode 的单卡两阶段训练短测，无需因本次显存结果缩小 batch 或 TBPTT。**这不是全数据最长轨迹、多卡 NCCL 或长期数值稳定性的验收。正式训练应重新从 Stage1 初始化，并使用 `configs/training/robomme-t2mem-episode.json` 的 `phase_length=500`；不要从本次阶段长度为 1 的测试 checkpoint 恢复正式训练。

## 合入 master 时的可迁移性兼容

集成 master 的机器配置、VLM 路径覆盖、内容哈希 identity v2 及运行目录日志管理。新训练生成可迁移 checkpoint；原有路径绑定 checkpoint 继续沿用原恢复协议，不会自动升级成可迁移身份。

机器配置中的 `assets.libero_dataset` 和 `assets.initial_checkpoint` 专用于 LIBERO。RoboMME 使用 `assets.robomme_dataset` 和 `assets.robomme_initial_checkpoint`；显式 `--dataset-root` 和 `--checkpoint` 仍优先于机器配置。VLM 路径通过共享的 `assets.vlm` 或 `--vlm-path` 指定。下载与配置步骤见 [跨机器训练文档](../portable-training.md)。例如在目标机器仓库根目录：

```bash
PYTHONPATH=src .venv/bin/python -m lerobot_policy_vla_ttt.training.train_episode_parallel \
  --machine configs/machine.local.toml --run-name robomme-t2mem \
  --benchmark robomme --architecture t2mem \
  --recipe configs/training/robomme-t2mem-episode.json --dry-run
```

将路径替换为目标机器实际位置。`--dry-run` 仅检查元数据，正式启动才验证数据、VLM 和初始化资源的内容哈希。代码中的训练参数不再依赖本机 `/home/hello/...` 路径；上文绝对路径仅记录本次实验资产位置。

合并后的完整 CPU/Gloo 回归为 **231 passed、7 skipped**（71.90 秒），Ruff 与 diff 空白检查通过。LIBERO、RoboMME 均测试了数据/VLM/checkpoint 路径搬迁后的精确恢复与资源内容变更拒绝；已有 GPU 短测 checkpoint 的旧版 identity 也通过了恢复 dry-run。独立代码审查未发现 critical/important 问题。合并验证日志为 `artifacts/robomme-memory-preflight-20261013/merge-pytest.log`。
