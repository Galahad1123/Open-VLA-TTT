# RoboMME 六任务动作能力训练

本实验使用无记忆的 evo1-groot，合并六个简化任务，按任务均衡采样随机动作
chunk。不开启数据增强，不使用 AQMEM/T2MEM，也不把演示视频或特权子目标输入模型。
旧版单任务 recipe 仍然保留。

## 数据与训练预算

| 任务筛选条件 | 轨迹数 | 执行帧数 |
| --- | ---: | ---: |
| PickXtimes：单次抓放 | 23 | 6995 |
| VideoUnmask：单目标容器 | 75 | 8442 |
| PickHighlight：单目标方块 | 50 | 11438 |
| ButtonUnmask：单目标容器 | 75 | 17170 |
| BinFill：放入一个物体 | 20 | 5861 |
| SwingXtimes：单次左右往返 | 23 | 7730 |
| 合计 | 266 | 57636 |

recipe 中保存原始 episode ID。筛选依据是执行帧子目标标注，而不是只看 easy 标签。
这些标注仅用于筛选轨迹，不会作为模型输入。部分任务仍有短暂提示或隐藏目标，
因此评测需要分别记录动作执行、目标选择和完整任务成功率。

`task_groups` 表示六个任务类别，不是不同颜色或目标对应的语言指令。
每个 batch 先保证每类都有样本，再轮换分配额外名额，并打乱 batch 内顺序。
各任务内部对执行帧起点均匀、有放回抽样：batch=12 时每类两个 chunk；
batch=8 时每类一个 chunk，另外两个名额轮换。固定 seed 可复现采样序列。
保留原始轨迹的动作查询边界，以及完整训练集执行帧的归一化统计。
旧 recipe 默认仍使用 `random_frames`，不会自动改变旧实验的采样行为。

| 配置 | Stage1 | Stage2 |
| --- | ---: | ---: |
| 更新步数 | 750 | 10000 |
| batch | 12 | 8 |
| VLM | 冻结 | 更新 |
| 动作模块 | 更新 | 更新 |
| 学习率 | 1e-5 | 1e-5 |
| warmup 步数 | 100 | 200 |
| 保存间隔 | 750 | 1000 |

两阶段均使用 BF16 autocast、FP32 VLM 参数存储、seed=1000、两个数据加载进程，
chunk 长度为 20、每次执行 10 个动作。解冻 VLM 时启用梯度检查点。
每阶段新建优化器和调度器，学习率按 cosine 衰减到零。
Stage2 总共采样 80000 个 chunk，每类约 13333 个；由于各类帧数不同，
其重复采样强度也不同。训练预算不是成功率或收敛的保证。

## 初始化约定

机器配置 `configs/machine.local.toml` 提供数据集、VLM、输出根目录和旧 RoboMME
checkpoint 的本机路径。

**Stage1 的 `--fresh` 只读取旧 checkpoint 的模型配置，不加载旧策略权重或处理器。**
通用预训练 InternVL 从本地 VLM 资源加载，动作相关模块随机初始化；处理器使用
完整执行帧统计重新创建。这不等于把 VLM 也随机初始化。
`run.json` 会记录 `fresh_policy=true`、`checkpoint=null` 和独立的 `config_template`。

**Stage2 必须加载本次 Stage1 的最终权重，不使用 `--fresh`。**
两阶段只是权重衔接，不是优化器状态的精确续训。

## 推荐：一条命令串行训练两个阶段

在项目根目录、普通 5090 环境中执行，不使用 `environments/v100`：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false \
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run --frozen python scripts/train_robomme_six_tasks.py \
  --machine configs/machine.local.toml \
  --run-prefix robomme-six-fresh-seed1000
```

脚本自动从 TOML 解析 `storage.runs_root`，输出目录为：

```text
<RUNS_ROOT>/robomme-six-fresh-seed1000-stage1
<RUNS_ROOT>/robomme-six-fresh-seed1000-stage2
```

Stage1 成功退出、完成记录达到预定步数、最终检查点必要文件存在后，脚本才会
启动 Stage2，并自动传入 `checkpoints/000750/pretrained_model`。任何阶段报错都会
停止，不会继续启动后续训练。任一输出目录已经存在时，脚本拒绝运行，避免覆盖。
重新开始一个完整实验时请换一个 `--run-prefix`。

可先预览两阶段命令：

```bash
uv run --frozen python scripts/train_robomme_six_tasks.py \
  --machine configs/machine.local.toml \
  --run-prefix robomme-six-fresh-seed1000 --dry-run
```

这个 `--dry-run` 只解析配置并打印命令，不启动训练、创建输出目录或加载模型，
也不要求未来的 Stage1 检查点已经存在。串行脚本不会自动转入后台；建议在 tmux
会话中运行，终端会连续显示两个阶段的日志。各阶段保留原有 metrics、summary
和 checkpoint 输出。它也不会自动恢复中断训练。

## 可选：手动分别启动两个阶段

Stage1：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false \
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run --frozen python -m lerobot_policy_vla_ttt.training.train_task_subset \
  --machine configs/machine.local.toml \
  --recipe configs/training/robomme-screen/multitask-six-stage1.json \
  --run-name robomme-six-fresh-seed1000-stage1 --fresh
```

Stage1 完成后，将以下 `/ABSOLUTE/RUNS_ROOT` 替换为 TOML 中解析后的输出根目录，
再启动 Stage2。串行脚本已经完成 Stage1、但 Stage2 启动失败时，也可使用此命令；
如果 Stage2 已创建输出目录，请更换其 run-name，并注意这会重新开始 Stage2：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false \
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run --frozen python -m lerobot_policy_vla_ttt.training.train_task_subset \
  --machine configs/machine.local.toml \
  --checkpoint /ABSOLUTE/RUNS_ROOT/robomme-six-fresh-seed1000-stage1/checkpoints/000750/pretrained_model \
  --recipe configs/training/robomme-screen/multitask-six-stage2.json \
  --run-name robomme-six-fresh-seed1000-stage2
```

单阶段命令支持 `--dry-run`。也可使用独立 run-name 加 `--stop-after 60` 做性能短测，
但短测不会保存检查点，不能用作 Stage2 初始化。当前入口仅支持单 GPU，不使用 torchrun。

## 从旧补丁状态迁移到远程分支

远程分支为 `experiment/robomme-task-screen`。如果服务器仍然保留旧的未提交补丁，
先备份已跟踪和未跟踪文件，再保留旧分支并切换：

```bash
git stash push -u -m "robomme task screen before remote branch migration"
git branch -m experiment/robomme-task-screen-before-sync
git fetch origin
git switch --track -c experiment/robomme-task-screen origin/experiment/robomme-task-screen
git status
```

如果备份分支名已经存在，换一个新名字。不要把 stash 直接 pop 到新分支：旧代码
和配置已纳入版本管理。保留 stash，必要时检查其中额外的服务器专属修改。
被 Git 忽略的数据、模型、输出和 machine.local 配置不会被 `stash -u` 移除；启动前
仍应检查机器配置。

已经完成迁移的服务器，后续直接更新：

```bash
git pull --ff-only
```

本实验不会修改 master 或 `feat/v100-training`。

## 带视频的子集评测

训练完成后按[六任务子集评测指南](../robomme-subset-evaluation.md)安装独立仿真环境，
使用固定训练 episode 和验证场景条件进行串行评测，保存视频、阶段进展与 HTML 报告。
