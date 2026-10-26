# RoboMME 六任务子集评测（5090）

在 `experiment/robomme-task-screen` 分支仓库根目录执行。模型服务使用训练环境，
仿真客户端使用独立 Python 3.11 环境。整个过程串行运行，既不重新训练，也不改变模型输入。

## 1. 安装与检查

```bash
git pull --ff-only
uv sync --frozen --project environments/robomme
ffmpeg -hide_banner -encoders | grep libx264
```

系统需要 `ffmpeg` 且包含 `libx264` 编码器。Ubuntu 未安装时可由管理员执行
`sudo apt-get install ffmpeg`。SAPIEN/Vulkan 渲染还依赖正确安装的 NVIDIA 驱动；
CUDA 训练成功不等于渲染验收成功。已经装好仿真环境时不需要重复同步。
主训练环境通过镜像安装的用户继续使用 `--no-sync`，无需改动锁文件。

## 2. 终端一：启动模型

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run --no-sync python -m lerobot_policy_vla_ttt.evaluation.server \
  --machine configs/machine.local.toml \
  --checkpoint outputs/robomme-six-fresh-seed1000-stage2/checkpoints/010000/pretrained_model \
  --implementation evo1_groot --benchmark robomme \
  --n-action-steps 10 --host 127.0.0.1 --port 8795
```

checkpoint 路径按本机 `storage.runs_root` 修改；VLM 路径使用本机 TOML。
本入口要求无记忆 evo1-groot；不会将演示视频写入模型，也不会传入特权子目标。
如果已经启动同一模型的服务，可直接复用。

## 3. 终端二：录制短测

先检查配置（不连接服务、不初始化仿真、不创建输出目录）：

```bash
uv run --no-sync --project environments/robomme python -m vla_ttt_eval.subset \
  --plan configs/evaluation/robomme-six-subsets.json \
  --split val --tasks PickXtimes --limit-per-task 1 \
  --max-steps 20 --output outputs/eval/six-10k-video-smoke --dry-run
```

去掉 `--dry-run` 即执行短测：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 \
uv run --no-sync --project environments/robomme python -m vla_ttt_eval.subset \
  --plan configs/evaluation/robomme-six-subsets.json \
  --server ws://127.0.0.1:8795 --split val \
  --tasks PickXtimes --limit-per-task 1 --seed 1000 --max-steps 20 \
  --output outputs/eval/six-10k-video-smoke
```

打开输出目录的 `report.html`，确认视频双相机画面正常。20 步超时不代表模型能力；
官方 wrapper 的 reset 计数偏移使实际预算为 21 步。

## 4. 六任务正式匹配子集

以下命令在一个客户端内依次执行六个任务，每任务最多取前 20 个符合条件的 val 场景：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 \
uv run --no-sync --project environments/robomme python -m vla_ttt_eval.subset \
  --plan configs/evaluation/robomme-six-subsets.json \
  --server ws://127.0.0.1:8795 --split val \
  --limit-per-task 20 --seed 1000 --max-steps 1300 \
  --output outputs/eval/six-10k-val-matched
```

与此前 PickXtimes/VideoUnmask 实验一致：按官方 episode 0–49 顺序检查场景，
最多取前 20 个符合条件的场景。若不足 20 个，报告实际分母，不重复补齐。
去掉 `--limit-per-task 20` 可评测全部符合条件的 val 场景，但应标注为另一评测范围。
`--tasks PickXtimes VideoUnmask` 可只运行指定任务。

训练集诊断用 plan 中全部 266 条固定训练轨迹，不设每任务数量上限：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 \
uv run --no-sync --project environments/robomme python -m vla_ttt_eval.subset \
  --plan configs/evaluation/robomme-six-subsets.json \
  --server ws://127.0.0.1:8795 --split train \
  --seed 1000 --max-steps 1300 \
  --output outputs/eval/six-10k-train-matched
```

上述命令依次运行，不并行。若要自动连续运行，可将 val、train 命令依次放入
以 `set -e` 开头的 shell 脚本；第一项失败则停止。

## 5. 子集语义与结果

| 任务 | reset 后、模型推理前的筛选规则 | 固定训练 episode 数 |
| --- | --- | ---: |
| PickXtimes | `num_repeats == 1` | 23 |
| SwingXtimes | `num_repeats == 1` | 23 |
| VideoUnmask | 当前难度配置 `pick == 1` | 75 |
| ButtonUnmask | 当前难度配置 `pick == 1` | 75 |
| PickHighlight | 当前难度配置 `pickup == 1` | 50 |
| BinFill | 三种颜色目标数量之和为 1 | 20 |

不是简单筛选 `easy`。训练 plan 来自 `multitask-six-stage2.json` 的 task_groups，
使用官方任务内 0–99 episode ID：原始数据连续 100 条任务分块的全局 ID 转为块内 ID。
启动时还检查训练场景是否满足规则，不匹配立即报错。val/test 不借用训练集 ID。
筛选依据是场景本身，不依据成功率或模型行为。每回合 policy seed 为 1000+episode；
环境 seed 由官方 split metadata 决定。阶段诊断仅写入结果，不输入策略。

- `run.json`：实际配置、候选 episode、plan 内容与 SHA256、服务端 checkpoint 身份。
- `results.jsonl`：已完成回合、被排除场景及基础设施错误；逐条落盘。
- `summary.json`：是否完整完成、各任务和总体成功率、任务宏平均、实际分母及排除数。
- `report.html`：汇总、执行视频、demo 视频与逐回合阶段进展。
- `videos/<task>/episodeNNN.mp4`：双相机执行视频，H264、10 FPS；旁边有 JSON。
- `episodeNNN-demo.mp4`：存在演示前缀时，每 5 帧取一帧、2 FPS 保存。
  仅用于人工诊断，不代表 baseline 看到了 demo。

总成功率按实际回合加权，宏平均按有已完成回合的任务等权；未完成报告不能与完整报告直接比较。
阶段进展仅表示官方子目标状态变化，不等于独立的动作质量或目标选择评估。
报告视频使用相对路径，复制整个输出目录即可在其他机器查看。

可通过 `--selection all --split val` 评测完整 val 场景；去掉 `--limit-per-task` 才是每任务全部 50 回合。
此时保留 `matches_subset` 标记，但不要把完整任务与匹配子集结果混合比较。
默认仅评测固定最终 checkpoint；不要在 test 上挑选 checkpoint。

输出目录必须不存在，不覆盖、不自动续跑。出错会停止并保留日志及部分视频，
不将基础设施错误计为策略失败。需要重跑时使用新的输出目录。

## 验证范围

CPU 测试覆盖六任务筛选、非法 plan、实际 H264 编码、reset 次数、策略输入隔离、
排除场景不触发推理和结果分母。沿用已验证的生产 rollout 与原实验录制协议。
本次整理未启动真实 GPU 仿真；目标机器请先运行上面的录制短测。
