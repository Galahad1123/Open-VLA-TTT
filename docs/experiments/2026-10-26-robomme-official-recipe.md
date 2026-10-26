# RoboMME 六任务：相对动作与官方式增强配方

本配方用于 `experiment/robomme-task-screen`，不修改旧版实验语义。
新配置为 `configs/training/robomme-screen/multitask-six-official-stage{1,2}.json`。
仍使用无记忆 evo1-groot、六任务266条轨迹、随机 chunk 和任务均衡采样，不含 RMBench。

## 配方

| 配置 | Stage1 | Stage2 |
| --- | ---: | ---: |
| 更新步数 | 3000 | 30000 |
| batch | 12 | 8 |
| VLM | 冻结 | 全量训练 |
| 峰值 LR | 1e-5 | 1e-5 |
| warmup | 300 | 1000 |
| cosine 最低 LR | 1e-6 | 1e-6 |
| 保存间隔 | 1000 | 5000 |

H=20、执行10步、seed=1000、BF16、无EMA。有效监督 masking 保持不变。
Stage1 加载通用 InternVL，动作模块随机初始化；旧 RoboMME checkpoint 只提供结构配置。
Stage2 只接本轮 Stage1 的最终检查点，两阶段重新建立 optimizer/scheduler。
新学习率下限是峰值的10%，不是 warmup 第一步 LR 的10%。

## 动作、统计与增强

训练目标前七维为 `absolute_target[t+k] - raw_state[t]`，整个 chunk 共用起点状态；
夹爪不变。变换在 padding、归一化之前执行。推理按下面顺序执行：

1. 用当前原始状态和图像生成归一化相对动作 chunk。
2. 整个 chunk 反归一化，加回本次生成时的七维原始 anchor。
3. 将还原后的绝对动作放入队列，执行前10步；期间新观测不改变队列 anchor。
4. episode reset 清空队列。

状态与相对动作使用 QUANTILES（q01/q99）。统计来自完整1600条训练轨迹的执行段，
不含 demo 起点、val 或 test；不只统计筛选的266条，也不按任务采样权重重加权。
相对动作统计覆盖每个执行起点的20步 chunk，episode 尾部重复最后动作，绝不跨轨迹。
这与官方数据构造顺序一致；实现采用 NumPy 精确线性分位数，官方 RunningStats 为近似直方图统计，
因此不声称统计值与官方发布文件逐位相同。训练 loss 仍屏蔽补齐位置。
统计与源数据哈希、分布范围写入 `run.json/data_audit`；checkpoint 保存实际归一化张量。
不复用旧 absolute/MIN_MAX 处理器，不写入或修改原始数据集。

训练增强只在 dataset 读取时生效，推理不使用增强。非 wrist 做95%随机裁剪、resize、±5°旋转；
两相机使用官方锁定 augmax 0.4.1 的 ColorJitter 语义（brightness=.3、contrast=.4、saturation=.5）。
特别注意该版本默认 p=.5、hue=.1，并存在 saturation 计算结果未使用的行为，本实现忠实保留。
几何操作采用 PyTorch bilinear/zero padding，分辨率、随机流和 uint8 量化与 JAX 不同；
这是官方式 CPU 适配，不是逐像素等价复现。没有额外图像翻转。

## 5090：一条命令串行训练

保留现有 `configs/machine.local.toml`；模型、数据和输出路径沿用原机器配置。
`assets.robomme_initial_checkpoint` 仍可指向旧 Stage1 配置模板，`--fresh` 不加载其中的策略权重。
根目录执行：

```bash
git pull --ff-only
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false \
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run --no-sync python scripts/train_robomme_six_tasks.py \
  --machine configs/machine.local.toml \
  --variant official --run-prefix robomme-six-official-seed1000
```

加 `--dry-run` 只打印两阶段命令。未指定 `--variant` 时仍运行旧配方。
输出：`<runs_root>/robomme-six-official-seed1000-stage1` 和 `...-stage2`。
Stage1 成功并存在 `checkpoints/003000/pretrained_model` 后自动启动 Stage2。
Stage2 保存5000、10000、15000、20000、25000、30000步。已有输出目录会被拒绝。

## 新权重评测

必须使用更新后的 policy server，它会自动根据 checkpoint 动作表示选择安全的还原路径。
直接调用 delta 模型 `select_action()` 会明确报错；`predict_action_chunk()` 仍返回模型空间
的归一化相对动作，外部调用者不能把它直接送入环境。
旧绝对动作权重继续走原路径。新版处理器校验会拒绝缺失 delta 标记、错误 crop、
错误归一化方式或 pre/post 不一致的动作统计。

终端一：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run --no-sync python -m lerobot_policy_vla_ttt.evaluation.server \
  --machine configs/machine.local.toml \
  --checkpoint outputs/robomme-six-official-seed1000-stage2/checkpoints/030000/pretrained_model \
  --implementation evo1_groot --benchmark robomme \
  --n-action-steps 10 --host 127.0.0.1 --port 8795
```

checkpoint 路径按本机 runs_root 修改。终端二沿用[子集评测指南](../robomme-subset-evaluation.md)：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 \
uv run --no-sync --project environments/robomme python -m vla_ttt_eval.subset \
  --plan configs/evaluation/robomme-six-subsets.json \
  --server ws://127.0.0.1:8795 --split val \
  --limit-per-task 20 --seed 1000 --max-steps 1300 \
  --output outputs/eval/six-official-30k-val-matched
```

训练场景诊断使用 `--split train` 并换输出目录。所有任务串行执行，保存 H264 视频及阶段进展。
不同动作表示/统计下的 loss 不可直接比较；实际动作误差应在反归一化、还原绝对目标后计算。

## 参考与验证

官方参考固定为 `RoboMME/robomme_policy_learning@ac846a9a4aec841e483b1dc5d620dfe426bde853`，
重点为 training/config.py、dataset_builder/build_robomme_dataset.py 和 openpi/models/model.py。
本实现只改变选定数据配方，不复现 π0.5 架构、EMA、执行16步或官方训练预算。

本轮验证包括 synthetic 数值/序列队列测试、旧 checkpoint 回归、实际全数据统计和 GPU 短训。
短训使用20步 Stage1 和30步 Stage2（warmup3），仅用于保存/重载与资源验收，不能评估任务成功率。
完整预算 recipe 没有被短测配置覆盖。本轮不会自动启动3000/30000步长程训练。

### 本机短测结果（2026-10-26）

GPU 为 RTX PRO 5000（72GB），不能把耗时直接当作5090实测。

| 阶段 | 实测更新数 | 稳态中位耗时/步 | 峰值 allocated | 峰值 reserved | 正式步数外推 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stage1，batch12 | 20 | 0.793秒 | 9.53GiB | 13.09GiB | 约40分钟 |
| Stage2，batch8 | 30 | 1.978秒 | 17.77GiB | 20.67GiB | 约16.5小时 |

以上是 PyTorch 显存统计，不是整卡总占用；外推不含启动、保存和评测开销。
短测步数少，仅供预算参考，正式训练请根据前几百步重新估计。
两阶段均保存成功；在新 server 进程加载 Stage2 后完成 PickXtimes 的21步环境 rollout，
生成22帧视频和 HTML 报告。短测权重超时未成功，不用于判断学习效果。

验证：全套测试370通过、7跳过；补充调度器新进程序列化检查后，配方测试10通过；
修改的 Python 文件 ruff 检查通过。新旧处理器、原始 anchor、队列 reset 和配置兼容均有测试。
本机证据保存在仓库根目录 `artifacts/experiments/robomme-official-gpu-20261026/`，
其中 `eval-smoke/report.html` 为评测报告，视频在其 `videos/PickXtimes/` 下。
