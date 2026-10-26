# RMBench Evo1-GROOT baseline

实现位于 `feat/rmbench-baseline`。训练与真实数据短测已验证；独立仿真环境、官方专家和一个500步 server-client episode 已实际通过链路验收。
安装见 [环境说明](../environments/rmbench/README.md)，证据见 [环境验收](experiments/2026-10-26-rmbench-environment.md)。
不要将短训模型的输出当作有效操作能力。主仓库和其他 worktree 未修改。

## 数据与资源

固定上游 RMBench `7d1cdfc7661c273a7a5c5f6126e2d1a7f0a3f4b7`，
数据 `TianxingChen/RMBench@5d8ff21038fba32ea549e9eee8497aaba9aab5cf`。
`all` 固定为论文九任务，不包含当前上游清单额外的 `place_block_mat`。
九任务 zip 合计约 **11.77 GiB**；解压及 LeRobot 派生数据额外占空间。

```bash
uv sync --frozen --extra rmbench
cp configs/machine.example.toml configs/machine.local.toml
# 编辑 assets.vlm、rmbench_source、rmbench_raw、rmbench_dataset、rmbench_assets，storage.runs_root。
.venv/bin/python scripts/download_rmbench.py --kind data --tasks all --output "$RMBENCH_RAW"
# 确认清单后，加 --download 开始下载；中断后重跑会续传 .partial 并校验 LFS SHA256。
.venv/bin/python scripts/download_rmbench.py --kind assets --output "$RMBENCH_ASSETS"
```

默认 `https://hf-mirror.com`；本机镜像 API 返回 403 时可加
`--endpoint https://huggingface.co`。目录元数据和 HEAD 可经过终端代理，**大文件正文始终绕过代理**，重定向后的 CDN 同样如此。小于等于2 MiB 的非 LFS 文件可回退到官方小文件连接。不改系统代理配置。
`--manifest` 接收预览输出的 revision/files 对象；下载结束保存的文件列表仅作校验记录，不能直接作为该参数输入。
下载 zip 后解压到 `rmbench_raw/data/<task>/demo_clean/`，检查内部有
`data/episode0.hdf5` 与 `instructions/episode0.json`，不要重复套一层目录。
本次 Rearrange Blocks 包内是新版 `<task>/aloha_agilex/`，应解压到
`rmbench_raw/demo_clean/`，得到 `demo_clean/rearrange_blocks/aloha_agilex/data/episode_0000000.hdf5`；转换器支持此布局。先检查ZIP成员再选择解压根目录。

单任务和多任务分别转换，避免统计混入其他任务：

```bash
.venv/bin/python -m lerobot_policy_vla_ttt.baselines.rmbench_data \
  --machine configs/machine.local.toml --tasks put_back_block --output "$SINGLE_DATASET"
.venv/bin/python -m lerobot_policy_vla_ttt.baselines.rmbench_data \
  --machine configs/machine.local.toml --tasks all --output "$MULTI_DATASET"
```

转换使用上游官方 JPEG codec。旧 HDF5：当前图像/state[t] 对应 joint[t+1] 动作；
新版 XPolicyLab 格式已对齐，不再次平移。三路相机为 head/left/right，原图 RGB
320×240；14D 绝对关节目标为左臂6+左夹爪1+右臂6+右夹爪1。
fps=15 沿用上游索引元数据，不代表重新采样到15Hz。
输出 image-backed LeRobot 数据及轨迹 SHA256/任务/指令的 `preparation.json`。
`--max-episodes 1` 仅供短测，正式入口会拒绝 partial 数据。

## 训练

以下是本项目初始实验配方，**不是已验证的官方 Evo1-GROOT RMBench 配方**。
保持 LIBERO 的 chunk50/执行10、448×448、dropout0.2、weight_decay0.001、
clip1、warmup1000；不做图像增强。Stage1 冻结本地通用 InternVL3；Stage2 解冻，
重新创建优化器。动作头随机初始化；不加载 LIBERO/RoboMME 策略。

| 配置 | Stage1 | Stage2 |
|---|---:|---:|
| batch | 8 | 4 |
| 每任务 chunk 曝光预算 | 40,000 | 320,000 |
| 单任务 steps | 5,000 | 80,000 |
| 九任务 steps | 45,000 | 720,000 |
| VLM 存储 dtype | bfloat16 | float32 |

`grouped.json`：动作头 LR=1e-4，VLM LR=1e-5；`uniform.json`：两者1e-5。
多任务先均匀选任务，再均匀选该任务有效帧；不是按任务总帧数加权。
九任务预算很大，分发前请 dry-run 核对，不会自动开始正式训练。

```bash
bash scripts/train_rmbench_single.sh \
  --machine configs/machine.local.toml --recipe configs/training/rmbench/grouped.json \
  --tasks put_back_block --dataset-root "$SINGLE_DATASET" --run-name put-back-grouped --dry-run

bash scripts/train_rmbench_multi.sh \
  --machine configs/machine.local.toml --recipe configs/training/rmbench/grouped.json \
  --tasks all --dataset-root "$MULTI_DATASET" --run-name all-grouped --dry-run
```

移除 `--dry-run` 才训练；加 `--smoke-steps 2` 可各阶段跑两步（batch改为1）。
两入口共用 trainer，当前支持独立单卡工作站，不声称支持多卡 DDP。
`--stage stage1` 只跑第一阶段；`--stage stage2 --checkpoint <stage1/pretrained_model>`
进行阶段切换。`--stage stage1|stage2 --resume <pretrained_model>` 是同阶段恢复。
resume 不允许修改任务、数据、recipe、seed、总步数或 batch；跨机器迁移应保留
run-name、旁侧 `.rmbench.json`、完整 checkpoint（包括 training_state 与
`rmbench_run.json`），只修改 machine 路径。每个任务组合必须有对应的独立数据统计。

## 评测

先完成 [独立环境准备](../environments/rmbench/README.md)。policy 侧：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/vla-ttt-serve \
  --checkpoint "$CHECKPOINT" --implementation evo1_groot --benchmark rmbench \
  --vlm-path "$VLM_DIR" --port 8000
```

仿真侧（与 policy 不同 Python 环境）：

```bash
"$RMBENCH_VENV/bin/python" -m vla_ttt_eval.rmbench_runner \
  --source "$RMBENCH_RUNTIME" --task put_back_block --uri ws://127.0.0.1:8000 \
  --output "$RESULT_DIR" --seed 0 --episodes 100
```

沿用官方 `100000*(1+seed)` 候选种子、专家可行性筛选、seen 指令和 task step_lim。
异常写入单独日志；策略/通信/视频异常立即中止，不能计为策略失败。
默认每个 episode 保存 head-camera H.264/yuv420p/faststart 视频，按动作调用采样、
10fps播放，不声称是物理实时帧率。策略仍接收三路相机。
查看 `progress.json` 成功率，`episodes.jsonl` 单条结果，`candidates.jsonl` 筛选记录。

分片使用相同 `--seed --episodes`，分别增加 `--start-episode 0 --end-episode 50`
和 `--start-episode 50 --end-episode 100`；后片会重放前面的专家筛选以保持同一集合。
每片使用独立输出、runtime 和 policy server。合并时检查重复 episode/seed 与模型身份：

```bash
"$RMBENCH_VENV/bin/python" -m vla_ttt_eval.rmbench_runner --merge "$PART1" "$PART2" > summary.json
```

`complete=false` 的合并结果只是部分进度，不是正式100次结果。
