# Evo-1 baseline：训练与隔离环境评测

**2026-10-08更新：** RoboMME Evo1-GROOT正式准备及命令见
[训练前检查报告](experiments/2026-10-08-robomme-baseline-preflight.md)。正式训练复用原视频的执行段视图，
不要把下文早期image-backed转换或300步smoke参数当作正式配方。

本轮实现的是可运行的 baseline 实验闭环。已有真实数据短程训练与真实仿真 smoke；
**尚未完成 RoboMME 正式训练、多任务/多种子成功率实验，不能宣称阶段 A 的论文级 baseline 已完成。**
RoboTTT / Titans 未进入本轮实现。

## 1. 三个独立 Python 环境

从项目根目录运行：

```bash
uv sync --frozen
uv sync --frozen --project environments/libero
uv sync --frozen --project environments/robomme
```

| 环境 | Python / Torch | 安装内容 |
| --- | --- | --- |
| `.venv` | 3.12 / 2.11 | LeRobot 0.6.1、本地 Evo-1、训练、policy server |
| `environments/libero/.venv` | 3.11 / 2.9.1 | hf-libero 0.1.4、MuJoCo、轻量 client；无 LeRobot |
| `environments/robomme/.venv` | 3.11 / 2.9.1 | RoboMME、官方 ManiSkill fork、SAPIEN、轻量 client；无 LeRobot |

Torch 是 benchmark 自身的依赖，并非 client 引入。
`packages/eval-client` 可独立安装，只依赖 NumPy、MessagePack、WebSocket；已在不含 Torch/LeRobot 的干净环境验证。
Git checkout 使用 `uv.sources` 解析本地 client 包；若安装 wheel，需先构建/安装 client wheel，再安装 policy wheel。
不需要复制 LeRobot 的其他模型代码。

## 2. server-client 边界

参考 [StarVLA model server](https://github.com/starVLA/starVLA/tree/4507931a625c844404c4a76128fb116536d8ca7c/deployment/model_server)：

- 二进制 WebSocket + MessagePack + NumPy；本项目协议为 `vla-ttt/1`，不声称 StarVLA wire-compatible。
- client 只发送两张 uint8 HWC 256×256 图、float32 state8、task goal。
- server 运行原生/本地 Evo-1、LeRobot pre/postprocessor 和反归一化，返回**环境单位**动作；client 不再归一化或反转 gripper。
- 每环境步一次 RPC；server 的 `select_action()` 维护 action queue。默认 chunk_size=50、执行前10步后重算。
- 一个 server 只允许一个活跃连接；每 episode 必须 reset，并校验 episode ID、请求序号。
- 断线、超时、重复请求、仿真错误均 fail-fast，不自动重试，不把基础设施错误记成任务失败。
- 当前 server 的 `torch.inference_mode` **仅适用于无 TTT baseline**。后续 TTT 需要显式改造梯度与 memory 生命周期。
- 默认仅绑定 localhost；协议没有鉴权/TLS。跨机器使用 SSH tunnel，不要直接暴露公网。

通用数据、训练循环、优化器、保存/恢复、processor 和 policy action queue 复用 LeRobot。
仿真端的简短 rollout 循环由轻量 client 实现，遵循官方 benchmark 接口及 LeRobot LIBERO 观测约定；
不把依赖整套 LeRobot 的 `lerobot-eval` 塞进仿真环境。

## 3. 固定资产与最小下载

资产均不进 Git；缓存默认 `~/.cache/huggingface/hub`。下载前应核对体积。
以下为本轮使用的不可变版本，完整清单见 [asset manifest](../configs/baselines/assets.json)。

| 资产 | revision | 全量大小；本轮范围 |
| --- | --- | --- |
| `zuoxingdong/evo1_libero` | `515921f4a2c1d3f3ad523721eafa26fdf2af315b` | 约3.1GB；复用已有缓存 |
| `OpenGVLab/InternVL3-1B-hf` | `014c0583a0d4bedf29fbe2dbff4f865eb998e171` | 约1.9GB；复用已有缓存 |
| `lerobot/libero-assets` | `0b3ea86be5fe169d0fd036ae63d1070ec09e90f6` | Hub文件大小总和422MB；下载完整资产 |
| `HuggingFaceVLA/libero` | `86958911c0f959db2bbbdb107eb3e17c5f9c798e` | 34.9GB；只下载首个101MB parquet及meta |
| `lerobot/robomme` | `92796abbc67db4ea6447e5031d04bf47aa3770bb` | 9.58GB；只下载首组415MB视频、首parquet及meta |

可使用 `huggingface_hub.snapshot_download(repo, repo_type="dataset", revision=..., allow_patterns=...)`：
LIBERO smoke patterns 为 `meta/*`、`data/chunk-000/file-000.parquet`；
RoboMME 另加 `videos/image/chunk-000/file-000.mp4` 与 `videos/wrist_image/chunk-000/file-000.mp4`。
这些分片均覆盖 source episode 0；不能拿部分缓存启动全量训练。

下面命令中的 `EVO1_CKPT`、`LIBERO_DATA`、`ROBOMME_DATA`、`LIBERO_ASSETS` 分别设置为上述快照目录。
运行 smoke 时建议 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`，避免底层库隐式下载。
checkpoint 引用的 InternVL backbone 也必须事先缓存。

## 4. RoboMME 数据准备

[官方环境格式](https://github.com/RoboMME/robomme_benchmark/blob/c85e060c830a72571f9517b6dd683d51e154490e/doc/env_format.md)
规定 joint_angle 是7个绝对关节角 + gripper（-1闭，+1开）。
state8 是7个关节角 + 第一个 gripper 宽度；两相机不翻转。

发布的 LeRobot 数据使用 `actions` 列，通用 trainer 的动作时间偏移只识别 `action`；仅 rename_map 不足以构造动作 chunk。
因此使用 LeRobot reader/writer 生成 canonical 数据，并重新计算统计量：

```bash
uv run python -m lerobot_policy_vla_ttt.baselines.prepare \
  --source-root "$ROBOMME_DATA" --episodes 0 \
  --output artifacts/datasets/robomme-smoke
```

- 输出列只有两相机、state、action、task及 LeRobot 必需索引。
- 去除 `is_demo=true` 的控制训练标签；不使用任何 simple/grounded subgoal。
- 当前是 latest-frame baseline，训练与评测均不使用视频演示历史；未来记忆模型另开序列数据路径。
- 本轮 source episode0 属于 RouteStick，96帧中48帧控制数据，输出 action chunk 为 `[50,8]`，尾部由 LeRobot mask。
- `preparation.json` 记录 source revision、episode映射、过滤策略。输出目录必须不存在。
- 输出使用无损 image-backed parquet，减少视频重编码依赖；**全量准备会增加磁盘用量**。

## 5. 初始化与训练

LIBERO checkpoint 的7维动作不能直接用于 RoboMME。
warm-start 保留全部24维 padded head 权重，声明目标8维动作，重建目标数据 pre/postprocessor，**不等于已在 RoboMME 训练**：

```bash
uv run python -m lerobot_policy_vla_ttt.baselines.initialize \
  --source "$EVO1_CKPT" --benchmark robomme --implementation local \
  --dataset-root artifacts/datasets/robomme-smoke --dataset-repo local/robomme-evo1 \
  --output artifacts/init/robomme-local

uv run python -m lerobot_policy_vla_ttt.baselines.train \
  --checkpoint artifacts/init/robomme-local --benchmark robomme \
  --dataset-root artifacts/datasets/robomme-smoke --dataset-repo local/robomme-evo1 \
  --steps 2 --batch-size 1 --seed 1000 --stage stage1 \
  --output outputs/robomme-local-smoke
```

原生对照仅将 initialization 改为 `--implementation native`，并换对应 checkpoint/output 目录。
两者训练均使用 LeRobot trainer；stage1冻结VLM、训练action head，stage2全量微调。
默认正式候选参数是20,000步、batch8，**尚未经正式收敛实验验证，不是论文超参数结论**。
`--dry-run` 打印配置；训练输出保存 LeRobot 标准 `train_config.json`、优化器/调度器和 checkpoint。
在线 env_eval 关闭，训练进程不导入仿真器；评测独立执行。

LIBERO native 可直接使用缓存 checkpoint：

```bash
uv run python -m lerobot_policy_vla_ttt.baselines.train \
  --checkpoint "$EVO1_CKPT" --benchmark libero \
  --dataset-root "$LIBERO_DATA" --dataset-repo HuggingFaceVLA/libero \
  --revision 86958911c0f959db2bbbdb107eb3e17c5f9c798e --episodes 0 \
  --steps 2 --batch-size 1 --seed 1000 --output outputs/libero-native-smoke
```

local LIBERO 使用相同 initialization 命令（benchmark=libero，目标数据为LIBERO）。
恢复训练仍使用已有 `vla-ttt-train --config_path=.../train_config.json --resume=true`。
不要复用已存在的训练 output 目录做新实验。

## 6. 隔离环境闭环评测

终端1：policy环境。LIBERO 可以对**同一 native checkpoint**选择 native/local；
local 分支显式复制权重到本地模型实现，不是 wrapper 调用 native forward。

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run vla-ttt-serve \
  --checkpoint "$EVO1_CKPT" --implementation local --benchmark libero \
  --n-action-steps 10 --port 8000
```

终端2：只启动 LIBERO 环境 Python，不要在这里 `uv run` 主项目。

```bash
MUJOCO_GL=egl HF_HUB_OFFLINE=1 environments/libero/.venv/bin/python -m vla_ttt_eval.runner \
  --benchmark libero --suite libero_spatial --tasks 0 --episodes 1 --seed 1000 \
  --libero-assets "$LIBERO_ASSETS" --server ws://127.0.0.1:8000 \
  --output outputs/eval/libero-local.jsonl
```

LIBERO 使用固定 init-state 编号、10步稳定动作、图像双轴翻转一次、四元数xyzw转axis-angle；
success调用官方 `check_success()`。各suite默认horizon遵循 LeRobot 的280/280/300/520/400设置。
资产配置写入**该仿真虚拟环境内**，不改用户全局 `~/.libero`，不触发hf-libero的浮动下载。

RoboMME：服务端换为训练后8维checkpoint及 `--benchmark robomme`；终端2运行：

```bash
HF_HUB_OFFLINE=1 environments/robomme/.venv/bin/python -m vla_ttt_eval.runner \
  --benchmark robomme --tasks RouteStick --split test --episodes 1 --seed 1000 \
  --server ws://127.0.0.1:8000 --output outputs/eval/robomme-local.jsonl
```

RoboMME 环境种子来自官方split/episode metadata；`--seed`控制policy随机性，不能替代官方环境seed。
RoboMME默认上限已改为官方MME-VLA实验的名义1300步，并跟随wrapper的reset计数偏移；
早期300步仅为smoke配置。Challenge的1500步协议不混报。
正式实验需固定各任务预算并与官方协议核对。不要以训练split代替test结果。

JSONL首行保存运行配置、模型/processor SHA256、依赖版本、动作语义与chunk设置；随后每episode一行。
成功、提前结束、达到步数上限分别记录；错误写 `type=error` 并终止。结果文件采用独占创建，拒绝覆盖。
RPC mean/p95包含观测传输与服务端处理；并非纯模型kernel延迟。

汇总命令：`python -m vla_ttt_eval.results outputs/eval/libero-local.jsonl`。
按任务输出成功数、已完成数、成功率和Wilson 95%区间；缺episode或含error时标记 `complete=false`。
单episode的区间极宽，不能将其分数作为正式baseline。

## 7. 本轮已验证与仍待完成

2026-10-01，RTX PRO5000 72GB，结果保存在当前worktree的 `outputs/`（不提交Git）：

| 实验 | native | local | 解读 |
| --- | --- | --- | --- |
| LIBERO真实episode0数据训练2步 | loss 0.001→0.182 | 同左 | 728个保存参数张量最大差0；非收敛结果 |
| RoboMME RouteStick真实控制数据训练2步 | loss 1.554→0.993 | 同左 | 训练后model文件SHA256相同；非收敛结果 |
| LIBERO spatial/task0/test初态0，已有checkpoint | 79步成功 | 79步成功 | 只有一个episode，不能称100%正式成功率 |
| RoboMME RouteStick/test0，训练2步checkpoint | 300步超时 | 300步超时 | 闭环工作；并不证明任务能力或记忆机制有效 |

训练日志记录的 CUDA peak allocated 约4.75GiB（stage1/batch1）；并非72GB卡的总进程/仿真显存。
两套benchmark的真实reset/render/step均已运行；RPC/直接推理与native/local数值等价测试通过。
Evo-1当前使用eager attention fallback。仿真器的macro/URDF材料警告保留在日志中，无运行异常。

下一步必须做：固定训练/验证/测试任务清单与正式预算，下载所需完整训练数据，训练达到可解释的baseline，
再按一致的checkpoint选择规则和episode集合汇总成功率与置信区间。
RoboMME全量训练和正式LIBERO回归矩阵尚未运行；阶段A路线图不能据此全打勾。

整体审查保留的两个非阻塞日志事项：RoboMME内部timeout目前可能记作terminated（不影响success）；
仿真构造阶段异常会直接退出，可能仅留下run header而无error行（汇总仍判为incomplete）。
