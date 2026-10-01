# VLA-TTT

`lerobot-policy-vla-ttt` 是一个独立的 LeRobot out-of-tree policy plugin，用于在 Evo-1 基线上研究 VLA 的测试时训练（TTT）。当前里程碑只建立可复现的 Evo-1 基线与扩展边界，尚未实现 RoboTTT、TTT-Linear、TTT-MLP、Titans、长序列采样或 TBPTT。

本项目依赖 LeRobot，但不隶属于 Hugging Face、Evo-1 作者或其关联机构。仓库包含从 LeRobot 0.6.1 一次性迁入的 Evo-1 专属源码，保留原始许可证和版权头；不包含模型权重或数据集。

## 环境与安装

RoboMME / LIBERO 的**训练与 server-client 评测**见 [baseline 操作指南](docs/baselines.md)。
policy、LIBERO、RoboMME 使用三个独立虚拟环境；下方旧的单环境 eval 入口仍保留用于兼容。
已完成真实数据两步训练和闭环 smoke，尚未完成正式多任务 baseline 实验。

要求 Python `>=3.12,<3.14`，当前严格支持 `lerobot==0.6.1`。

```bash
uv sync --frozen
```

当前推荐独立安装仿真环境（不把 LIBERO/RoboMME 装入 policy 环境）：

```bash
uv sync --frozen --project environments/libero
uv sync --frozen --project environments/robomme
```

安装后，LeRobot 会通过包名前缀与配置注册自动发现 `policy.type=vla_ttt`，无需 clone 或修改 LeRobot 仓库。

## 基线训练

下面的命令直接使用 LeRobotDataset、Evo-1 processor、训练循环和 checkpoint API：

```bash
uv run vla-ttt-train \
  --policy.type=vla_ttt \
  --policy.training_stage=stage1 \
  --policy.device=cuda \
  --policy.use_amp=true \
  --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/libero \
  --output_dir=outputs/evo1-baseline
```

运行该命令可能下载 `lerobot/libero` 数据集和 InternVL3/Evo-1 相关模型。按本项目规则，首次下载前应先确认 repo、不可变 revision、缓存位置和 dry-run 体积。

插件 checkpoint 可通过项目的薄训练入口 resume；具体 `--config_path` 应指向 LeRobot 保存的 `train_config.json` 或对应 `pretrained_model/` 目录：

```bash
uv run vla-ttt-train \
  --config_path=outputs/evo1-baseline/checkpoints/last/train_config.json \
  --resume=true
```

## 从普通 Evo-1 checkpoint 初始化

`from_pretrained()` 只接受已保存为 `vla_ttt` 的 checkpoint。普通 `evo1` checkpoint 必须使用显式入口：

```python
from pathlib import Path

from lerobot_policy_vla_ttt import VLATTTPolicy, convert_evo1_checkpoint

policy = VLATTTPolicy.from_evo1_pretrained(
    "/path/to/evo1-checkpoint",
    local_files_only=True,
)

convert_evo1_checkpoint(
    "/path/to/evo1-checkpoint",
    Path("artifacts/vla-ttt-checkpoint"),
    local_files_only=True,
)
```

转换会严格检查全部参数 key 和 shape，并保存 plugin config、权重、preprocessor、postprocessor 与转换 metadata。转换后的目录不再依赖原始 checkpoint。

## Rollout 与 LIBERO 评测

真实机器人部署通过 `vla-ttt-rollout` 调用 LeRobot 0.6.1 的 rollout 框架；本地 Evo-1 保留原始 action queue 行为：

```bash
uv run vla-ttt-rollout \
  --strategy.type=base \
  --policy.path=/path/to/vla-ttt-checkpoint \
  --robot.type=<robot-type> \
  --task="<instruction>"
```

`vla-ttt-eval` 调用 LeRobot 0.6.1 的仿真评测框架：

以下是保留的单环境入口，需要自行安装 `--extra libero`；隔离环境实验请用上方 baseline 指南中的 server-client 入口。

```bash
uv run vla-ttt-eval \
  --policy.path=/path/to/vla-ttt-checkpoint \
  --env.type=libero \
  --env.task=libero_spatial \
  --env.task_ids='[0]' \
  --eval.n_episodes=1 \
  --eval.batch_size=1
```

## 测试

CPU 单元测试不需要模型、数据集或 GPU：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run pytest tests/unit -v
```

真实 checkpoint 数值等价性测试是 opt-in：

```bash
VLA_TTT_EVO1_CHECKPOINT=/local/evo1-checkpoint \
uv run pytest -m integration tests/integration/test_evo1_equivalence.py -v -s
```

Hub checkpoint 还必须设置不可变的 `VLA_TTT_EVO1_REVISION`。只有在已经向用户报告下载信息后，才设置 `VLA_TTT_ALLOW_DOWNLOAD=1`。

LIBERO 单 episode smoke test 使用 `VLA_TTT_LIBERO_CHECKPOINT` 和 `VLA_TTT_LIBERO_REVISION`：

```bash
VLA_TTT_LIBERO_CHECKPOINT=/local/vla-ttt-checkpoint \
uv run pytest -m "integration and libero" tests/integration/test_rollout_smoke.py -v -s
```

## 上游与许可证

受支持的精确版本与验证状态见 [`UPSTREAM.md`](UPSTREAM.md)。本项目采用 Apache-2.0 许可证。

## 本地 Evo-1 固定快照

模型源码位于 `src/lerobot_policy_vla_ttt/evo1/`，保留上游七个文件的原名。
`VLATTTConfig` / `VLATTTPolicy` 继承本地配置与 policy，实际执行本地
`Evo1Model`、`InternVL3Embedder` 和 `FlowmatchingActionHead`。
InternVL3 底层 Transformer 仍由 Transformers 提供。

`policy.type=evo1` 保留原生 LeRobot 基线；`policy.type=vla_ttt` 使用本地基线。
来源与导入时的必要修改见 [UPSTREAM.md](UPSTREAM.md)。此快照不主动追踪上游更新。

### CLI 与 processor 恢复

本地配置不继承上游 Evo1Config，因此 LeRobot 0.6.1 通用 factory 中针对上游类型的
checkpoint reconciliation 不会触发。请对本地模型使用上述 `vla-ttt-*` 入口：
它们仅在运行期间替换 runner 模块内的 processor factory 引用，结束后恢复；
训练、dataset、评测和机器人运行循环仍由 LeRobot 提供。该适配只用于单个 CLI
进程，不用于同一进程内的并发 runner。

直接调用上游 CLI 虽然能发现插件，但恢复本地 checkpoint 时不能保证自定义字段、
统计量 padding 和动作后处理覆盖项完整生效。Python 调用方应使用：

```python
from lerobot_policy_vla_ttt.cli import make_pre_post_processors

pre, post = make_pre_post_processors(
    policy.config,
    pretrained_path="/path/to/checkpoint",
    local_files_only=True,
)
```

本地 Evo-1 processor steps 使用 `vla_ttt_evo1_*` 注册名。恢复旧 checkpoint 时，
本地 loader 将原来的 `evo1_*` steps 转为等价本地类；原生注册项保持不变。
模型参数路径与 tensor shape 保持一致，权重严格加载，不使用宽松 key 匹配。

后续研究顺序：RoboTTT → Titans；RoboMME 为主评测、RMBench 为第二评测、
LIBERO 为回归评测。参考 StarVLA 的 server-client baseline 已实现，见 [操作指南](docs/baselines.md)。
