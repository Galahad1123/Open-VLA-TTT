# VLA-TTT

`lerobot-policy-vla-ttt` 是一个独立的 LeRobot out-of-tree policy plugin，用于在 Evo-1 基线上研究 VLA 的测试时训练（TTT）。当前里程碑只建立可复现的 Evo-1 基线与扩展边界，尚未实现 RoboTTT、TTT-Linear、TTT-MLP、Titans、长序列采样或 TBPTT。

本项目依赖 LeRobot，但不隶属于 Hugging Face、Evo-1 作者或其关联机构。仓库不包含 LeRobot 源码、模型权重或数据集。

## 环境与安装

要求 Python `>=3.12,<3.14`，当前严格支持 `lerobot==0.6.1`。

```bash
uv sync --frozen
```

LIBERO 是可选环境，不进入默认开发环境：

```bash
uv sync --frozen --extra libero
```

安装后，LeRobot 会通过包名前缀与配置注册自动发现 `policy.type=vla_ttt`，无需 clone 或修改 LeRobot 仓库。

## 基线训练

下面的命令直接使用 LeRobotDataset、Evo-1 processor、训练循环和 checkpoint API：

```bash
uv run lerobot-train \
  --policy.type=vla_ttt \
  --policy.training_stage=stage1 \
  --policy.device=cuda \
  --policy.use_amp=true \
  --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/libero \
  --output_dir=outputs/evo1-baseline
```

运行该命令可能下载 `lerobot/libero` 数据集和 InternVL3/Evo-1 相关模型。按本项目规则，首次下载前应先确认 repo、不可变 revision、缓存位置和 dry-run 体积。

插件 checkpoint 可继续通过标准训练 CLI resume；具体 `--config_path` 应指向 LeRobot 保存的 `train_config.json` 或对应 `pretrained_model/` 目录：

```bash
uv run lerobot-train \
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

真实机器人部署继续使用 LeRobot 0.6.1 的标准 `lerobot-rollout`；本项目不实现机器人循环、transport 或 action queue：

```bash
uv run lerobot-rollout \
  --strategy.type=base \
  --policy.path=/path/to/vla-ttt-checkpoint \
  --robot.type=<robot-type> \
  --task="<instruction>"
```

LeRobot 0.6.1 的 LIBERO 仿真入口是 `lerobot-eval`，它复用相同 policy、processor 和 rollout/evaluation 基础设施：

```bash
uv run lerobot-eval \
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
