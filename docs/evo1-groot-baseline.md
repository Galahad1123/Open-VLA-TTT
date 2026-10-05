# evo1_groot 独立 baseline

`policy.type=evo1_groot` 是预训练 InternVL3 + 随机初始化轻量 GR00T-style action expert 的**无记忆**模型。它与原生 `evo1`、本地快照 `vla_ttt` 分别注册、训练和保存，不用于证明原生/本地 Evo-1 数值等价。尚未完成任务训练或成功率评测。

2026-10-05 已将实现提交 `56b55f2`、`d952130` 快进合入 `feat/evo1-plugin-foundation`。训练与后续 RoboTTT 开发统一使用 `/home/hello/lizk/vlas/vla-ttt/.worktrees/evo1-plugin-foundation`；模型代码位于该工作树的 `src/lerobot_policy_vla_ttt/models/evo1_groot/`，现有 `.venv` 已能直接发现新策略。原 `evo1-groot-baseline` 工作树仅保留开发历史和 smoke 资产，不是第二个独立项目。

交接训练会话时，从 foundation 根目录使用下文的初始化/训练入口；现有环境可直接用 `.venv/bin/python` 代替 `uv run python`，无需重装。正式训练请新建 foundation 下的初始化和输出目录；旧的两步 smoke 权重及日志仍在原开发工作树，未复制到 foundation，也不作为正式训练起点。先固定与 Evo-1 对照的训练预算及超参数，再开展训练；评测服务指定 `--implementation evo1_groot`。本次移植没有启动正式训练或评测。

## 模型与代码

| 部分 | 实现 / 复用边界 |
| --- | --- |
| VLM | 复用固定 `evo1/internvl3_embedder.py`；InternVL3-1B-hf，14 层语言模型；沿用图像、任务文本和 camera mask 处理 |
| Token | `models/evo1_groot/tokens.py`：state token、可选 register tokens、每步 action token；动作位置编码与连续时间编码 |
| DiT | `action_head.py`：8 个交替 cross/self-attention 子层，共 4 对；width 896、8 heads、FFN 1024；时间 AdaLN；cross-attention 读取 VLM tokens |
| Readout | 逐 action token 预测 velocity；没有 Evo-1 的整段 pooling；默认 horizon 50、执行前缀 10 |
| Flow | 高斯初始噪声；`t=0.999*(1-Beta(1.5,1))`，`x_t=(1-t)ε+t*a`，目标 `a-ε`；32 步 Euler 从 0 积分到 1；积分累加与 loss 使用 fp32 |
| Policy / model | `modeling_evo1_groot.py` / `model.py`：独立构造动作头，复用 Evo-1 policy 的 batch、loss、冻结控制与动作队列 |
| Processor | 新 factory 入口，复用本地 Evo-1 normalization/padding/crop；checkpoint 恢复时保留任务、embodiment 和 mask 字段 |
| Config | `configuration_evo1_groot.py`；注册独立类型并检查尺寸、时间分布、完整 attention 对与不支持的 RTC |

`num_layers` 指 attention **子层**数，每个子层均带 FFN。`num_register_tokens=0` 是默认值；非零时只是每次调用重新展开的可学习常量，不是跨帧 memory。state/register/action 参与 self-attention；VLM context 只参与 cross-attention。整段动作使用一个 flow 时间；尚未实现跨观测序列的 Sequence Action Forcing。

原 Evo-1 的 CategorySpecificMLP 用于 state/action 编解码（新参数），从预训练资产加载的只有 VLM。新 checkpoint 只支持严格恢复 `evo1_groot`，不接受 `evo1`/`vla_ttt` checkpoint，不提供隐式 head 权重迁移。加载时还校验架构与输入语义，拒绝不改变权重形状的 `num_heads` 修改或 camera 顺序变化；允许设备、训练阶段、精度、推理步数、执行前缀、优化器及加载路径等运行参数覆盖，具体白名单位于 `Evo1GrootPolicy._runtime_overrides`。恢复仍需初始化时记录的 VLM 资产路径；迁移机器时将 `config.vlm_model_name` 改为同一固定版本的本地路径。

`dtype="bfloat16"` 是训练器与 policy autocast 的权威配置；`dtype="float32"` 关闭动作头 autocast，VLM 参数精度仍由 `vlm_dtype` 独立决定。继承的 `use_amp` 只作为通用工具兼容字段，不能用它选择新模型的精度。显式 dtype 让 LeRobot 0.6.1 的 Accelerator 选择 bf16，避免 fp16 GradScaler 无法处理 Stage 2 的 bf16 VLM 梯度；不支持 `dtype="float16"`。CPU policy 不启用内部 autocast。

结构参考：[NVIDIA Isaac-GR00T 固定版本](https://github.com/NVIDIA/Isaac-GR00T/tree/51d4c89f72fda44cbf77285c6a8114b52676b8a1)，`gr00t/model/modules/dit.py`、`gr00t/model/gr00t_n1d7/gr00t_n1d7.py`，以及已安装的 LeRobot 0.6.1 GR00T 实现（Apache-2.0）。本动作头独立使用 PyTorch 编写；未搬入 NVIDIA 的完整模型或 diffusers DiT。它不是官方 GR00T 的权重兼容实现。

后续 TTT 可从 `GrootActionHead.blocks` 中的 self-attention 位置接入；跨调用状态、更新规则和 episode reset 必须另行实现。本次不添加空的 TTT 框架，不改变 `evo1/` 快照。

## 初始化与训练

在本工作树安装项目环境（`uv sync --locked`），以下命令从本工作树根目录运行。VLM 使用已下载的固定版本 `014c0583a0d4bedf29fbe2dbff4f865eb998e171`；下列数据路径需替换成实际本地目录。

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export GROOT_VLM_PATH=/home/hello/.cache/huggingface/hub/models--OpenGVLab--InternVL3-1B-hf/snapshots/014c0583a0d4bedf29fbe2dbff4f865eb998e171
export GROOT_DATASET_ROOT=/path/to/local/libero

uv run python -m lerobot_policy_vla_ttt.baselines.initialize_groot \
  --config configs/baselines/evo1_groot/libero \
  --dataset-root "$GROOT_DATASET_ROOT" --dataset-repo HuggingFaceVLA/libero \
  --benchmark libero --vlm-path "$GROOT_VLM_PATH" \
  --output artifacts/init/libero-evo1-groot --seed 1000

uv run python -m lerobot_policy_vla_ttt.baselines.train \
  --checkpoint artifacts/init/libero-evo1-groot --benchmark libero \
  --dataset-root "$GROOT_DATASET_ROOT" --dataset-repo HuggingFaceVLA/libero \
  --revision 86958911c0f959db2bbbdb107eb3e17c5f9c798e \
  --output outputs/libero-evo1-groot-smoke --stage stage1 \
  --steps 2 --batch-size 1 --seed 1000 --dry-run
```

去掉 `--dry-run` 执行两步管线 smoke。初始化目录必须不存在，写入 config、model、processor 和 provenance；dataset stats 来自目标数据集。初始化不加载旧策略 head。Stage 1 冻结 VLM，Stage 2 以 Stage 1 checkpoint 为 `--checkpoint` 并指定 `--stage stage2`，默认解冻 VLM。已有训练 runner 也支持此策略，无需修改 LeRobot。

正式对照训练时另行固定训练预算、优化器、warmup、数据 split 与 seed；上述 smoke 命令不是已审计的 5k/80k Evo-1 实验配置。新模型的 Gaussian/Beta flow、AdaLN、attention 和 readout 均不同，不能只用相同步数声称唯一变量是 TTT。

RoboMME 使用 `configs/baselines/evo1_groot/robomme`、`--benchmark robomme` 与已准备的 canonical control 数据（8 维 state / 8 维 action）；要求 `preparation.json`，不能把 demo 或 privileged labels 当作控制监督。其余初始化步骤相同。

## 推理与验证

训练后通过已有 server/client 管线评测，显式选择新模型：

```bash
uv run vla-ttt-serve \
  --checkpoint /path/to/evo1-groot/pretrained_model \
  --implementation evo1_groot --benchmark libero --device cuda --n-action-steps 10

uv run pytest tests/unit -q
VLA_TTT_GROOT_VLM_PATH="$GROOT_VLM_PATH" \
  uv run pytest tests/integration/test_evo1_groot_vlm.py -q -s
```

CPU 测试使用真实新 action head/policy/processors，只有外部 VLM 替换为小型模块。GPU 测试使用真实缓存 InternVL3、正式尺寸动作头，两阶段分别完成 backward/optimizer step，以及 32 步采样、padding 和固定 seed 检查；Stage 2 额外通过实际 `Accelerator/update_policy` 验证训练器混合精度。没有设置 VLM 路径时显式 skip，不自动下载。

新模型的训练结果应另开实验记录；当前代码正确性 smoke 不能替代 LIBERO/RoboMME 闭环评测。

本次验证结果、两阶段 smoke checkpoint 与已修复问题见 [实现验证记录](experiments/2026-10-05-evo1-groot-implementation-smoke.md)。
