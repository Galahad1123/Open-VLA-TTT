# 上游版本记录

本文件记录通过本项目测试的、可复现的上游组合。升级任一组件时必须重新运行数值等价性、checkpoint round-trip 和 rollout 测试。

| 组件 | 版本或 revision | 验证状态 |
| --- | --- | --- |
| LeRobot | release `0.6.1` | 安装、插件 discovery、processor、checkpoint 与 rollout CLI 已验证 |
| Evo-1 checkpoint | `zuoxingdong/evo1_libero@515921f4a2c1d3f3ad523721eafa26fdf2af315b` | 原生/插件数值等价与 checkpoint round-trip 已验证 |
| InternVL3 checkpoint | `OpenGVLab/InternVL3-1B-hf@014c0583a0d4bedf29fbe2dbff4f865eb998e171` | 由 Evo-1 离线加载并完成真实推理 |
| LIBERO dataset/assets | 尚未验证 | 尚未下载 |
| Python | `3.12.2`（uv managed） | CPU 单元测试与 GPU 集成测试已验证 |
| PyTorch | `2.11.0+cu130`（由 `uv.lock` 固定） | CPU 与 CUDA 推理已验证 |
| Transformers | `5.5.4`（由 `uv.lock` 固定） | 真实 InternVL3 模型加载与推理已验证 |
| CUDA runtime | `13.0`（PyTorch build） | 已验证可用 |
| NVIDIA driver | `595.91.07` | 已验证可用 |
| GPU | NVIDIA RTX PRO 5000 72GB Blackwell（73,415 MiB，compute capability 12.0） | 数值等价与 round-trip 集成测试通过 |

2026-09-30 的真实 checkpoint 验证结果为 `2 passed in 17.75s`。原生 Evo-1 与插件在相同随机数状态下，以 `atol=5e-3`、`rtol=5e-3` 比较 action chunk；转换后的插件 checkpoint 还通过了保存、重载、推理和 action queue reset 验证。当前运行时没有 Flash Attention 2，LeRobot 自动回退到 eager attention。

LeRobot 0.6.1 的真实机器人部署入口为 `lerobot-rollout`；LIBERO 仿真评测入口为 `lerobot-eval --env.type=libero`。当前未安装或下载 LIBERO 环境资产，单 episode 测试保持显式 skip。

## 本地固定快照

2026-10-01 将 LeRobot 0.6.1 的七个 Evo-1 专属 Python 文件迁入本地正式包。
来源哈希、必要修改和兼容性边界见 [快照记录](docs/evo1-snapshot.md)。
本地版本不主动追踪后续上游 Evo-1 差异。原生 `evo1` 注册项保持不变。

真实 GPU 测试（2026-10-01）：本地/原生 action chunk、训练 loss、action-head
梯度以及转换 checkpoint 的保存/重载/reset 均通过，`2 passed in 18.89s`。
比较采用相同 RNG 状态，`atol=rtol=5e-3`；这属于合成 batch 等价性验收，
不代表已完成 benchmark 成功率评测或真实数据训练。

## 下载规则（模型与数据）

下载任何模型、数据集或 benchmark 资产前，先向用户报告仓库或 URL、不可变 revision、缓存目标和 dry-run 预计字节数。模型和数据不进入 Git。
