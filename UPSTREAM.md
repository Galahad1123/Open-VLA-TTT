# 上游版本记录

本文件记录通过本项目测试的、可复现的上游组合。升级任一组件时必须重新运行数值等价性、checkpoint round-trip 和 rollout 测试。

| 组件 | 版本或 revision | 验证状态 |
| --- | --- | --- |
| LeRobot | release `0.6.1` | 待验证安装与 discovery |
| Evo-1 checkpoint | 尚未验证 | 尚未下载 |
| InternVL3 checkpoint | 尚未验证 | 尚未下载 |
| LIBERO dataset/assets | 尚未验证 | 尚未下载 |
| Python | `3.12.2`（uv managed） | CPU 单元测试已验证 |
| PyTorch | `2.11.0+cu130`（由 `uv.lock` 固定） | CPU 单元测试已验证；CUDA 尚未验证 |
| Transformers | `5.5.4`（由 `uv.lock` 固定） | 导入已验证；真实模型尚未验证 |
| CUDA runtime | 尚未验证 | 尚未验证 |
| NVIDIA driver | 尚未验证 | 尚未验证 |
| GPU | RTX PRO 5000 72GB（用户提供） | 尚未运行集成测试 |

当前执行环境未设置 `VLA_TTT_EVO1_CHECKPOINT`，且 `nvidia-smi` 无法连接驱动；因此真实 checkpoint 数值等价性测试保持显式 skip，不能据此宣称 GPU 验证完成。

LeRobot 0.6.1 的真实机器人部署入口为 `lerobot-rollout`；LIBERO 仿真评测入口为 `lerobot-eval --env.type=libero`。当前未安装或下载 LIBERO 环境资产，单 episode 测试保持显式 skip。

## 下载规则

下载任何模型、数据集或 benchmark 资产前，先向用户报告仓库或 URL、不可变 revision、缓存目标和 dry-run 预计字节数。模型和数据不进入 Git。
