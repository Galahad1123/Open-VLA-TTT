# 上游版本记录

本文件记录通过本项目测试的、可复现的上游组合。升级任一组件时必须重新运行数值等价性、checkpoint round-trip 和 rollout 测试。

| 组件 | 版本或 revision | 验证状态 |
| --- | --- | --- |
| LeRobot | release `0.6.1` | 待验证安装与 discovery |
| Evo-1 checkpoint | 尚未验证 | 尚未下载 |
| InternVL3 checkpoint | 尚未验证 | 尚未下载 |
| LIBERO dataset/assets | 尚未验证 | 尚未下载 |
| Python | `3.12` | 待验证 |
| PyTorch | 尚未验证 | 尚未验证 |
| Transformers | 尚未验证 | 尚未验证 |
| CUDA runtime | 尚未验证 | 尚未验证 |
| NVIDIA driver | 尚未验证 | 尚未验证 |
| GPU | RTX PRO 5000 72GB（用户提供） | 尚未运行集成测试 |

## 下载规则

下载任何模型、数据集或 benchmark 资产前，先向用户报告仓库或 URL、不可变 revision、缓存目标和 dry-run 预计字节数。模型和数据不进入 Git。
