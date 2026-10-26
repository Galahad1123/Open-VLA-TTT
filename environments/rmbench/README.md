# RMBench 独立仿真环境

policy 使用项目 `.venv`，仿真使用本目录 `.venv`。本机 RTX PRO 5000 Blackwell 已实际验证专家轨迹和 server-client 完整 episode；详见 [环境验收记录](../../docs/experiments/2026-10-26-rmbench-environment.md)。

## 安装

从仓库根目录运行：

```bash
bash environments/rmbench/setup.sh
# 在 machine TOML 中配置 assets.rmbench_source 和 assets.rmbench_assets。
environments/rmbench/.venv/bin/python environments/rmbench/prepare_runtime.py \
  --machine configs/machine.local.toml --runtime environments/rmbench/runtime

environments/rmbench/.venv/bin/python environments/rmbench/preflight.py \
  --runtime environments/rmbench/runtime --task put_back_block --seed 100000 \
  --output artifacts/rmbench-environment/expert.json
```

安装脚本固定自己的 UV_PROJECT_ENVIRONMENT，Python 包通过清华源下载；大文件传输清除代理。CUDA 编译工具链安装在本目录 toolkit，无需修改系统 CUDA 或驱动。CUDA 包和 OIDN 压缩包有 SHA256 校验。OIDN 官方直连失败时使用 GitHub 下载镜像，仍校验同一散列。

实际组合：Python 3.11、PyTorch 2.9.1+cu128、CUDA Toolkit 12.8、SAPIEN 3.0.0b1、mplib 0.2.1、Open3D 0.18、CuRobo 0.7.8（d64c4b005459db10c5dd867d8b30a87d5bda9bdb）。uv.lock 固定 Python 依赖；CuRobo 从固定源码单独编译，installed-freeze.txt 记录最终环境。

上游 RMBench 固定 7d1cdfc7661c273a7a5c5f6126e2d1a7f0a3f4b7。相对官方旧环境，为兼容 Blackwell，采用上述 Python/PyTorch/CUDA 组合，并将 SAPIEN 的 OIDN 更新为 2.3.3；[官方变更记录](https://github.com/RenderKit/oidn/blob/v2.3.3/CHANGELOG.md)明确加入 Blackwell 支持。原 2.0.1 无法创建该显卡上的 CUDA 去噪设备。保留官方渲染器和物理设置。RGB 流程不使用点云，不安装可选 PyTorch3D；上游打印 missing pytorch3d 不影响本次 RGB 路径。

patch_dependencies.py 应用官方 SAPIEN UTF-8 和 mplib 补丁，采用原子替换，避免改动 uv 的共享硬链接缓存。prepare_runtime.py 私有化机器人配置，按官方脚本改写本机 URDF 路径，网格和物体资源共享只读。每个并发评测进程使用独立 runtime。

## 闭环评测

在 policy 环境启动 vla-ttt-serve（benchmark=rmbench，implementation=evo1_groot）后：

```bash
environments/rmbench/.venv/bin/python -m vla_ttt_eval.rmbench_runner \
  --source environments/rmbench/runtime --task put_back_block \
  --uri ws://127.0.0.1:18330 --output artifacts/rmbench-eval \
  --seed 0 --episodes 1 --max-candidates 10
```

保留官方专家筛选，默认输出 H.264/yuv420p 视频。正式评测参数与训练入口见 [baseline 文档](../../docs/rmbench-baseline.md)。本次只验证一个任务的专家和一个完整策略 episode，不代表九任务性能验收。
