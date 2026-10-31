# RMBench 独立仿真环境

policy 使用项目 `.venv`，仿真使用本目录 `.venv`。本机 RTX PRO 5000 Blackwell 已实际验证专家轨迹和 server-client 完整 episode；详见 [环境验收记录](../../docs/experiments/2026-10-26-rmbench-environment.md)。

## 安装

Linux 主机需提供 NVIDIA Vulkan 渲染驱动及系统库；仅 `nvidia-smi` 或 CUDA 可用不足以确认仿真可用。Ubuntu 最小容器可先安装：

```bash
sudo apt-get install -y --no-install-recommends libvulkan1 libegl1 ffmpeg
# 如果 NVIDIA ICD 位于 /etc（先确认此文件存在），在运行仿真时设置：
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
```

4090x2 远端曾因缺少 `libEGL.so.1` 出现 `failed to find a rendering device`；补装 `libegl1` 后恢复。`missing pytorch3d` 是上游对可选点云依赖的提示，不是 RGB 评测失败的原因。

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

## 执行诊断日志

评测默认按 episode 写入 `<output>/diagnostics/episode-000-seed-100000.jsonl`。
单任务命令和九任务批量调度使用同一入口，都会生成；可用 `--no-diagnostics` 关闭。
`episodes.jsonl` 的 `diagnostics` 字段指向对应文件，manifest记录 `execution-jsonl-v1`。

- `start` / `end` / `error`：任务、episode、种子、schema版本、初始/最终状态及异常。
- `action`：每次实际执行的14维关节目标、动作后的观测状态、动作计数、奖励、成功标记，以及视频帧号与播放时间（10fps，不是仿真物理时间）。
- `transition`：在官方原有 `check_success` 调用前后观察阶段、按钮计数/标记、奖励等变化，或成功返回；记录原返回值与前后状态，不额外调用判定、不修改动作或阈值。瞬间发生的变化位于记录的前后两个视频帧之间，可能无法在视频帧本身看到。

所有九任务都有通用执行记录；仅存在的通用阶段/计数字段会被记录，不存在不等于零。
当前三个任务有专属诊断：
- `rearrange_blocks`：两个方块/目标位置、XY误差、按钮关节位移、夹爪值及阈值条件。
- `put_back_block`：原始垫子和中央目标、方块坐标、回位/中央位置误差及高度条件、按钮位移与阶段。
- `swap_blocks`：原始/目标托盘对应关系、两个方块交换位置误差、高度与夹爪条件、按钮位移，以及历史有效标记 `flag`。记录导致标记失效的即时X/Z距离条件，但不调用会修改它的 `is_two_blocks`。

夹爪值是上游使用的控制变量，不应等同于真实指尖间距。
这些条件是瞬时状态观测，不是重新实现的成功状态机；最后哪个条件为false不一定就是最早失败原因，应结合transition历史。
其余六个任务的专属成功条件尚未逐一解析。

日志不含图像或权重，每行即时flush；正常结束及异常都会恢复原判定方法。
文件写入失败会让评测报错，不会被计为策略任务失败。
诊断仅在策略episode启用；专家筛选仍记录在candidates.jsonl。
旧评测没有这些状态，无法从旧视频补出精确值，需要使用新入口复跑。
