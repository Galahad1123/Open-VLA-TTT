# RMBench 仿真环境实际验收（2026-10-26）

独立环境已安装并完成真实 GPU 专家及策略闭环；不再受缺少 nvcc 阻塞。
位置为本 worktree 的 environments/rmbench；未修改系统 CUDA/驱动和其他基准环境。

## 资源与安装证据

- 仿真资产位于 /home/hello/lizk/datasets/rmbench/assets，共 346 文件、1,352,865,097 字节（1.26 GiB）。117 个 LFS 文件逐个校验 SHA256，其余小文件校验长度。
- 数据 revision：5d8ff21038fba32ea549e9eee8497aaba9aab5cf；上游代码：7d1cdfc7661c273a7a5c5f6126e2d1a7f0a3f4b7。
- 清华源安装 Python 依赖；用户目录 CUDA 12.8.93 编译 CuRobo 0.7.8 的五个 CUDA 扩展，目标 sm120。PyTorch 2.9.1+cu128；真实 CUDA Pose 运算和 SAPIEN 渲染通过。
- OIDN 2.0.1 在 Blackwell 报 unsupported device type: CUDA。升级官方 2.3.3 后检测到 1 个 CUDA 设备，实际专家渲染不再报错。官方包 64,809,031 字节，SHA256 3c385230d9e6f63527ba72f2229594dbac5051674219d72e0044b5d0b841796f；镜像下载与 19 段官方 CDN 内容交叉核对一致。
- 所有大文件传输绕过终端代理。九任务训练压缩包约 11.77 GiB，本轮未下载完整训练集；已有单 episode 短测样本仍保留。环境及缓存占用另外计算。

## 实际运行

官方 put_back_block 专家，seed=100000：plan_success=true、expert_success=true，耗时 16.614 秒。三路 RGB 均为 240×320×3，状态14维且有限。证据：artifacts/rmbench-environment/expert-oidn233-100000.json 及三张 PNG。

随后项目 policy 环境加载已有两阶段各2步短训检查点，仿真环境通过 websocket 连接 localhost:18330，运行同一任务完整500步。completed_episodes=expected_episodes=1，耗时180.742秒，success=false。此模型仅用于验证链路；0/1不能作为正式模型表现。

客户端命令：

```bash
environments/rmbench/.venv/bin/python -m vla_ttt_eval.rmbench_runner \
  --source environments/rmbench/runtime --task put_back_block \
  --uri ws://127.0.0.1:18330 --output artifacts/rmbench-environment/closed-loop \
  --seed 0 --episodes 1 --max-candidates 10
```

输出 progress.json、episodes.jsonl、候选筛选记录及视频。视频 episode-000-seed-100000.mp4：H.264、yuv420p、320×240、501帧、50.1秒、972,532字节；ffprobe 确认格式，ffmpeg 全片解码无错误。短测结束后关闭本次 policy 服务。

## 审阅与限制

独立审阅发现安装脚本可能继承外部 UV_PROJECT_ENVIRONMENT，现强制指向本环境 .venv，回归先失败再通过。另验证依赖补丁与 OIDN 更新不会修改 uv 共享硬链接缓存，runtime 不写共享机器人配置。

下载器的 --manifest 接收含 revision/files 的预览元数据对象；下载完成记录为文件列表，不能直接作为 --manifest 输入。两种格式暂不统一，文档明确区分。

本次验收范围：一个任务的真实专家及完整策略 episode，不是九任务正式评测。安装流程各组件已实际执行；一键脚本完成语法与隔离回归验证。复现入口见 environments/rmbench/README.md。

最终验证：RMBench、评测协议/结果、Evo1-GROOT、可迁移性回归共61项通过；ruff、shell语法、uv锁文件离线校验、151个已安装包依赖检查和 git diff --check 均通过。
