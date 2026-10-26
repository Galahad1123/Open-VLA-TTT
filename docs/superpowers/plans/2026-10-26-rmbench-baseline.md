# RMBench baseline 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox syntax for tracking.

**Goal:** 同时交付 Evo1-GROOT 单任务/多任务两阶段训练和隔离环境下的官方 RMBench 闭环评测。
**Architecture:** 官方 HDF5 转换为本地 LeRobot 数据；公共 trainer 和两个 shell 入口；官方环境通过 websocket 调用本地 policy。
**Tech Stack:** LeRobot 0.6.1、PyTorch、HDF5、SAPIEN、websockets、ffmpeg。
**Spec:** ../specs/2026-10-26-rmbench-baseline-design.md

用户已批准整体设计及实施，并确认提供单任务和多任务两种入口。
按 Native 方式连续执行，不再次请求批准同一范围。正式长训练不自动启动。

## Global Constraints

- 固定官方 RMBench commit `7d1cdfc7661c273a7a5c5f6126e2d1a7f0a3f4b7`。
- 默认 H.264/yuv420p/faststart；所有机器路径由 machine TOML 提供。
- 两阶段、通用本地 VLM + 随机动作头；分组 LR 动作头 1e-4/VLM 1e-5，另提供统一 1e-5 对照。
- 仿真依赖独立环境；镜像下载，缺少资源不在训练时自动联网。
- 不修改 robomme-task-screen，不推送、不合并、不自动开始正式长训练。

## Review Focus

- RGB 新旧 JPEG 格式、action/state 错位：使用官方 codec 与真实 HDF5 验证。
- 多任务采样或归一化串用：检查 task/episode manifest，checkpoint 保存统计。
- 分组 LR 被统一 scheduler 覆盖：测试两个参数组实际更新与调度比例。
- 不同工作站 checkpoint 的 VLM 路径：重定位测试，禁止网络回退。
- 官方 seed 筛选与错误处理：分别记录无效场景、环境异常和策略失败；视频失败不能伪报完成。

### Task 1: 官方协议、资源与本机配置

**Files:** `src/lerobot_policy_vla_ttt/baselines/rmbench.py`, `configs/machine.example.toml`, `src/lerobot_policy_vla_ttt/portability.py`, `tests/unit/test_rmbench.py`。
**Interfaces:** 固定 TASKS、UPSTREAM_REVISION；`resolve_tasks(names)`；machine 新增 rmbench_source/rmbench_assets/rmbench_dataset/rmbench_raw。
- [x] 核对数据/schema/相机/动作和镜像清单。
- [x] 测试任务集合、未知/重复任务和可迁移路径，先失败后实现。
- [ ] 下载最小真实样本，检查数据与资源体积；提交协议与配置。

### Task 2: 数据转换

**Files:** `src/lerobot_policy_vla_ttt/baselines/rmbench_data.py`, `tests/unit/test_rmbench_data.py`。
**Interfaces:** `prepare(source, raw, output, tasks, ...)`，输出 LeRobot 数据和 `preparation.json`。
- [x] 先用小 HDF5 测试官方图像解码、动作对齐、任务语言、轨迹边界。
- [x] 调用固定版本官方 codec，转换时保存原始轨迹映射和协议身份，不使用特权子任务。
- [ ] 真实样本转换并读取；提交。

### Task 3: 单/多任务训练及分组优化器

**Files:** `training/train_rmbench.py`, `models/evo1_groot/configuration_evo1_groot.py`, `models/evo1_groot/modeling_evo1_groot.py`, `scripts/train_rmbench_single.sh`, `scripts/train_rmbench_multi.sh`, `configs/training/rmbench/`, `tests/unit/test_rmbench_training.py`（源码路径均相对主包）。
**Interfaces:** 公共入口 machine/recipe/tasks/run-name/stage/checkpoint/resume/dry-run；两种入口仅选择任务模式。
- [x] 测试冻结、分组 LR、采样、预算、阶段切换及配置身份，先失败后实现。
- [x] 保留已有默认优化器行为，新增可选 VLM LR；沿用 LeRobot trainer/checkpoint。
- [x] 输出解析后的预算和每任务采样预期，提供保守初始预算及 uniform-LR 对照。
- [ ] 实际数据短训、重载、恢复；提交。

### Task 4: 官方仿真协议与 server-client

**Files:** `packages/eval-client/src/vla_ttt_eval/rmbench.py`, `evaluation/policy.py`, `evaluation/server.py`, `environments/rmbench/`, `tests/unit/test_rmbench_eval.py`。
**Interfaces:** 14 维关节目标及状态、三路 RGB，使用官方 task setup/play_once/check_success/take_action；协议明确 seed 与成功计数。
- [ ] 测试观测/动作 schema、reset、错误及 H264 编码，先失败后实现。
- [x] 扩展 benchmark 路由，保留旧基准校验；记录逐 episode JSONL，默认视频。
- [ ] 安装独立环境，先回放真实演示，再用短训模型 server-client rollout；提交。

### Task 5: 可迁移文档、回归与审阅

**Files:** `docs/rmbench-baseline.md`, `docs/experiments/2026-10-26-rmbench-baseline.md`。
- [x] 记录下载、安装、单/多任务训练、评测、分发及版本复现命令。
- [x] 运行已有相关回归与新测试，记录真实验证范围，不将 mock 当真实验证。
- [ ] 全分支独立审阅，修复重要问题，提交并报告命令与限制。

## 当前验收状态（2026-10-26）

已完成真实数据转换、两阶段GPU短训、检查点重载/恢复、客户端协议与H264测试。
独立审阅的检查点来源和分片/合并问题已修正，相关回归55项通过。
Task 4 的真实仿真安装、专家回放和闭环尚未完成，不能将整个计划标记完成。
详见 `docs/experiments/2026-10-26-rmbench-baseline.md` 的证据及裁定记录。
