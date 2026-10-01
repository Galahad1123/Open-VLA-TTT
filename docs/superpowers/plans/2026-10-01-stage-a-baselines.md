# 阶段 A：隔离环境的 Evo-1 baseline 实施计划

> 执行方式：Native，使用 superpowers:executing-plans 与 TDD，逐项验证。

**目标：** 接通 RoboMME / LIBERO 数据训练、原生/本地 Evo-1 和隔离环境评测；严格区分接口验证、短程 smoke 与正式 baseline 分数。

**依据：** [总路线图](2026-10-01-vla-ttt-research-roadmap.md) 与本轮用户对 server-client 的明确要求。

**执行记录（2026-10-01）：** 下述接入实现与smoke验收已完成；最终全量测试52通过、1跳过，ruff/锁文件/打包/已安装wheel导入通过。正式训练收敛与多任务评测仍在总路线图阶段A待办中，不以本计划的smoke代替。

**架构：** 参考 StarVLA 的 WebSocket + MessagePack + NumPy 通信。独立轻量 client 包不依赖 LeRobot、Torch 或模型包；policy server 负责模型、processor、反归一化，benchmark client 负责仿真、坐标约定、episode 与结果记录。协议由本项目版本化，不声称与 StarVLA 线协议兼容。

## 约束与决策

- 复用既有 worktree；不修改 LeRobot 安装源码，不合并主分支。
- 训练使用 LeRobot dataset / trainer / processors；不另写训练循环。
- 同时支持 native 与 local baseline，动作/相机/种子/chunk 配置必须一致。
- 默认仅监听 localhost；单连接独占 policy，显式 reset 与请求序号，错误不计作任务失败。
- LIBERO：eef_delta 7 维，状态 8 维，图像双轴翻转一次。
- RoboMME：joint_angle 8 维，状态 8 维，不翻图；仅 task goal 进入 policy，不输入 privileged subgoal。
- 数据与仿真资产固定 revision，下载前报告；正式训练预算另以配置记录，不把短程训练冒充已收敛结果。
- RMBench、TTT、RTC、多并发及跨 episode memory 不在本轮范围。

## Review Focus

1. episode reset/断线后队列与随机状态是否隔离，重试是否造成重复执行。
2. client 不安装 policy 依赖；RoboMME 的 ManiSkill fork 是否被正确固定。
3. 7/8 维动作、图像方向、gripper 符号是否匹配训练数据，是否双重反归一化。
4. 非法数组/错误响应/仿真错误不能被吞掉或污染成功率分母。
5. RoboMME 演示帧不作为控制标签；训练集 task goal 与评测一致、没有子目标泄漏。

### Task 1: 轻量协议与 client

Files: `packages/eval-client/pyproject.toml`, `packages/eval-client/src/vla_ttt_eval/{wire,client}.py`, `tests/unit/test_eval_protocol.py`。

Interfaces: `pack/unpack` 数值数组；`PolicyClient.reset(episode_id, seed)`、`infer(observation)`；握手 metadata 包含 benchmark/action_dim/state_dim。

- [x] RED：数组往返、禁止 object dtype、错误/序号和 reset 的测试。
- [x] GREEN：实现有大小边界的二进制协议和显式客户端生命周期。
- [x] 验证独立安装不引入 Torch/LeRobot。

### Task 2: 原生/本地 policy server

Files: `src/lerobot_policy_vla_ttt/evaluation/{policy,server}.py`, `tests/unit/test_policy_server.py`。

Interfaces: metadata + reset + infer；规范观测两张 uint8 HWC 图、float32 state8、task 字符串；返回经过 postprocessor 的环境动作。

- [x] RED：reset 前拒绝推理、重复序号/episode 拒绝、动作维度检查、真实 socket 往返。
- [x] GREEN：复用 native/local 加载与 processor，服务端保留 action queue，每环境步一次请求。
- [x] 真实 checkpoint 测试网络/直接推理与 native/local 等价。

### Task 3: benchmark adapters 与隔离环境

Files: `packages/eval-client/src/vla_ttt_eval/{libero,robomme,runner}.py`, `environments/`, `tests/unit/test_benchmark_adapters.py`。

Interfaces: adapter reset/step/close；逐 episode JSONL 包含成功、步数、seed、耗时、服务端 metadata 与完整配置。

- [x] RED：图像方向、axis-angle、状态维度、成功/步数上限与 reset 测试（内部timeout日志细节见审查记录）。
- [x] GREEN：官方 env 接口，固定初态编号与 episode split；policy包不进入 env，Torch为仿真器自身依赖。
- [x] 安装隔离环境并分别做真实渲染、单 episode smoke。

### Task 4: 数据与训练配置

Files: `src/lerobot_policy_vla_ttt/baselines/`, `configs/baselines/`, `tests/unit/test_baseline_training.py`。

Interfaces: 固定数据 revision、rename、非演示帧采样、native/local checkpoint 初始化与 LeRobot train config。

- [x] RED：维度、特权标签排除、演示过滤与训练配置测试；真实训练验证目标统计量。
- [x] GREEN：保存审计与可复现训练配置，复用 trainer。
- [x] 真实数据短程训练，保存并重新加载 RoboMME checkpoint 后评测；LIBERO已有checkpoint回归。

### Task 5: 端到端验证与记录

Files: `docs/baselines.md`, `README.md`, 本计划与路线图状态。

- [x] 全量 pytest、ruff、打包、独立 client 环境依赖检查。
- [x] 记录实际训练/评测的运行结果与限制，不用 mock 分数代替真实实验。
- [x] 整体审查；保留分支供用户验收。

## 实现裁决与审查边界

- 每步RPC、服务端action queue：便于后续逐步memory入口，代价是增加传输开销。
- RoboMME标准化数据导出替代trainer内部补丁：修复chunk/统计量/演示标签问题，代价是额外存储。
- 正式收敛、统计显著性与leaderboard可比性未被宣称；正式阶段A实验仍未验收。
- GPU证据由主执行者实际运行，独立审查者未重复GPU实验；保留日志/checkpoint供复核。
- 既有vendor模型不在本次改动/审查范围，保持固定快照；不承担上游算法局限的修正。
- 单连接/断线安全逻辑经审查，专门并发与断线socket回归仍待补，边界覆盖有限。
- 非阻塞事项：RoboMME内部timeout可能记作terminated；仿真构造失败可能只留下run header。
- 打包验证发现默认sdist会读到本地模型，已用显式白名单修复；新增无VCS源码导出回归先失败后通过。
