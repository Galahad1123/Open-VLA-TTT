# evo1_groot 实现与管线 smoke

日期：2026-10-05。工作树 `.worktrees/evo1-groot-baseline`，分支 `feat/evo1-groot-baseline`，基于 foundation `9566a53`；实现起点提交 `56b55f2`，后续修复见本分支提交历史。

结论：独立无记忆 baseline 已实现；新配置、训练、严格 checkpoint 恢复和评测 policy 入口已验证。**这是实现与短管线验证，不是正式训练或闭环成功率实验。** 原 `evo1/` 快照无改动，另一个工作树的原生/本地对照实验未修改。

## 验证结果

| 检查 | 结果 |
| --- | --- |
| 完整离线 `pytest tests -q` | 68 passed、6 skipped；未提供旧模型集成资产 / 新 VLM opt-in 路径的测试按条件跳过 |
| 显式启用真实 VLM GPU 测试 | 2 passed；Stage 1 / Stage 2 backward、optimizer step、32 步采样；Stage 2 含真实 Accelerator/update_policy |
| Ruff（改动 Python 文件）、`git diff --check` | 通过 |
| 本地 VLM + LIBERO stats 初始化 | 成功生成独立 config/model/processors/provenance |
| LIBERO Stage 1 → Stage 2 | 各两步，均 `End of training`，两阶段 checkpoint 保存成功 |
| Stage 2 checkpoint 评测边界恢复 | 严格加载，processor 恢复，输出有限 7 维动作；reset + 同 seed 重放完全一致 |

CPU 测试只替换昂贵的 VLM，保留真实新 action head、policy、processor。GPU 测试使用本地 InternVL3-1B-hf snapshot `014c0583a0d4bedf29fbe2dbff4f865eb998e171`；Flash Attention 2 不可用，走现有 eager attention 回退。权重加载报告中的 `lm_head.weight` 不参与 VLM feature extractor。

## 真实数据短训练

数据：`/home/hello/lizk/datasets/libero`，`HuggingFaceVLA/libero` revision `86958911c0f959db2bbbdb107eb3e17c5f9c798e`，仅 episode 0、214 帧。batch=1、seed=1000、每阶段两步，继承训练 runner 的 smoke optimizer/scheduler 设置。不是完整数据训练预算。

| 项目 | Stage 1 | Stage 2 |
| --- | ---: | ---: |
| trainable params | 60,183,320 | 713,352,600 |
| 总参数 | 713,352,600 | 713,352,600 |
| Step 1 loss | 1.328 | 1.132 |
| Step 2 loss | 1.032 | 0.929 |
| 日志峰值 allocated 显存（GiB） | 2.44 | 7.30 |

Stage 1 初始化来源：`artifacts/init/libero-evo1-groot-smoke`，预训练 VLM、随机 head、目标数据 stats。修复后的最终输出：

- `outputs/libero-evo1-groot-bf16-stage1/checkpoints/000002/pretrained_model`
- `outputs/libero-evo1-groot-bf16-stage2/checkpoints/000002/pretrained_model`
- 日志：`outputs/logs/evo1-groot-training-bf16-stage1.log`、`outputs/logs/evo1-groot-training-bf16-stage2.log`。

以上权重、训练状态和日志保留在新工作树的 ignored 目录，不进入 git。

## 检查中修复的问题

1. checkpoint 恢复的通用 processor converter 会丢失 embodiment 等字段：新策略接入既有本地恢复/校正逻辑，真实 pipeline 往返测试覆盖。
2. 严格 tensor 加载不能发现 `num_heads` 等不改变 tensor shape 的语义变化：额外比较模型与输入配置，校验 camera 顺序；设备、训练阶段、执行前缀等白名单运行参数仍可覆盖。两项回归先观察失败再修复通过。
3. 初次实际 Stage 2 训练失败：继承的 `use_amp=True` 使 LeRobot 选择 fp16 Accelerator，GradScaler 无法 unscale bf16 VLM gradients。通过新配置 `dtype="bfloat16"` 对接 LeRobot 现有精度接口，让训练器与内部 autocast 一致；保留 `dtype="float32"` 关闭 head autocast。GPU 回归在修复前复现 unscale 异常，修复后通过，真实两阶段训练也重新通过。原始失败日志保留为 `outputs/logs/evo1-groot-training-stage2-smoke.log`。

独立只读审查发现/确认两项 Important（配置语义、AMP），均已修复并回归；无 Critical、无延期 Minor。没有为此修改原生/本地 Evo-1 代码或 LeRobot 安装包。

操作入口及配置解释见 [evo1_groot 指南](../evo1-groot-baseline.md)。下一步是固定公平对照预算开展正式无记忆训练/评测，再接入 RoboTTT；当前没有 TTT memory，也没有新模型的 benchmark 成功率。

## Foundation 集成与交接

2026-10-05，按用户要求将 `56b55f2`、`d952130` 快进合入 `feat/evo1-plugin-foundation`。实现、配置、测试和文档现已位于 foundation；未修改原 `evo1/` 快照。

- 在 foundation 根目录使用现有 `.venv`，不设置跨工作树 `PYTHONPATH`，完整离线测试 **68 passed、6 skipped**。
- 实际包路径指向 foundation 的 `src/lerobot_policy_vla_ttt`；factory 分别解析 `evo1`、`vla_ttt`、`evo1_groot`。初始化与评测服务 CLI help 正常，新评测 implementation 可用。
- 合并前已有的 roadmap、架构笔记与三份 LIBERO 实验文档，共 5 份未提交文件，合并后 SHA256 全部保持一致；未将这些其他会话的修改纳入本次提交。
- 原开发工作树及其 ignored smoke 资产保留，未移动/删除权重，也未在 foundation 启动新的训练或评测。后续训练会话从 foundation 新建实验目录；本会话后续在同一项目内接入 RoboTTT。
