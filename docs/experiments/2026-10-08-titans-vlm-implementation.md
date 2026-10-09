# 2026-10-08 MAC/MAG/MAL 实现验证

## 范围

在 `evo1-plugin-foundation` 工作树完成 VLM 内三种记忆连接，复用 Evo1-GR00T
动作头和 RoboTTT 公共 TTT 核心；动量/遗忘是独立更新器，四种结构均可选择。
实际结构与论文的适配边界、默认值及启动命令见 [Titans 指南](../titans.md)。

本次保留工作树已有改动，没有启动 GPU 训练、LIBERO rollout、资产下载或修改基线权重。

## 最终回归

在工作树根目录运行：

```bash
PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES='' \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m pytest tests -q -rs
```

结果：**154 passed, 7 skipped，26.47 秒**。

首次沙箱内运行有 6 项环境失败：虚拟环境 CLI 不在 PATH、本地 socket 创建受限、
构建缓存权限受限。加入虚拟环境 PATH 并在允许 localhost/socket 和现有 uv 缓存访问
的测试环境重跑后全部通过；未为通过测试删改相应测试断言。

跳过项均为需要显式指定本地资产的既有集成测试：

- 2 项 Evo-1 等价性测试与 1 项 server 等价性测试：未设置 `VLA_TTT_EVO1_CHECKPOINT`。
- 3 项真实预训练 VLM 测试：未设置 `VLA_TTT_GROOT_VLM_PATH`。
- 1 项 LIBERO rollout：未设置 `VLA_TTT_LIBERO_CHECKPOINT`。

本次修改涉及的 Python 文件通过 Ruff lint、format 检查；`git diff --check` 通过。

## 机制覆盖

- 手算两步动量/遗忘更新、关闭对应项退化为 GD、数据依赖系数外层梯度。
- masked token/样本隔离、NaN padding 排除、权重和非零动量同时冻结、reset/detach。
- MAC 的旧状态检索与 attention 输出写回；MAG 的并行输入与门控；MAL 的 memory
  输出直接进入 attention，避免额外 pre-attention residual 混淆对照。
- persistent/retrieved 前缀可见性、当前观测 causal mask、缺失视角与前导 padding。
- 全 token/视觉位置选择、观测/token 更新计数、跨观测状态、纯函数 checkpoint 重算。
- memory-only 和 joint 训练梯度，穿过冻结 DiT 与未插入 memory 的语言层。
- 小型真实 Transformers InternVL vision/projector/Qwen2 与真实项目 DiT：
  baseline 权重映射、严格 bypass 的有效位置特征和动作对齐、bf16 VLM 配合 FP32
  fast weights/momentum、训练和推理有限值。模型随机初始化，权重加载和 tokenizer
  由本地测试替身提供，不使用外部预训练资产。
- 三种 policy 的本地 LeRobot 图像数据集 CLI 训练、TBPTT、segment transfer、保存、
  重载和下一阶段 warm start；严格配置校验、processor 保存及评测服务 action queue。
- 推理失败不提交候选状态、去噪次数不改变 VLM memory 更新次数、episode reset。
- RoboTTT 使用新更新器的训练、重载、去噪更新计数和推理动量 detach。

## 尚未验证

此记录不证明预训练全尺寸模型的 GPU 内存/吞吐、长轨迹稳定性、训练收敛或机器人
成功率。后续使用固定的已训练 Evo1-GR00T checkpoint，分别比较三种结构、两种更新器、
输入范围和更新粒度；实际运行预算、随机种子及任务结果另行记录。
