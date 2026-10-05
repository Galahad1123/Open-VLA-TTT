# 独立 evo1_groot baseline 实施计划

状态：实现及初次验证完成，独立审查中；用户已授权按既有布局实现。工作分支 `feat/evo1-groot-baseline`，起点 `9566a53`。原 Evo-1 工作树与快照保持独立。

## 实现约定

- 新策略类型 `evo1_groot`；复用本地 Evo-1 的 InternVL3 embedder、图文融合、数据处理、训练阶段与动作队列。预训练 VLM + 随机新动作头；不加载不等价的原 Evo-1 action head。
- 轻量 GR00T-style action expert：state token + action tokens；可配置的 learned register tokens 默认 0（不是跨调用记忆）；交替 cross/self attention，8 个子层（4 对），时间 AdaLN，逐 action token 解码。VLM token 仅作为 cross-attention context。
- 高斯噪声，连续 flow 时间 `t = 0.999 * (1 - Beta(1.5, 1))`；路径 `x_t=(1-t)noise+t*action`，目标 `action-noise`；从 0 到 1 显式 Euler，32 步。保留 Evo-1 的 horizon 50、width 896、FFN 1024 作为轻量初始预算。这是新研究 baseline，不声明等价于官方 GR00T。
- no-memory；不实现 TTT、历史缓存、RTC 或跨模型权重转换。允许动作队列执行前缀，reset 清空队列。
- 新代码位于 `src/lerobot_policy_vla_ttt/models/evo1_groot/`。使用 `configuration_evo1_groot.py`、`modeling_evo1_groot.py`、`processor_evo1_groot.py` 满足 LeRobot 动态 factory 命名；动作头与 token 编码单列文件。
- checkpoint 严格加载，拒绝 `evo1`/`vla_ttt` checkpoint 与非 strict 加载；processor 与 checkpoint 同步保存/恢复；评测 implementation 显式为 `evo1_groot`。

## 顺序与验证

1. 先写 factory/config、真实小尺寸 action-head 的训练与采样行为测试，观察缺失实现失败。
2. 实现 config、tokens、交替 DiT 与 flow matching；验证 mask、状态/上下文影响、batch 隔离、有限梯度与确定性采样。
3. 接入独立 model/policy、processor、CLI 和评测，测试真实 policy（仅用小型 embedder 替代外部 VLM）、严格 checkpoint 往返、阶段冻结与队列 reset。
4. 提供 LIBERO/RoboMME 配置和启动文档；跑完整 CPU 测试、lint、打包检查。若资源允许，运行缓存真实 VLM 的 smoke，不开展完整训练/benchmark。
5. 对整个改动做独立审查，修复关键问题后交付。

## 参考与范围

- 既有设计：foundation 工作树 `docs/superpowers/specs/2026-10-01-ttt-research-architecture-notes.md`。
- NVIDIA Isaac-GR00T `gr00t/model/modules/dit.py`、`gr00t/model/gr00t_n1d7.py` 与 LeRobot 0.6.1 `policies/groot`：结构参考；本实现用现有 PyTorch 组件，不引入其整套模型/依赖。
- 本地 `evo1/`：固定快照，复用不修改。
- 本阶段不声称复现 RoboTTT 结果，也不把 LIBERO 单次初评当作完整成功率。

## Review Focus

重点检查 CPU 小型测试未覆盖的真实 VLM dtype/device、Hub 离线恢复、LeRobot 训练 CLI 的 processor 分派、混合精度下的 flow 数值、所有 context 被遮挡的输入、用户传入不匹配 config 的严格 checkpoint 行为。检查新 baseline 是否意外改变 native/local 路径。

## 执行记录

- action-head 测试先因缺少 `models` 失败，实现后 9/9 通过。
- policy 集成测试捕获 checkpoint 恢复丢失 `embodiment_id`，接入本地 processor 恢复后 13/13 通过。
- 初始化工具测试先因缺少 `initialize_groot` 失败，实现及 preset 添加后 17/17 通过。
- 完整 `pytest tests -q`：66 passed、6 skipped（未设置旧模型集成测试资产 / 新 VLM opt-in 路径）。原始测试环境 PATH、socket 与构建缓存访问问题已通过正确环境运行解决。
- 真实 VLM GPU opt-in 测试：2/2 通过，两阶段各一次 optimizer step + 两次 32 步采样；Stage 1 / Stage 2 loss 均 1.218032（相同随机初始化与输入），梯度 tensor 数分别 120 / 732；head 参数 60,183,320。Flash Attention 2 不可用，使用 eager attention。
- 改动文件 Ruff 检查及 `git diff --check` 通过；没有修改 `evo1/`。
