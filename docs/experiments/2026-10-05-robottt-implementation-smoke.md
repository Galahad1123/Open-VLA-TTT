# Evo1-GR00T + RoboTTT CPU 实现验证

日期：2026-10-05。位置：`feat/evo1-plugin-foundation` worktree。
未运行 GPU 训练，未改动另一会话的 `evo1_groot` 模型实现和训练配置。
[使用说明](../robottt.md)；[设计约定](../superpowers/specs/2026-10-05-evo1-groot-robottt-design.md)。

## 范围

新增公共 `ttt/`、独立 `models/robottt/`、`training/`，并扩展 policy 注册和现有
评测服务。论文机制与社区去噪协议分别记录，不宣称完整复现论文实验。

CPU 机制测试覆盖：手算 GD 对照；更新后读取；外层梯度到 W0/QKV/学习率；
批内隔离；masked write；reset/detach；逐去噪累计；slow 参数不变；
关闭记忆及 registers 后与 baseline 数值对齐；严格 checkpoint 配置检查；
服务 reset 与动作队列；episode 窗口和 padding；真正分段 backward；
整段梯度与显式展开一致；context-only 不读取动作标签；真实小型 LeRobot
图像数据集经 CLI 初始化、训练、保存并重新加载。

外部 InternVL 在测试中替换为小型可训练 embedder，DiT/TTT/processor/训练器均为
真实代码。HF 数据集为测试创建的两条短 episode，不是 LIBERO/RoboMME 任务。

因果测试：改变后续轨迹位置不影响先前输出；改变历史会影响当前输出。
历史真实动作 chunk 扰动也能改变当前输出——这是记录中的信息路径诊断结果，
不是无泄漏证明。detach 后后续 loss 不再回传 W0，但仍能训练 QKV。

## 小模型优化检查

CPU、seed=12，固定两位置随机 context/state/action，外层每步 seed=100 固定 flow
噪声，只优化 memory/registers，Adam lr=0.01，共 60 步。
DiT dim=16、hidden=24、2 layers/4 heads；memory dim=4、hidden=6；H=3、action/state=4。
初始 loss=2.4259538651，末尾 loss=1.8234789371（比值 0.75165）。
这只说明内外梯度链可被优化，不说明真实任务训练已经收敛。

## 验证命令

```bash
PATH="$PWD/.venv/bin:$PATH" OMP_NUM_THREADS=1 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest tests -q

.venv/bin/ruff check src/lerobot_policy_vla_ttt/ttt \
  src/lerobot_policy_vla_ttt/models/robottt src/lerobot_policy_vla_ttt/training \
  tests/unit/robottt src/lerobot_policy_vla_ttt/__init__.py \
  src/lerobot_policy_vla_ttt/evaluation/policy.py \
  src/lerobot_policy_vla_ttt/evaluation/server.py
```

全量回归首次有效运行：83 passed、7 skipped；之后补充两个因果/截断测试。
最终结果与审查修复在下方追加。现有回环 socket/打包测试需要沙箱允许本地端口
与 uv 缓存写入；首次受限运行的环境失败不计为通过。

## 未完成的阶段 B 实验

正式 GPU 训练与 baseline 公平比较、LIBERO/RoboMME 任务收敛、闭环成功率、
显存/吞吐测量、去噪步数与训练 stride 消融、RoboMME demo 视频上下文协议。
CLI 目前是权重/归一化 warm start，不支持精确恢复 optimizer/RNG/数据位置。

## 独立审查

独立审查未发现默认配方的 Critical/Important 缺陷，并额外核对了联合训练的
VLM 梯度及不同 TBPTT 段长度的数值状态一致性。

审查发现关闭记忆且选择 memory-only 会无有效训练；实现方将其提升为应立即
修复的问题。新增回归先复现未报错，再加入训练入口检查；17 个 RoboTTT 测试通过。
关闭记忆的推理对照仍可使用。

后续补充的回归覆盖：混合 batch 行 reset 紧邻 padding；分段梯度与显式 detach
参考及 activation 释放监测；形状不变的 checkpoint 语义/相机顺序拒绝；完整
服务对全部 slow 参数和 gradient buffer 的前后比较。已有 primitive 级测试不
等同于这些更全面的组合覆盖；目前未观察到对应实现错误。

最终审查修复后全量结果：**86 passed, 7 skipped**（23.97 s）；新增范围 Ruff 全部
通过，`git diff --check` 通过。7 项跳过对应现有外部 checkpoint/CUDA/基准环境条件。
