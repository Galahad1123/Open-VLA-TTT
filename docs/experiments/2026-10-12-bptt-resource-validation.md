# LIBERO 完整 BPTT / TBPTT 实测验收

日期：2026-10-12。工作区：`.worktrees/observation-memory`，分支 `feat/observation-memory`。

**结论：本机 T²Mem 单卡短程训练预检通过，已验证候选 TBPTT=64、microbatch=1、有效 batch=4。最长轨迹完整 BPTT 在优化器常驻后 OOM。已修复 GPU 入口与增强顺序恢复问题，最终恢复对照精确一致。没有合并 master，也没有启动正式训练。**

后续讨论澄清：当前训练方案仍保留 TBPTT=8；本报告证明 64 的资源可行性，不代表用户已确定以 64 替代 8。2000 步为探索预算，训练量问题留待后续决定。文中“未合并 master”为测试当时状态。

## 测试范围

本次测试使用真实 Stage1 5k 权重与 canonical LIBERO，不是 tiny 模型或随机图像。通过新增的 `training.train_episode_parallel` 入口运行，保留真实 processor、同一 episode/相机一致的图像增强、三级损失归一化、分组 AdamW、梯度裁剪、交替阶段和 checkpoint 保存。资源探测只替换采样器以连续选取最长 episode，并增加显存日志。

- T²Mem-style；VLM 14 层、Action Expert 8 层，沿用 Evo1-GR00T 主干。
- 双相机 448×448，预测 H=50，执行 E=10，每 5 个环境步写入一次。
- FP32 慢参数 / fast weights，BF16 AMP，MATH SDPA，非重入 gradient checkpointing。
- Policy 阶段全量训练 VLM 与 Action Expert；Memory 阶段训练新增记忆组件。
- 两个 AdamW 优化器在同一进程常驻，跨阶段不清空 moments。
- 最长 episode=306：505 帧、101 个观测事件、51 个监督决策。
- 压力测试采用 Memory→Policy→Memory→Policy，每阶段 1 次外层更新，以尽快覆盖状态常驻与反复切换。它不替代正式 recipe 的 500 步交替或收敛验证。

环境：RTX PRO 5000 72GB Blackwell，CUDA 实际总容量 71.10 GiB，开始时可用约 69.93 GiB；驱动 595.99.02，PyTorch 2.11.0+cu130。没有其他训练进程，桌面远程服务约占 493 MiB。本次没有停止该服务。

## 完整 BPTT

当前 `Recipe.tbptt_events` 要求正整数。探测设为 100000，大于数据集最长的 101 个事件，因而整个 episode 只在末尾反向，等价于完整 BPTT；不要设置 0。

最长轨迹、microbatch=1、有效 batch=1 的第一次 Memory 和第一次 Policy 更新均可完成；但第二次 Policy 在双方优化器状态已建立后 OOM：

| 测试 | 结果 |
| --- | --- |
| 67 GiB allocator 保护上限 | 首次 Policy 峰值 allocated 64.70 GiB；第二次 Policy OOM |
| 取消 allocator 上限 | 第二次 Policy 仍 OOM；请求分配 194 MiB 时整卡仅剩 42.31 MiB |

未设上限的失败信息：PyTorch 已分配 68.68 GiB、缓存未分配 1.11 GiB，进程含非 PyTorch 分配占用 70.22 GiB。这是实际容量压力下的标准 CUDA OOM，与 10 月 10 日受驱动影响的 NVML 错误不同。

因此，**当前配置在这张卡上不能把最长轨迹完整 BPTT 作为稳定训练协议**。首步成功不能代表 optimizer 状态常驻后的稳态可行。该结论限定于本机、当前模型、精度和内存分配设置；不代表更大显存 GPU 或未来 offload/重计算实现也不可行。

## TBPTT 设置解释

`tbptt_events=64` 表示每 64 个观测事件反向并 detach fast state，约对应 320 个环境步的采样周期；并非执行 64 次动作或去噪。跨段保留 fast weights 数值，不重置记忆。段内 loss 使用整个 episode 的监督决策数和全局有效 batch 作分母，全部段/episode 反向结束后才更新慢参数。

按真实 LIBERO 元数据计数：

| 分段长度 | 无需跨段截断的 episode |
| --- | ---: |
| 8 个事件 | 0 / 1693 |
| 32 个事件 | 1128 / 1693（66.63%） |
| 64 个事件 | 1647 / 1693（97.28%） |

所以 64 个事件在保留完整历史的同时，也能让绝大多数 LIBERO episode 使用完整 episode 的梯度传播；最长轨迹分为 64+37 个事件。这里描述的是可传播范围，不保证每条梯度路径都非零。

## TBPTT=64、有效 batch=4 的实际结果

每个外层更新顺序处理 4 条完整最长轨迹，使用不同的 slot seed（增强与动作噪声独立）；microbatch 始终为 1。这里为压力测试而重复选取 episode 306，不是声称正式 sampler 总是选择同一条轨迹。

| 外层更新 | 阶段 | 峰值 allocated / reserved（GiB） | 更新耗时 | Loss | 裁剪前梯度范数 |
| --- | --- | --- | ---: | ---: | ---: |
| 1 | Memory | 15.95 / 17.01 | 93.29 s | 1.045824 | 40.1949 |
| 2 | Policy | 45.71 / 48.46 | 163.16 s | 0.950580 | 13.1473 |
| 3 | Memory | 21.81 / 48.51 | 87.49 s | 0.671846 | 8.4416 |
| 4 | Policy | **51.06 / 54.04** | 159.52 s | 0.789433 | 17.4148 |

4 次更新全部完成，覆盖 16 次完整 episode、1616 个观测事件、816 个监督决策。Loss 与梯度范数均有限。裁剪阈值仍为 1.0；表中记录的是裁剪前范数。第二轮 Memory 的 reserved 包含上一阶段留下的 allocator 缓存，不等于其活跃张量占用。

测试全过程约 520.57 s（包含初始化、日志与第 2/4 步 checkpoint 保存）。表中更新时间不包含紧随该更新的保存耗时；它们是最长轨迹压力测试吞吐，不是随机抽样正式训练的平均吞吐。

第 2 步 checkpoint 已验证同时保存 Memory / Policy 的 AdamW 状态，分别约 0.483 / 5.315 GiB。第 4 步保存后计数为 Memory=2、Policy=2。保存产物位于本机 artifacts；不得作为正式实验的初始化替代共同 Stage1 起点。

[完整压力测试日志](../../artifacts/preflight-20261012/tbptt64-max-b4.log) / [运行 manifest](../../artifacts/preflight-20261012/tbptt64-max-b4/run.json)。

## 启动入口修复与回归

实际 GPU 启动暴露了两处设备格式兼容问题，已经在独立 episode trainer 内修复：

1. 单进程 Accelerate 返回未带 index 的 `cuda`，而 `torch.cuda.set_device` 要求具体设备编号；现在只对未编号的 CUDA device 补充当前编号，保留多进程的 rank 分配。
2. LeRobot 配置的 AMP 检查只接受 `cuda` 等类型名，不接受 `cuda:0`；配置使用类型名，实际运行仍使用 Accelerate 指定设备与当前 CUDA device。

新增 indexed-CUDA 配置回归测试，先复现失败再修复。全项目测试在允许 localhost socket、正确设置虚拟环境 PATH 的环境下：**201 passed、7 skipped**；18 条上游 torch.jit 弃用警告。7 个跳过项为需额外 GPU/checkpoint/仿真条件的集成测试，不算通过。Ruff 检查与格式检查通过。

首次沙箱测试为 193 passed、7 skipped、7 failed：失败项来自禁用 socket、CLI 未在 PATH、打包缓存权限限制。修正测试运行环境后同一完整测试集全部通过，没有删除或忽略失败测试。

## 恢复测试发现的增强顺序问题

首轮 GPU 恢复运行本身完成，但第 3 步 Memory 的 loss 从连续运行的
0.6718463985 变为 0.6732396159，裁剪前梯度范数从 8.441594 变为
7.553220。没有把这次运行记为精确恢复通过。

定位发现：checkpoint 的 `canonical()` 使用 `sort_keys=True`，把增强字典
从 `brightness, contrast, saturation, hue, sharpness, affine` 重排成
`affine, brightness, contrast, hue, saturation, sharpness`。LeRobot 按字典
顺序构建 transform 列表，随机数抽到同一索引时会选中不同增强。相机顺序、
模型配置、阶段与数据计数并没有变化。

修复：JSON 类型归一化保留映射顺序；运行身份另存 `augmentation_order` 列表，
防止普通字典等值比较放过顺序变化。新增像素级增强 round-trip 回归测试，
先复现 756/768 个像素元素不一致再通过；CLI 恢复检查也覆盖同内容、不同顺序
的增强配置并明确拒绝。

旧测试 checkpoint 已把顺序丢失，不能声称修复后自动恢复所有旧产物。
本次从原 Stage1 `train_config.json` 找回当时实际使用的顺序，创建独立诊断副本，
仅修复 `identity.augmentation` 并添加 `identity.augmentation_order`；
模型、优化器、RNG、计数和 recipe 均不改，原文件保留。该副本用于重新跑第 3 步，
随后从修复代码生成的新第 3 步 checkpoint **不作人工修改**恢复第 4 步。
这一恢复链同时检查新格式的正常保存/加载。旧格式没有顺序字段时，正式 CLI
会拒绝直接恢复；不要将本次诊断元数据恢复当成通用自动迁移功能。

恢复对照最终结果：第 3/4 步的 episode loss、外层 loss、梯度范数和计数与连续运行一致；第 4 步 **937 个模型张量全部逐元素完全一致**。两个优化器、RNG、计数、recipe 等训练状态也完全一致。训练身份的比较仅对连续运行旧产物补上上述已知的增强顺序元数据；不忽略任何模型、optimizer 或 RNG 差异。

[最终比较结果](../../artifacts/preflight-20261012/resume-comparison.json) /
[修复后的 Memory 恢复](../../artifacts/preflight-20261012/resume-fixed-step3.log) /
[新 checkpoint 的 Policy 恢复](../../artifacts/preflight-20261012/resume-fixed-step4.log)。

## 建议配置与训练就绪范围

使用 [72GB 实测 recipe](../../configs/ttt/t2mem/libero/recipe-72gb.json)：

- 每卡 microbatch 为 1 条完整 episode，有效 batch=4；单卡通过梯度累积实现。
- TBPTT=64 个观测事件；跨段保留记忆数值，episode 边界重置。
- 正式阶段保持 2000 次更新、每 500 次切换、各组 warmup 50；从共同 Stage1 5k 权重新建 optimizer，不加载本次压力测试权重。
- `Recipe` 类默认值仍为 8，必须显式传入上述 JSON 才使用本机实测的 64。没有因为本机资源扩大就改变其他架构的默认值。

正式命令结构（本次只做配置 dry-run，未实际启动正式预算）：

```bash
PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m lerobot_policy_vla_ttt.training.train_episode_parallel \
  --architecture t2mem \
  --checkpoint /path/to/common-stage1-5k/pretrained_model \
  --dataset-root /path/to/libero \
  --recipe configs/ttt/t2mem/libero/recipe-72gb.json \
  --output outputs/libero-t2mem-episode-seed1000
```

目标服务器要使用其实际数据与 checkpoint 路径、已安装本项目的 Python 环境。
本次证据足以支持 **T²Mem 在相当单卡资源上的正式训练准备**，仍有以下边界：

1. 单 GPU 无法验证多 GPU 的 CUDA/NCCL 通信；CPU/Gloo 双进程回归通过不能替代目标服务器的多卡短测。目标服务器先做涵盖 Memory/Policy 切换的短 recipe、checkpoint 恢复与显存检查，再启动正式预算。
2. 没有测量 AQMem/MAC/MAG/MAL 的真实模型资源，不把 T²Mem 的 51.06 GiB 峰值套用给它们。为公平比较，尽量统一截断协议，并在各架构实测后确认。
3. 未运行 2000 步收敛实验或 LIBERO rollout 成功率评测。全数据集训练中的数值稳定性与效果不能由 4 次外层更新保证。
4. 本次没有合并、提交 master 或发布代码。修改留在 observation-memory worktree；可据此进入合并审查和目标服务器验证。

## 原始证据

本机 `artifacts/preflight-20261012/` 保存探测脚本、JSON 配置、日志、运行 manifest 与后续 checkpoint。该目录被 Git 忽略，不自动随代码分发。

- [机器可读报告](../../artifacts/preflight-20261012/report.json)
- [探测脚本](../../artifacts/preflight-20261012/probe.py)
- [67 GiB 保护上限完整 BPTT](../../artifacts/preflight-20261012/full-max.log)
- [无上限完整 BPTT](../../artifacts/preflight-20261012/full-max-uncapped.log)
- [完整回归测试](../../artifacts/preflight-20261012/pytest-final.log)
- [既定训练 recipe](2026-10-10-libero-observation-memory-training-recipe.md)
- [episode 训练入口说明](../episode-parallel-training.md)
