# Episode 并行训练

独立入口 `python -m lerobot_policy_vla_ttt.training.train_episode_parallel` 支持 MAC、MAG、MAL、AQMem、T²Mem，已合入 master。原 `train_robottt`、`train_titans`、`tbptt.py` 保持独立。

本入口落实 [LIBERO recipe](experiments/2026-10-10-libero-observation-memory-training-recipe.md) 的完整 episode 训练，既可单进程运行，也可用 Accelerate 启动多 GPU 进程。已记录 tiny 真实模型的 CPU/Gloo 和真实权重单 GPU 验收，见文末；多卡 CUDA/NCCL、BF16 一致性及加速比仍需目标机器验收。首次部署请看 [跨机器训练快速开始](portable-training.md)。

## 并行和损失语义

本协议使用 `HuggingFaceVLA/libero`，固定 revision 为
`86958911c0f959db2bbbdb107eb3e17c5f9c798e`。当前不支持将 `lerobot/libero` 作为替代源，
其导出可能缺少 episode 级 `tasks` 字段。正确下载命令及误下载后的处理见
[跨机器训练快速开始](portable-training.md)。

每次外层更新统一均匀抽取一个 task，再在该 task 内有放回均匀抽取 4 条完整 episode。按全局 slot 分配给各 rank，各自从当前 W0 初始化独立 fast state。默认每卡 microbatch 为 1 条 episode；1/2/4 卡分别在每卡累积 4/2/1 条 episode。全局 batch 必须整除进程数，不自动增大 batch 或学习率。

每个动作 chunk 对有效动作元素求均值，再按整条 episode 的监督决策数平均，最后按全局 episode 数平均。长短 episode 具有相同权重。TBPTT=8 个观测事件，每 E/2=5 个环境步写入一次、每 E=10 步计算动作监督；H=50。中点删除动作输入、不采样动作噪声。跨段仅 detach fast state，slow weights 在完整有效 batch 完成前不更新。

Accelerate 负责启动、进程信息、设备和 backward。本入口**没有把模型交给 DDP 包装**，而是在完整外层更新边界对慢参数梯度显式分桶 `all_reduce(SUM)`。本地 loss 已除以全局 batch，归约时不再除进程数。这样可处理变长轨迹的不同段数、动态冻结和原有 `forward_step` 接口。初始模型参数/buffer 广播一次；episode fast weights 从不归约。globally unused 参数维持 `grad=None`，避免误触发 AdamW decay。

该实现暂不重叠通信与反向传播，也不支持 FSDP/DeepSpeed；不要在启动配置中启用这些模式。各进程使用完整模型和优化器状态，不提供状态分片节省显存。

## 参数与恢复

默认 Memory 500 / Policy 500，合计 2000 次更新。两阶段 optimizer moments 独立保留，各自按实际更新次数 warmup 50，然后 cosine 到峰值 LR 的 10%。短程实验若某组预算小于 warmup，则仅执行该组 warmup。Policy 全量训练视觉、投影、语言模型和 Action Expert；新增记忆组件仅在 Memory 阶段更新。FP32 权重/fast state、CUDA BF16 AMP、MATH SDPA、非重入 checkpointing。

参数分组：VLM/Action Expert LR=1e-5、decay=1e-3；新增记忆矩阵 LR=7.5e-5、decay=1e-5；W0/内层步长不 decay；T²Mem interface gate LR=5e-4、不 decay；bias/norm 不 decay。其它变体融合组件不套用 T²Mem gate 的高 LR。

图像增强优先读取初始化 checkpoint 的 `train_config.json` 中 `dataset.image_transforms`。使用相同 episode、同相机的固定随机种子重放变换，且不消耗模型的 flow/dropout RNG。没有保存增强配置时必须提供 `--augmentation-config`，不会静默替换。局部 episode 流式加载，不常驻整条轨迹的 GPU 图像；首版没有额外 DataLoader workers，数据解码可能限制吞吐。

每 250 次更新及运行终点保存：

```text
run.json
metrics.jsonl
logs/rank-0.log
step-00000250/
  COMPLETE
  training.pt
  pretrained_model/
    config.json
    model.safetensors
    ...processor files...
```

保存只发生在完整外层更新结束、没有活动 episode 时。`training.pt` 包含两组 optimizer、实际更新计数、各 rank Python/NumPy/PyTorch RNG、recipe、数据 metadata 哈希与采样进度（下一外层 step）。采样和 episode 随机流由 seed/step/slot 决定。阶段及 scheduler 值由 recipe 和计数恢复。临时目录写完后原子改名，`COMPLETE` 缺失的产物拒绝恢复。

`--resume` 要求相同 world size、recipe、模型配置和数据身份；输出必须指定新的目录，避免覆盖旧日志。更改 world size 时用 `--checkpoint .../pretrained_model` 开始新的训练阶段，明确属于 warm start。`training.pt` 包含 Python RNG 对象，只加载自己信任的本地 checkpoint。

新运行使用内容身份，增加数据集和 VLM 完整 SHA256，允许相同资源换目录。
旧 checkpoint 保留原有路径限制。`--machine` 管理路径，`--vlm-path` 可覆盖 VLM 位置；
二者均不修改源 checkpoint。完整配置与目录约定见快速开始文档。

## 命令

从已安装的仓库根目录运行，先按快速开始文档填写 `configs/machine.local.toml`。

仅解析配置，不加载权重、不初始化 Accelerator 或 CUDA：

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 uv run --frozen python \
  -m lerobot_policy_vla_ttt.training.train_episode_parallel \
  --architecture t2mem \
  --machine configs/machine.local.toml \
  --run-name libero-t2mem-episode-parallel --dry-run
```

之后在分配的两张 GPU 上进行短程检查：

```bash
CUDA_VISIBLE_DEVICES=0,1 HF_HUB_OFFLINE=1 \
  uv run --frozen python -m accelerate.commands.launch \
  --multi_gpu --num_processes=2 --mixed_precision=no \
  -m lerobot_policy_vla_ttt.training.train_episode_parallel \
  --architecture t2mem \
  --machine configs/machine.local.toml \
  --run-name libero-t2mem-episode-parallel --stop-after 2
```

`--mixed_precision=no` 只关闭 Accelerate 的全局精度包装，policy 自身仍在 CUDA 使用 BF16 autocast，内层记忆计算显式 FP32。`--stop-after 2` 用于短程冒烟，保留完整 2000 步 scheduler，而非缩短训练预算。完整阶段切换资源验收应另外配置较短 phase 或跑到切换边界，检查两组 AdamW 状态同时常驻后的峰值。

同样的启动方式恢复，替换来源参数和输出目录：

```text
--resume /srv/experiments/vla-ttt/libero-t2mem-episode-parallel/step-00000002
--machine configs/machine.local.toml
--run-name libero-t2mem-episode-parallel-resumed
```

通过 `--recipe recipe.json` 覆盖 Recipe 字段，例如受控测试用：

```json
{"steps": 4, "phase_length": 1, "warmup_steps": 1, "save_every": 2}
```

该 JSON 改变实验预算，不能混同正式 2000 步 recipe。`--memory-config` 只允许从 baseline 初始化时指定新记忆结构字段；不允许借此修改主干。首版入口仅支持 canonical LIBERO（action_dim=7），RoboMME baseline 保持现有入口。

## 验证

CPU 单进程与双进程测试，包括五种实际 tiny InternVL/Qwen2+memory 模型：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
  uv run --frozen python \
  -m pytest tests/unit/episode_parallel -q
```

Gloo 需要本机 socket 权限。受限沙箱如果禁止 socket，应在允许 localhost 通信的环境执行，不能把它记成通过或跳过后宣称双进程验证完成。

后续 CUDA probe 使用同一测试脚本，需显式指定设备；不要在当前占用中的 GPU 上执行：

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 PYTHONPATH=src:tests/unit \
  uv run --frozen python -m torch.distributed.run \
  --standalone --nproc-per-node=2 -m episode_parallel.distributed_probe \
  --device cuda --output /tmp/episode-cuda-two
```

另用单 GPU 直接执行同模块得到 `rank-0.pt`，比较保存的逐步 loss、梯度和最终 state。CPU 回归容差为 atol=2e-6、rtol=2e-4；CUDA/BF16 的误差必须另行测量和记录，不应直接放宽容差掩盖分布式语义问题。tiny probe 不能替代真实 Stage1 权重与完整最长轨迹的资源验收。

## 2026 年 10 月 12 日本地验收结果

- CPU 回归：199 passed，7 个 GPU/仿真集成测试未执行；18 条上游 torch.jit 弃用警告。
- 五种真实 tiny 模型：单进程与双进程 loss、慢参数梯度、交替更新结果在 atol=2e-6、rtol=2e-4 内一致；两 rank 状态完全一致。
- 两阶段 optimizer moments 均已建立后的 Gloo 双进程恢复，与不中断运行的模型和 optimizer 状态完全一致；单进程另覆盖第 1/2/3 步中断。
- 真实 Stage1 5k 配置 dry-run 通过；读取了配置和小型 metadata，没有加载真实模型权重。
- Ruff 检查与格式检查通过；原单卡 trainer 无改动。

本轮修复了 CPU 多进程恢复时 Accelerate 的 cpu:0 标识与 safetensors 的 cpu 标识兼容性。以上结果不包含 GPU/NCCL、真实权重训练、显存峰值或吞吐验收。

### 同日补充：真实 GPU 资源验收

驱动恢复且 GPU 空闲后，使用真实 Stage1 5k 权重、完整最长 LIBERO
轨迹进行验证。完整 BPTT 在优化器状态常驻后的第二轮 Policy 出现实际
CUDA OOM；`TBPTT=64`、microbatch=1、有效 batch=4 完成两轮交替，
峰值 allocated / reserved 为 51.06 / 54.04 GiB。修复了 GPU 入口的
未编号 / 带编号 CUDA device 与 LeRobot 配置检查的兼容问题，新增回归后
完整测试为 201 passed、7 skipped（含增强顺序恢复回归）。

运行默认值仍是 `tbptt_events=8`；采用本机测试配置时显式传入
`--recipe configs/ttt/t2mem/libero/recipe-72gb.json`。该文件保留正式的
2000 步预算和 500 步交替，而不是压力测试的 4 步短预算。
详见 [GPU BPTT / TBPTT 验收](experiments/2026-10-12-bptt-resource-validation.md)。

GPU 恢复验收另发现并修复了增强 transform 字典排序导致的随机抽样变化。
现在序列化保留顺序，identity 记录 `augmentation_order`；缺少该字段的旧测试
checkpoint 不直接作为正式恢复来源。新格式恢复后的 937 个模型张量及 optimizer /
RNG / 计数与连续运行精确一致，详见上述报告中的旧元数据恢复说明。

## RoboMME

增加 `--benchmark robomme` 可使用保留 demo 前缀的数据视图；H/E 从初始化 checkpoint 继承，默认 demo 每 E/2 帧写入且不产生监督。训练入口、实测范围和限制见 [RoboMME 训练适配记录](experiments/2026-10-13-robomme-observation-memory-training.md)。LIBERO 默认入口与已有恢复协议保持兼容。

## AQMem LIBERO 联合训练（2026-10-14）

新增 opt-in `training_mode="joint"` 和 `sampling_mode="mixed_tasks"`，配置与命令见
[AQMem LIBERO recipe](../configs/ttt/aqmem/libero/README.md)。未设置时仍为
`alternating` / `same_task`，已有 RoboMME/T²Mem recipe 行为不变。

混合采样先按 task 的随机排列逐轮抽取，再在各 task 内均匀抽 episode：
B ≤ task 数时任务不重复；B > task 数时各 task 计数相差至多 1。
联合模式用一个 AdamW 包含原先的互斥参数组，每个 outer step 同时更新 policy 与 memory，
保留各组 LR/decay，统一裁剪所有活动梯度，warmup/cosine 按联合更新数计算。
完整 VLM 保持可训练。默认候选全局 batch=8、TBPTT=8、gate=0.001、Stage2 20k 初始化。

保存时两种模式进入 recipe；恢复旧 checkpoint 只为缺失的这两个字段补旧默认值，
其余字段仍严格匹配。更换模式必须作为新实验 warm start，不能 exact resume。
日志新增全局 `global_task_count` / `global_task_counts` 和 `optimizer_groups`。
联合 checkpoint 保存一组 joint optimizer；原交替 checkpoint 仍保存两组 optimizer。

后续真实 Stage2 20k 单 GPU 短测已完成 8 步：峰值 allocated/reserved 为 19.46/21.19 GiB，
本机 RTX PRO 5000 72GB 的 2000 步耗时估计约 63 小时，详见 recipe 文档。
CPU/Gloo 精确恢复与跨机器资源重定位测试通过；多 GPU CUDA/NCCL 仍未验收。
