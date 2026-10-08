# LIBERO：从预训练 VLM + 随机 action head 启动的短程检查

日期：2026-10-01。结论：本次范围内的初始化、训练数值等价性、两阶段切换、保存和隔离环境闭环检查通过。**不是收敛或任务能力验证，不是正式 LIBERO baseline。**

## 1. 范围与来源

- 模型实现基于工作树提交 `9566a538dec4ae288e982d92cd074d6d92e71405`；本次未修改模型、processor、trainer 或评测实现。
- GPU：NVIDIA RTX PRO 5000 72GB Blackwell。训练依赖：LeRobot 0.6.1、Torch 2.11.0、Transformers 5.5.4、Accelerate 1.15.0。
- VLM：本机 `OpenGVLab/InternVL3-1B-hf`，固定 revision `014c0583a0d4bedf29fbe2dbff4f865eb998e171`。
- **未加载 `zuoxingdong/evo1_libero`，Stage 1 的 `pretrained_path=None`。** action head 随机初始化；保留通用预训练 VLM。
- 数据：`/home/hello/lizk/datasets/libero`，`HuggingFaceVLA/libero` revision `86958911c0f959db2bbbdb107eb3e17c5f9c798e`；加载完整的 1,693 episodes / 273,465 frames / 40 tasks，训练只随机抽取短程所需 batch，没有完整遍历数据集。
- 只使用本机已有模型、数据和仿真资产；设置 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`。

## 2. 实验设置

native 使用上游 `evo1`，local 使用本地 `vla_ttt`（未安装 TTT 扩展）。双方使用相同 seed=1000、数据、增强、初始化及超参数，顺序运行，不争用 GPU。

共同设置：两相机输入，模型图像分辨率 448×448；chunk_size=50，n_action_steps=10；dropout=0.2；AdamW lr=1e-5、weight_decay=1e-3、grad_clip=1.0；AMP 开启；FlashAttention 关闭；gradient checkpointing 配置开启（冻结 VLM 时不启用其重计算）；图像增强开启；4 个 DataLoader workers；不在 trainer 内运行仿真。

| 阶段 | 步数 | batch | warmup | VLM 权重 dtype | 可训练参数 |
| --- | ---: | ---: | ---: | --- | ---: |
| Stage 1 | 100 | 8 | 10 | bfloat16 | 122,029,360 |
| Stage 2 | 20 | 4 | 5 | float32 + AMP | 775,198,640 |

Stage 2 从各自 Stage 1 的新 checkpoint 加载，`resume=false`，新建优化器和调度器。检查保存的 AdamW step：Stage 1 的 114 个状态为 100，Stage 2 的 726 个状态为 20，未继承 Stage 1 优化器步数。

短 warmup 和短 cosine schedule **仅用于本次检查**，不应直接用作正式训练配置，亦不应把这些 checkpoint 继续当作正式从头训练的起点。

## 3. 结果

### 3.1 同初始化、真实 batch 的严格数值对照

使用 episode 0 的真实数据，batch=1，真实 action chunk 为 `[1,50,7]`。预测的内部 padded chunk 为 `[1,50,24]`，环境动作由 processor 裁剪、反归一化为 7 维。

| 项目 | Stage 1 | Stage 2 |
| --- | ---: | ---: |
| 初始 state_dict 张量数 | 728 | 728 |
| 初始权重最大绝对差 | 0 | 0 |
| 预测最大绝对差 | 0 | 0 |
| native / local loss | 0.4922797382 / 0.4922797382 | 0.4932191670 / 0.4932191670 |
| 有梯度的张量数 | 114 | 726 |
| 其中 VLM 梯度张量数 | 0 | 612 |
| 梯度最大绝对差 | 0 | 0 |

比较使用 `atol=0, rtol=0`；所有检查到的梯度均有限。

### 3.2 实际 LeRobot 训练循环

| 实现 / 阶段 | 前 10 步平均 loss | 后 10 步平均 loss | 峰值 allocated GiB | 稳态更新中位秒/步 |
| --- | ---: | ---: | ---: | ---: |
| native / Stage 1 | 0.585014 | 0.547710 | 6.755 | 0.510 |
| local / Stage 1 | 0.585014 | 0.547710 | 6.755 | 0.523 |
| native / Stage 2 | 0.566390 | 0.570976 | 14.496 | 0.977 |
| local / Stage 2 | 0.566390 | 0.570976 | 14.496 | 0.987 |

- 稳态更新中位数排除各次运行的前 5 步；不包含模型加载、DataLoader 等待和 checkpoint 保存时间。
- 显存为 PyTorch `max_memory_allocated`，不是整卡进程占用；峰值 reserved 约为 Stage 1 8.404 GiB、Stage 2 17.924 GiB。
- 四次运行共 240 步，无非有限 loss / 梯度范数，无 AMP 跳过更新。
- 两阶段中 native/local 的逐步 loss、梯度范数完全一致，初始化及训练结束后的全部模型权重哈希完全一致。
- Stage 1 的 VLM 权重哈希不变，action head 权重改变；Stage 2 的 VLM 和 action head 权重均改变。
- 四份保存的 checkpoint 各有 728 个张量，规范化的张量内容哈希与训练结束内存状态完全一致。
- Stage 1 初步出现 loss 下降；Stage 2 仅 20 步，loss 波动，**不构成收敛证据**。前后窗口是不同随机 batch，不是固定验证集指标。

### 3.3 保存后加载与 server-client 闭环

加载 local Stage 2 checkpoint，policy 在主项目 Python 3.12 环境运行；LIBERO 在 `environments/libero/.venv` 的 Python 3.11 环境运行，通过仅绑定 localhost:8769 的 WebSocket 服务连接。

- LIBERO Spatial，task 0，初态 episode 0，seed=1000，1 episode，预算 280 步。
- 完整运行 280 步，无接口、渲染、非有限动作或其他基础设施错误。
- `success=false`，`termination=timeout`，耗时 9.534 秒。
- RPC mean=20.34 ms、p95=156.41 ms；包含传输和服务端处理，且大部分调用只取 action queue，不能作为纯模型 forward 延迟。
- 汇总结果 `complete=true, expected=1, completed=1, errors=0`。

这是链路 smoke，不是能力测试。仅训练 100+20 步后单 episode 失败，既不能证明代码错误，也不能证明模型无法学会 LIBERO。实验结束已关闭本次 policy 服务。

## 4. 可复查产物

以下路径均相对于当前工作树：

- `artifacts/experiments/libero-scratch-20261001/run.sh`：四次训练的完整命令。
- `artifacts/experiments/libero-scratch-20261001/probe.py`：包裹实际 LeRobot trainer，只记录指标、权重哈希与有限性，不改变训练算法。
- `artifacts/experiments/libero-scratch-20261001/parity.py`、`parity.json`：真实 batch 的初始化、预测及全梯度比较。
- 同目录四份训练 `.log`、`server.log`、`rollout.log`。
- `outputs/libero-scratch-preflight-20261001/*.audit.json`：初始化来源、命令、参数数量、权重哈希、显存等。
- 同目录 `*.metrics.jsonl`：每一步原始 loss、梯度范数、LR、更新耗时、显存及 AMP skip 状态。
- 四组 `*/checkpoints/last/`：模型、processor、优化器、调度器、RNG 和训练配置。
- `outputs/libero-scratch-preflight-20261001/rollout-local-stage2.jsonl`：闭环配置、checkpoint 身份及 episode 结果。

实验 checkpoint 等占用约 23 GiB，保留以便审计；未删除任何已有资产。`artifacts/`、`outputs/` 是 Git 忽略目录，完整实验产物不会自动随源码提交。

## 5. 下一步边界

可以进入本地 Evo-1 的正式 LIBERO 两阶段训练，不必先重复 native 的完整长训练。正式训练应重新随机初始化 action head，恢复正式训练预算和调度器，不续训本次 smoke checkpoint。

当前仅证明 batch 8（Stage 1）和 batch 4（Stage 2）能稳定运行，尚未寻找 72GB 单卡的最优吞吐 batch。正式训练前可做独立 batch 扫描；长训练后的闭环成功率仍需另行验证，不能从本报告推断可复现官方分数。
