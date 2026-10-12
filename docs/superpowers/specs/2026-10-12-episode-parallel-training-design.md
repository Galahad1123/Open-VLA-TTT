# Episode 并行训练设计

在 observation-memory worktree 新增独立训练入口，落实 2026-10-10 LIBERO recipe 并支持 Accelerate 多进程。保留 train_robottt、train_titans 和 tbptt 原有行为；根目录与正在运行的 RoboMME 训练不变。用户已授权推进工程实现。

## 数据与训练语义

每次外层更新均匀抽取一个 task，再在该任务内有放回均匀抽取 global_batch_size 条完整 episode。全局 batch 默认 4；microbatch 固定一条 episode，各 rank 按全局 slot 分配，要求 batch 可被进程数整除。随机种子由 seed、外层更新和全局 slot 确定，不依赖 rank。完整轨迹每 E/2 步产生一个观测事件，每 E 步监督一次。动作 H=50、E=10，TBPTT=8 个事件。每条 episode 从 W0 开始，只 detach 跨段图，不重置数值；fast state 从不归约。

损失为有效动作元素的 chunk 均值，再按 episode 决策数平均，再按全局 episode 数平均。各 rank backward 的分母直接使用全局 batch，最终梯度用 SUM 归约，不再除 world size。每个完整 batch 仅裁剪和更新一次。中点仅观测写入，无动作监督或动作噪声。

## 同步实现选择

Accelerator 管理进程、设备及 backward；使用显式分桶 all_reduce SUM 同步慢参数梯度，不将 policy 传入 Accelerator.prepare 的 DDP 包装。选择理由：现有 forward_step 是显式状态接口，变长 episode 的段数不同，交替阶段会改变 requires_grad。分段期间不通信，外层更新边界以一致的参数顺序同步。通信前归约 grad-present 标志，某 rank 缺失的梯度按零贡献，全局缺失参数保持 grad=None，避免误触发 AdamW decay。初始参数和 buffer 从 rank 0 广播。有限性与本地计算异常在所有 rank 到达更新边界后共同检查。

替代方案为 DDP 分段 forward 适配器加 no_sync：可能获得更好的通信重叠，但需额外处理最后一段未用参数、冻结和不同段数，本轮不采用。FSDP 与 DeepSpeed 留待显存需求出现后评估。

## Recipe 与运行状态

默认 Memory 500 / Policy 500，2000 步；两组 AdamW 独立保持 moments 和实际更新计数。组内按 recipe 分类 VLM、Action Expert、记忆矩阵、W0、T2Mem gate、内层步长；排除 bias/norm decay。warmup=50，cosine 到峰值 10%。FP32 权重与 fast state，CUDA BF16 AMP，统一 MATH SDPA，非重入 checkpointing。图像增强使用 LeRobot 的变换设置，同 episode 同相机重放相同随机参数，独立于 flow/dropout 随机流。

只在外层更新边界保存。保存完整 policy/processor、两组 optimizer、阶段计数、recipe、数据身份、采样进度和各 rank RNG。原子完成标记区分完整与中断 checkpoint。精确恢复要求相同 world size、recipe、模型配置和数据身份；跨 world size 仅支持显式新运行的权重初始化。

## 本地验收与资源约束

不安装或升级共享依赖，不执行 GPU 探测或加载。测试设置 CUDA_VISIBLE_DEVICES 为空、OMP/MKL/OpenBLAS 各 1 线程，低优先级执行，使用 tiny CPU 模型与合成数据。CPU/Gloo 双进程比较单进程同 batch 的 loss、梯度和更新，覆盖实际可微记忆、变长轨迹、两阶段切换、unused 参数及恢复。真实 checkpoint 单 GPU、NCCL 双 GPU、BF16 与吞吐单独记录为未验收；提供后续执行命令。本轮不启动正式训练。
