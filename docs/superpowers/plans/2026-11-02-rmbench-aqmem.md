# RMBench AQMem 四轮交替训练

已批准：从三任务85k baseline加载全部VLM/DiT参数；仅put_back_block、swap_blocks，冻结整个VLM；四轮memory 3K / policy K，LR5e-4/2e-6，各自累计更新5% warmup后cosine至0.1倍。H50/E24，中点12，global batch4，TBPTT8，seed1000。

实现：复用现有完整episode累积与显式跨卡梯度求和；RoboMME远端参考提交3cfe8cc的冻结标志逻辑，保留stage defaults关闭。新加不等长阶段和阶段边界保存，RMBench按preparation的任务名筛选且保留原全局索引。复用原processor/数据统计和逐episode时序一致增强。优化器与调度跨轮保留；快状态每episode初始化，不跨轨迹共享。

计划任务：
1. 调度/冻结、独立cosine和恢复兼容测试。
2. RMBench数据/三摄像头/14D action序列接入。
3. 双卡短测：完整交替周期、保存恢复、吞吐和显存；按23h预算计算K，正式24h含保存余量。
4. GitHub同步代码后SSH后台启动正式训练；大文件全部autodl-tmp。
5. 保存每阶段检查点；同种子20episode/任务，baseline/AQMem E24，h264+诊断，官方和免按钮指标并列。

暂以逐episode梯度累积作为数值参考，每卡处理两个episode；批处理实验必须额外验证，不能引入其他RoboMME模型结构改动。正式K只由实际吞吐确定，仓库recipe的K=1为短测模板，不能作为正式预算。所有8个阶段边界保存；预计内存/磁盘用量需远端短测确认。
