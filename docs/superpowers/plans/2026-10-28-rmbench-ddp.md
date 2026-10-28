# RMBench 单机双卡适配与远端短测

用户已批准：本机实现与 CPU 验证，通过 GitHub 推送/拉取，在 `095-4090x2` 短测；不使用本机 GPU，不开启正式训练。

1. 复用 LeRobot Accelerate/DDP，增加 `--num-processes`；每个阶段启动独立 torchrun 作业，保留原单卡入口。
2. 增加明确的每阶段 `steps` 预算，与 `chunks_per_task` 互斥；batch 为每卡 batch，记录全局 batch/world size，恢复时禁止改变 world size。
3. rank 0 独占运行记录写入；Accelerate 分片任务均衡采样，CPU 测试覆盖采样与恢复。
4. 新配方：Stage1 5k / 每卡8，Stage2 75k / 每卡4，每5k保存。短测可保留配方 batch，但使用独立 run-name 和小步数。
5. CPU 测试、整体验证和独立审查；推送 `feat/rmbench-baseline`，远端 fast-forward 拉取。
6. 远端使用已有数据/VLM/环境；检查点、日志、缓存、临时文件均位于 `/root/autodl-tmp`。短测两阶段、断点恢复、检查 GPU/训练状态/耗时；记录实测结果及24h预算的不确定性。

## 决策与进度

- 两阶段使用独立分布式进程，避免在同一组进程中重复初始化和销毁 Accelerate 状态。
- 保留旧单卡运行身份格式；DDP 新增 world size/global batch 身份字段，旧单卡恢复仍有效。
- 仅训练代码通过 GitHub 同步；SSH 用于配置本机路径和执行命令，不传输源码。
- 本机目标测试：15 passed；新80k配方 CPU dry-run：全局batch16/8、steps5000/75000。
- 独立审查发现跨epoch恢复被 Accelerate set_epoch(0) 覆盖：已用真实 prepare_data_loader 回归测试复现并修复；同时固定恢复后末尾padding取自原epoch前缀。原始 EpochAwareSampler 不修改。
- 全套测试首次在沙箱下遇到 CLI PATH 缺失和 socket 权限限制；重新使用虚拟环境 PATH、CUDA_VISIBLE_DEVICES 置空并允许 CPU 本地端口测试。
- 完整 CPU 套件：301 passed，7 skipped（GPU/资产集成测试），18条现有 torch.jit 弃用警告；ruff与diff检查通过。未使用本机GPU。
- 远端第一轮短测显示无worker时读图时间占主导；新增 `--num-workers`（每卡），用于保留短测步数的同时测正式数据预取。目标测试15 passed；完整CPU套件再次301 passed / 7 skipped。
