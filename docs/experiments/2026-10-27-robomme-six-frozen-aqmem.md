# RoboMME 六任务：冻结 VLM 的 AQMEM 对照实验

在 `feat/robomme-task-screen` 工作树实现；不修改 master、RMBench 工作树、共享虚拟环境或数据集。
本轮不做VLM编码批处理、特征缓存、数据预取等性能优化，不自动启动正式训练。

## 配方与对齐

两组从同一 `robomme-six-official-seed1000-stage2/checkpoints/030000/pretrained_model` 初始化。
baseline只训练action DiT，AQMEM联合训练action DiT和memory慢参数；VLM均冻结。
保留相对关节动作与检查点内的q01/q99统计，不重算或覆盖统计。

- 配方：`configs/training/robomme-screen/multitask-six-frozen-baseline.json` 与 `multitask-six-frozen-aqmem.json`。
- 使用相同266条轨迹，六个任务每次各抽一条，global batch=6；microbatch仍为一条episode。
- 每次更新完成六条episode；各episode内部先平均监督chunk loss，再平均episode。更新频率未改为每chunk一次。
- H=20、执行间隔10、memory写入间隔5、demo stride=5、TBPTT events=8；完整episode重置memory，TBPTT仅截断梯度。
- sampler按任务族分组，传给模型的原始语言指令不变。
- 官方式增强按episode/camera固定随机参数，保持时间一致性；两组使用相同变换，不消耗flow-time随机流。
- action LR峰值1e-5，memory慢参数外层LR峰值1e-4；warmup 5%，cosine最低比例0.1；梯度裁剪1。
- AQMEM沿用 `t2mem_adaptive` 内循环规则、gate初始值0.001。1e-4不是内循环快权重步长。
- 关闭额外的LeRobot generic图像增强，避免与官方式增强叠加；评测不做增强。

**1000更新、warmup50、保存250只是未校准模板，不是已验证的10小时预算。**
暂按AQMEM单组约10小时理解预算；待GPU短测后，两组同步修改steps、warmup和save_every。
不能分别训练满10小时，否则监督量不一致。baseline可以较早完成相同更新数。
指标新增 `sampled_episode_ids`、峰值allocated/reserved显存，可核对两组采样和有效监督数量。

## 安全GPU短测

5090已有完整30k权重时直接使用；本机测试前需复制完整 `pretrained_model` 目录（含模型权重、config和处理器文件）。
先在5090现有 `experiment/robomme-task-screen` 分支执行 `git pull --ff-only`。
以下在本分支根目录运行，沿用已有machine配置：

```bash
uv run --no-sync python scripts/probe_robomme_six_memory.py \
  --machine configs/machine.local.toml \
  --checkpoint outputs/robomme-six-official-seed1000-stage2/checkpoints/030000/pretrained_model \
  --gpu 0 --updates 3 \
  --output outputs/probes/robomme-six-frozen-run01
```

脚本首先检查权重和GPU计算进程；占用时拒绝运行，不停止或修改已有训练。
按baseline→AQMEM串行，各完成3次完整更新并保存独立检查点，不覆盖正式recipe或已有输出。
进程检查不能防止其他会话在检查之后启动任务，因此仍应安排空闲GPU时间窗。

输出各组log、metrics、检查点和summary，报告loss、显存、每次更新耗时和10小时更新数估算。
完成后自动生成 `outputs/probes/robomme-six-frozen-run01/probe-logs.tar.gz`，只需复制这个压缩包回来。
它包含recipe、硬件信息、运行元数据和逐步日志，不包含模型或optimizer权重。
子进程报错时也会打包已有日志，先发送日志诊断；不要直接复用旧输出目录重跑，换run02。
运行期间可在另一终端执行 `tail -f outputs/probes/robomme-six-frozen-run01/aqmem.log` 查看AQMEM进度。
每次更新累积六条完整episode，不是六个chunk，首次出指标可能需要等待较久。
估时使用第一步之后的中位数，排除启动与保存开销；只有三个batch，仍需后续根据更多轨迹校准。
第一轮global batch维持6；其内episode串行累积，增大global batch并不等同于增加GPU计算microbatch。

## 正式命令模板（预算校准后使用）

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false \
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run --no-sync python -m lerobot_policy_vla_ttt.training.train_episode_parallel \
  --machine configs/machine.local.toml \
  --checkpoint outputs/robomme-six-official-seed1000-stage2/checkpoints/030000/pretrained_model \
  --architecture aqmem --benchmark robomme \
  --recipe configs/training/robomme-screen/multitask-six-frozen-aqmem.json \
  --memory-config configs/training/robomme-screen/multitask-six-frozen-memory.json \
  --run-name robomme-six-frozen-aqmem-run01
```

baseline改为 `--architecture evo1_groot`、baseline recipe和独立run-name，并去掉memory-config。
两组均是权重初始化，optimizer/scheduler重新建立；完整恢复使用trainer保存的 `--resume`，不是原随机chunk训练的checkpoint。

## 推理与验证

相对动作在每个chunk起点反归一化并加回原始关节anchor，整段队列保持这个anchor。
AQMEM执行队列期间仍将每个新观测交给HalfExecutionLifecycle，由其在5步中点写入memory；
决策点继续使用原有读写顺序。demo和reset沿用现有接口。

旧配方新增字段默认关闭，旧检查点recipe补齐默认值后仍可恢复。
CPU验收包含冻结参数/优化器、任务分组、增强随机数隔离、实际观测时钟的demo/0-5-10/reset路径。
本机目前仅复制了正式训练日志，没有30k模型权重；已有RMBench训练占用GPU，因此尚未进行本轮真实GPU短测。

本轮CPU回归：378 passed、7 skipped；新增7项测试通过；修改的Python文件ruff与git diff --check通过。
GPU短测未执行，因此当前没有可报告的实测显存、吞吐或正式10小时更新数。

补充验收：短测脚本的GPU占用拒绝和日志包排除权重测试2项通过；未在本机启动GPU计算。
