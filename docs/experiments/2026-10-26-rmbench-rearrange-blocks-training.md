# Evo1-GROOT / RMBench Rearrange Blocks 单任务训练

用户于2026-10-26批准：先进行 Rearrange Blocks 单任务实验，检查点每5000更新保存一次。

- 入口：scripts/train_rmbench_single.sh；专用 recipe：configs/training/rmbench/rearrange-blocks-grouped.json。
- Stage1：batch8、5000更新，冻结本地通用 InternVL3；随机动作头，LR=1e-4。
- Stage2：加载本次Stage1检查点，解冻VLM，batch4、80000更新；动作头LR=1e-4、VLM LR=1e-5。重建优化器。
- 两阶段save_freq=5000；chunk50、执行10、448×448、dropout0.2、wd0.001、clip1、warmup1000+cosine，关闭图像增强，seed1000。
- 独立任务 recipe 避免改动已有 grouped/uniform 配置。机器路径保存在忽略的 configs/machine.local.toml，训练离线。
- 原始数据固定 TianxingChen/RMBench@5d8ff21038fba32ea549e9eee8497aaba9aab5cf，单包789047110字节；SHA256=26fcda70d4d09935093d8028339dfcf05644939e527782d9557c4ec85e6195c8。
- Stage2计划在20k/40k/60k/80k验证；此轮只启动训练，不隐式运行并发评测。

## 变更验证

新增dry-run回归先因Stage2仍为10000保存间隔失败，再实现recipe save_freq覆盖，两个阶段均5000通过。相关回归62项通过。独立审阅未发现阻止启动的重要问题；save_freq的通用正整数显式校验留待后续，当前固定值5000有效。

## 启动命令

从该worktree根目录执行，输出路径由machine配置决定：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_rmbench_single.sh \
  --machine configs/machine.local.toml \
  --recipe configs/training/rmbench/rearrange-blocks-grouped.json \
  --tasks rearrange_blocks --run-name rearrange-blocks-evo1-groot-grouped-seed1000 \
  --seed 1000
```

数据下载已完成并验证SHA256。包内为新版XPolicyLab数据，50条演示，置于 rmbench_raw/demo_clean/rearrange_blocks/aloha_agilex。首次因解压层级不匹配产生的12KiB未完成派生目录已另名保留，未用于训练。后台运行信息见下方。

原始数据审计：50条演示、20085帧，轨迹长度396–408；14维状态与动作均有限，三路相机帧数一致。审计详情位于 artifacts/experiments/rmbench-rearrange-blocks-20261026/raw-audit.json。

## 正式启动验收

2026-10-26 21:57:54 +08:00 后台启动，启动时代码 commit=491e7ccda52277b4d33b3a6fa7982deb49359e8d。监督shell PID=3271886，使用独立session，可在本会话结束后继续执行；Stage1正常返回后自动执行Stage2。

- 数据转换成功：50 episodes、20085 frames、1 task，preparation.partial=false；归一化统计有限。
- dry-run确认两阶段batch8/4、steps5000/80000、save_freq均5000。结果存于 artifacts/experiments/rmbench-rearrange-blocks-20261026/dry-run.json。
- 实际Stage1读取50条轨迹、20085帧；可训练参数60,183,320，总参数713,352,600。
- step50：loss=1.585、grad_norm=3.895、lr=2.6e-6（warmup中）、update_s=0.523，均有限。约1.9 step/s，GPU利用率99%；nvidia-smi总占用约9.2GiB，包含桌面进程。
- 两阶段完整日志：/home/hello/lizk/vlas/vla-ttt/outputs/logs/rmbench-rearrange-blocks-evo1-groot-grouped-seed1000.log。
- 输出：/home/hello/lizk/vlas/vla-ttt/outputs/rmbench/rearrange-blocks-evo1-groot-grouped-seed1000-stage1（以及-stage2）。
- 启动信息：artifacts/experiments/rmbench-rearrange-blocks-20261026/launch.json；运行结束写exit-code.txt。无该文件不代表一定存活，须同时检查PID和日志进展。

本记录只验收训练已正常开始，不代表收敛或任务成功率。Stage2时间待实际吞吐确认。
