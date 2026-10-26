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

数据下载已完成并验证SHA256。包内为新版XPolicyLab数据，50条演示，置于 rmbench_raw/demo_clean/rearrange_blocks/aloha_agilex。首次因解压层级不匹配产生的12KiB未完成派生目录已另名保留，未用于训练。后台进程信息待实际启动后补充。
