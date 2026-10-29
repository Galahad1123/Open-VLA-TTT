# RoboMME 两任务1～3次重复操作：配对序列训练

175条训练轨迹：PickXtimes 23/25/27条，SwingXtimes 23/31/46条，对应1/2/3次。
每批6个episode，每个任务×次数分层各一条；microbatch=6。
原始语言指令不替换成分层标签。全部轨迹无demo，次数已与完整simple_subgoal序列交叉核验。
生成器 `scripts/prepare_robomme_repeats.py` 接受 `--dataset-root`、`--output`；
重新审计时输出到临时目录，不覆盖已确定预算的recipe。

两组从同一原始30k baseline策略开始，重新建立优化器和调度器；AQMEM memory重新初始化。
VLM冻结，action DiT LR=1e-5，memory慢参数LR=1e-4，warmup100，最低LR比例0.1，
每500步及最终步保存，H20/E10、每5步写入、TBPTT8、官方式增强及原dropout不变。
正式训练步数由AQMEM的30步短测确定，预算90%×17小时，余量留给波动。
取排除前5步后的更新耗时75分位数，加预计检查点保存开销，选择最大100步整数倍。
两组最终步数、采样与调度长度一致；初始2000步配置仅用于短测，正式预算会单独提交。

`diagnostics_every=50` 在第1步和每50步记录到diagnostics.jsonl。
诊断包括各优化器组裁剪前梯度范数、最多4096个固定位置的参数更新估计、
各层tanh(gate)统计、记忆残差/主分支幅度比和最终fast-weight矩阵RMS。
幅度统计排除padding和无动作监督行；是训练状态下的观测，不是独立验证loss。
多卡时梯度是同步后的梯度，特征/状态仅来自rank0；无额外随机数或训练前向。
旧recipe默认关闭诊断。fast-weight状态统计当前覆盖静态episode microbatch路径。

## 远程启动

5090-1运行aqmem，30k路径：
`/home/hello/lizk/models/robomme-six-task-stage2-30k/pretrained_model`

5090-2运行baseline，30k路径：
`outputs/robomme-six-official-seed1000-stage2/checkpoints/030000/pretrained_model`

先git pull --ff-only，在仓库根目录使用训练环境Python：

```bash
.venv/bin/python scripts/run_robomme_repeat_experiment.py \
  --kind aqmem --machine configs/machine.local.toml \
  --checkpoint /home/hello/lizk/models/robomme-six-task-stage2-30k/pretrained_model \
  --run-name robomme-repeat-two-aqmem-probe-run01 --probe
```

baseline机器将kind、checkpoint、run-name对应修改。`--probe`只跑30步；
正式运行删除该选项，使用新的run-name。`--dry-run`只打印实际命令，不创建目录。
脚本固定单卡、离线、线程数2和PYTHONHASHSEED=0，日志与进程信息保存于
`<runs_root>/launchers/<run-name>.{json,log}`，训练数据位于`<runs_root>/<run-name>`。
已有输出或启动记录会被拒绝，脚本不自动重启、不停止其他任务。

查看：`tail -f outputs/launchers/<run-name>.log`。
启动记录中有parent_pid和child_pid；complete表示训练命令成功退出。
最终检查点还需存在COMPLETE；短测完成后timing_summary.json记录资源和速度。
恢复使用原train_episode_parallel入口的--resume指向完整step目录，以及新的run-name；
不使用本启动器重新加载30k来冒充恢复。

## 后续评测（本次不自动启动正式评测）

计划文件：configs/training/robomme-screen/repeat-two-evaluation.json。
val使用全部匹配场景，不传--limit-per-task；train固定每分层前10条，共60条。
使用vla_ttt_eval.subset，seed1000，max-steps1300，保存视频和任务/次数分层汇总。
先评共同30k基座，再评中间/最终检查点；同种子不保证物理仿真逐位复现，需核对场景指令。
