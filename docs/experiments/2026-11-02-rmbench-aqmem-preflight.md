# RMBench AQMem 双卡短测与正式配方

基于分支feat/rmbench-aqmem，源baseline为远端三任务stage2/085000。
本地90项episode测试通过（分布式通信测试需沙箱外本地端口），另新增RMBench dry-run路由测试。

远端095-4090x2，短测路径：
`/root/autodl-tmp/outputs/rmbench-aqmem-preflight-20261102`。
三个memory更新52.20/42.28/48.61秒，一个policy更新39.66秒；完整周期含两次保存195秒。
TTT loss约.00146–.00192；DiT约.00289。不同阶段dropout/样本不同，不直接解释为性能升降。
TTT结束保存逐张量比对baseline：733个原张量全部不变，无缺失键。
实际冻结验证及断点恢复另见运行日志；本记录不代表任务成功率验证。

正式配置four-rounds-24h.json：K=100，四轮300memory+100policy，总1600次完整episode batch更新（global4）。
按短测均值计算约20.3h，考虑约10%吞吐波动约22.4h；余量覆盖初始化、保存和轨迹长度差异。
循环安全停止上限23.5h，不含初始化；短测初始化约2分钟、单次保存4–5秒，距离24h留有余量。
这是时间预算上限而非保证恰好跑满24h。八个阶段边界均保存，训练阶段按各自累计更新余弦衰减，不重启调度。

正式模型必须重新从baseline初始化，不能从短测权重续训。全部资源使用本地路径和离线标志。
逐episode梯度累积为数值参考，保留完整episode记忆；未引入RoboMME分支的实验性模型批处理改动。
对照E24评测和最终AQMem评测另计时间，训练期间不混跑GPU评测；视频默认h264。

补充真实权重验证：memory→policy后119个原动作头张量变化，VLM和新增memory张量零变化。
双卡从step4恢复至step5成功，累计memory4/policy1，LR4.22343665e-4，未重新warmup；loss.00164285，耗时49.71s。
每个完整检查点约4GB，八阶段约32GB；恢复短测结束预计剩余约60GB，足够正式保存。

正式启动：源码快照655fd6d，远端`/root/autodl-tmp/runsources/rmbench-aqmem-655fd6d`，复用`/root/Open-VLA-TTT/.venv`。
训练输出`/root/autodl-tmp/outputs/rmbench-aqmem-four-rounds-20261102`；
启动器日志`/root/autodl-tmp/outputs/rmbench-aqmem-four-rounds-20261102-launch/run.log`，同目录记录pid、started_at、结束后的exit_code/finished_at。
使用四轮配方重新加载原85k baseline，未续用短测权重。后续E24基线对照与正式模型评测尚未执行，不在训练期占用GPU。
