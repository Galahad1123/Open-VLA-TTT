# RoboMME Evo1-GROOT 正式训练前检查

## 范围与结论

在本地master工作目录实施，未修改foundation worktree。未启动正式训练，
未下载模型、数据或环境资源；通过uv本机缓存创建`environments/robomme/.venv`。
policy、LIBERO、RoboMME三个环境独立。训练准备、真实数据两阶段短程检查、
16任务仿真初始化及跨环境RPC验证通过，正式任务成功率仍待训练后评测。

## 固定评测协议

- RoboMME commit：`c85e060c830a72571f9517b6dd683d51e154490e`。
- ManiSkill fork：`07be6fbc66350ddca200abfb0a11b692f078f7fd`。
- 4套件16任务；每任务官方train/val/test为100/50/50个episode。
- 48组task/split元数据均核验episode ID及seed存在；逐任务三种split的seed集合无交集。
- 使用MME-VLA实验名义`max_steps=1300`，不是示例300，也不是Challenge1500。
  来源：[固定版本官方evaluation.py](https://github.com/RoboMME/robomme_benchmark/blob/c85e060c830a72571f9517b6dd683d51e154490e/scripts/evaluation.py)。
- 官方wrapper将上限加2，reset消耗1步；实测16任务均如此。因此按wrapper剩余预算，
  最多执行1301次外部step，不另加1300次的较短外层截断。LIBERO保持原步数逻辑。
- 记录env seed、policy seed、difficulty、实际step_limit、成功和终止类型；
  环境报错记为error并中止，不悄悄计成普通失败；timeout与terminated区分。
- `--tasks all --split test --episodes 50`展开16任务共800回合。这是单个policy seed规则下
  的完整测试，不等于复现论文所有多seed重复实验。
- 无记忆输入：最新前视/腕视图、8维state和task_goal；不读demo/history或特权subgoal。

## 数据核验与训练读取

源目录：`/home/hello/lizk/datasets/robomme`；预期revision：
`92796abbc67db4ea6447e5031d04bf47aa3770bb`。离线核验内容和info哈希，未与Hub逐文件比对。

| 项目 | 结果 |
| --- | --- |
| episodes / 全部帧 | 1600 / 768897 |
| demo / 执行帧 | 292040 / 476857 |
| fps | 10 |
| parquet / 引用视频 | 1292 / 48，均存在 |
| 语言变体 | 116，不是benchmark任务数 |
| 每episode执行帧 | 24–1044 |
| state / action | 均8维，全部数值有限 |
| demo边界 | 全部为前缀，与exec_start_idx一致 |
| 解码抽查 | 16个episode执行首/中/尾，共48样本、96相机帧通过 |
| chunk尾部 | 最后一帧查询20动作，19个padding位置正确标记 |

数值、索引、episode边界检查覆盖全部数据；不是全量视频逐帧解码验证。

`baselines/robomme_data.py`复用原始`LeRobotDataset`的视频和时间轴，仅提供执行段训练视图：

1. 为EpisodeAwareSampler提供压缩后的执行帧episode边界；不物理删除demo或重编码视频。
2. 查询未来20个`actions`，保留并重命名`actions_is_pad`，不跨episode取标签。
3. 仅两路图像、state、action、padding mask、task进入训练batch，不传特权信息。
4. 仅用执行帧计算state/action MIN_MAX统计量，随processor保存。

显式入口`vla-ttt-train-robomme`只在当前进程临时替换runner的数据工厂，退出时恢复。
普通`vla-ttt-train`与LIBERO数据路径不变。恢复RoboMME训练也须使用专用入口。

## 两阶段短程检查

执行`bash scripts/train_robomme_evo1_groot.sh --smoke`，各3步；仅smoke的warmup改为1、
save/log间隔缩短，其余关键设置与正式候选一致。

| 项目 | Stage 1 | Stage 2 |
| --- | --- | --- |
| batch | 32 | 16 |
| VLM | 冻结bf16 | 解冻float32权重，bf16 AMP |
| 可训练参数 | 60,156,440 | 713,325,720 |
| 三步loss | 1.277 / 1.255 / 1.192 | 1.259 / 1.173 / 1.198 |
| GPU max_memory_allocated | 18.46 GiB | 26.47 GiB |
| 后两步update时间 | 约2.0秒/步 | 约3.7秒/步 |

forward/backward、优化器更新、checkpoint/processor保存和阶段切换通过。
Stage2确实解冻VLM并新建优化器。3步loss不构成收敛证据。
粗略外推5000+80000步约85小时（3.5天），未包含持续解码瓶颈和checkpoint开销；
需用正式训练稳定区间日志更新估计。

初始化只使用本地通用InternVL3 snapshot和随机GROOT动作头，没有加载LIBERO策略权重。
模型和数据均以HF离线模式读取。

## 环境与server-client验证

- 16任务各test episode0真实reset/render/step通过。
- 独立Python3.11仿真客户端连接根目录Python3.12策略服务，使用本轮Stage2第3步权重。
- PickXtimes、VideoUnmask、PickHighlight、RouteStick各名义30步，实际31步正常timeout，
  无基础设施错误；这不是成功率评测。
- 默认1300协议另跑PickXtimes/test0，正确记录step_limit=1301；环境在178步正常失败，
  因此该回合未验证到1301步自然timeout；短预算自然timeout已经实测。
- 验证后关闭临时8013端口policy服务。
- 最终回归：`tests/unit`及rollout/server-equivalence入口测试共78 passed、2 skipped；
  两个跳过项需要显式checkpoint参数，本轮真实GPU/RPC检查另见上述记录。
  Ruff、shell语法、离线lock一致性、`git diff --check`均通过。

证据位于`artifacts/experiments/robomme-preflight-20261008/`：
`data-audit.json`、`data-decode-audit.json`、`environment-audit.json`、
`rpc-smoke-aligned.jsonl`、`rpc-full-budget-probe.jsonl`。

训练日志：`outputs/logs/robomme-evo1-groot-preflight-20261008-stage{1,2}.log`。
smoke权重：`outputs/robomme-evo1-groot-preflight-20261008-stage{1,2}/checkpoints/000003/`。

## 下一步：正式训练（本轮未执行）

```bash
cd /home/hello/lizk/vlas/vla-ttt
bash scripts/train_robomme_evo1_groot.sh --train
```

正式设置：Stage1 5000步/batch32；Stage2 80000步/batch16；两阶段lr1e-5、warmup1000、
cosine；chunk20/执行10；448图像、关闭增强、dropout0.2、wd1e-3、seed1000。
脚本默认仅smoke，只有显式`--train`开启正式训练；拒绝覆盖已有输出，不自动恢复。
正式Stage1不会加载本次smoke权重。

Stage2每10k保存；20k/40k/60k/80k使用固定官方val episode0–9/任务观察趋势，
候选checkpoint使用完整val进一步比较；test仅用于最终报告，不用于调参。
正式评测示例（先另启对应checkpoint的policy server）：

```bash
HF_HUB_OFFLINE=1 environments/robomme/.venv/bin/python -m vla_ttt_eval.runner \
  --benchmark robomme --tasks all --split test --episodes 50 --seed 1000 \
  --server ws://127.0.0.1:8000 --output outputs/eval/robomme-evo1-groot-test.jsonl
```

本轮没有自动启动全套评测，尚无RoboMME正式成功率。
