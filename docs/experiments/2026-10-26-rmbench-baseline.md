# RMBench baseline 实施验证（2026-10-26）

分支 `feat/rmbench-baseline`，基于 master `08fd511`，只在专属 worktree 修改。
本次不启动正式长训练，不合并、不推送。使用 Native 执行并做一次独立整体审阅。

## 已验证

- 独立 policy `.venv`，LeRobot 0.6.1；新增可选依赖 h5py 3.15.1，锁文件更新。
- 官方代码固定 `7d1cdfc7661c273a7a5c5f6126e2d1a7f0a3f4b7`。
- 真实 put_back_block/episode0：原始348帧，next-state对齐后347个训练帧。
  原文件29397700字节，SHA256
  `abb3ca905311025787570dd77a1bbec2986fbce2d4b3bab93cf21c1aba616b3f`。
- 下载镜像 API 403，最终仅通过代理读取 HEAD 元数据，数据正文直连 CDN。
  数据位置 `/home/hello/lizk/datasets/rmbench/raw`，派生 `lerobot-smoke`。
- 两阶段各2步，batch1、离线本地 InternVL3、随机动作头。
  Stage1 可训练60,183,320参数，loss2.168→1.805，记录峰值约2.44GiB。
  Stage2 可训练713,352,600参数，初次loss1.270→1.138，约13.13GiB。
  新增逐步保存后复测Stage2末步1.139；短测数值只说明管线可运行。
- Stage1四步，恢复第2步检查点到第4步；日志确认epoch0/sample2恢复，
  loss第3/4步为1.656/1.497，与未中断运行对应。
- 重载Stage2检查点，输入真实三相机图像和关节状态，输出有限14D反归一化动作。
- 真正启动policy server，通过websocket客户端reset/infer真实观测成功，输出shape=(14,)。
  这项是网络推理验证，不是仿真闭环。
- 来源校验修正后重新完成两阶段短训，并从Stage1第1步检查点恢复到第2步。
- H264视频实际ffprobe为h264/yuv420p，ffmpeg完整解码成功。
- 真实LeRobot轨迹末帧action.shape=(50,14)，action_is_pad计数49，chunk不跨episode。
- 新测试及已有协议、结果、Evo1-GROOT、可迁移配置回归：55 passed。

## 独立审阅与修正

审阅指出两项重要问题，均先增加失败测试后修正：

1. 检查点来源：每次保存写入 `rmbench_run.json`，阶段切换校验Stage1/任务/数据/recipe/
   seed/run-name；resume校验完整身份。Stage2另记录初始化检查点内容哈希。
2. 分片与合并：增加accepted-episode区间，后片重走官方专家筛选；合并检查重复
   episode/seed和模型/协议身份，未完整覆盖标记complete=false。

## 尚未完成的验收

本机未找到CUDA编译器 `nvcc`，尚未安装独立的SAPIEN/CuRobo仿真环境。
官方Torch2.4.1/CuRobo0.7.8配方在当前Blackwell显卡的兼容性也未验证。
因此**没有执行官方专家动作回放或真实闭环 rollout**；单元测试、训练和直接推理
不替代这两项。仿真安装说明如实标记为待验证，不提供伪装成已验证环境的锁文件。

资源清单：九任务zip约11.77GiB；embodiments/objects约1.26GiB，未全量下载。
正式训练前须完整准备选定任务数据，确认逐任务轨迹数量和独立统计，再验收闭环。
官方协议默认每任务100个专家可行episode，九任务共900；不是RoboMME的800协议。

## 实施裁定

- Ruling: 初始预算沿用LIBERO单任务5k/80k，以每任务曝光量扩展多任务。
  这是待实验验证的本地配方；代价是九任务Stage2达720k步，须先dry-run核对成本。
- Ruling: 仿真缺少编译工具链时，不跳过官方专家筛选，也不修改系统CUDA环境来冒充完成。
  代价是真实回放/闭环及最终环境锁定仍待完成。
- Ruling: 当前支持多个独立单卡工作站，拒绝未经验证的DDP启动。
  代价是暂不提供单个任务的多卡训练能力。

使用命令见 `docs/rmbench-baseline.md`。短测产物不进入Git。
日志保存于 `artifacts/rmbench-smoke/logs/`；保留最终
`lineage-stage2/checkpoints/000002/pretrained_model`（约2.7GiB）。
已清理本次生成的其余约30.71GiB权重/优化器状态；保留小型配置与日志，
清理过的测试目录不再是可恢复训练的完整检查点。

## 后续环境验收

本记录中的 nvcc/真实仿真阻塞已解决，详见 [独立环境实际验收](2026-10-26-rmbench-environment.md)。
