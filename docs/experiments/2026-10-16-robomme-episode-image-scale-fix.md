# RoboMME episode 图像尺度修复与训练影响（2026-10-16）

## 已确认的问题

旧版 `RoboMMEMemoryDataset` 返回 uint8 [0,255] 相机图像，episode 训练路径未除以 255；InternVL 的 resize 假定 [0,1]，因而使图像几乎全白。T2MEM RoboMME run01 的真实样本诊断中，64 张图像平均 99.82% 像素通道饱和为白色。详细证据见 `artifacts/analysis/t2mem-robomme-diagnosis-20261016/diagnosis.md`。

这属于训练数据路径错误，不是 TTT 更新规则专属问题。前后向能运行和 loss 下降都不能验证图像输入语义正确。

## 修复

`EpisodeSource.events` 在图像增强前，将所有 `observation.images.*` 的 uint8 张量转为 float32 并除以 255；已为浮点的图像保持原值。覆盖 demonstration、执行决策和执行中点，适用于该入口支持的全部架构、交替/联合训练、单卡/多卡。状态、动作、标签掩码、时钟与原始数据集不改变。

回归测试比较 uint8 数据与对应浮点评测输入在事件出口和 InternVL resize 后的严格一致性，并验证增强收到 [0,1]、浮点数据没有重复 /255、原始数据未修改。

## 训练影响表

以下判断针对当前仓库提供的入口和默认数据加载设置；“不受影响”仅指本次 uint8 尺度错误，不代表模型效果已达标。

| 模型与数据 | 当前入口/路径 | 本次问题 | 处理 |
| --- | --- | --- | --- |
| Evo1-GROOT + LIBERO | LeRobot 常规训练器 | 不受影响 | 保留 Stage1/Stage2 |
| Evo1-GROOT + RoboMME | `vla-ttt-train-robomme` → LeRobot 常规训练器 | 不受影响；通用循环已转换 uint8 | 保留 Stage1/Stage2 |
| T2MEM + LIBERO | episode，默认浮点图像 | 不受影响 | 无需因此重训 |
| T2MEM + RoboMME run01 | episode，uint8 图像 | 已确认受影响 | 修复后从正常 Evo1-GROOT RoboMME Stage1 重训 Stage2 |
| AQMem + LIBERO | episode，默认浮点图像 | 不受影响 | 可继续训练 |
| AQMem + RoboMME | 旧版 episode，uint8 图像 | 按相同代码路径受影响 | 停止旧进程，更新后从正常 Stage1 开新 run |
| MAC/MAG/MAL + LIBERO | episode，默认浮点图像 | 不受影响 | 无需因此重训 |
| MAC/MAG/MAL + RoboMME | 旧版 episode，uint8 图像 | 同样受影响（若已经训练） | 更新代码后启动；已有受影响 Stage2 需要重训 |

旧 `train_titans` 转发到 `train_robottt`；该入口直接创建 LeRobotDataset，未开启 return_uint8，默认浮点图像，不存在同一条已确认的触发链。自定义入口、数据集或加载参数需要按实际 dtype 判断，不应仅凭模型名认定受影响。

本机实际检查了 T2MEM run01 的 run.json 所记录提交 `6adff693`；AQMem 远端进程未直接访问，上表依据用户确认它们使用相同旧 master 代码及仓库训练入口。任一模型若从已污染的 T2MEM/AQMem 检查点继续初始化，也不能视为干净训练。

## 其他机器的处理方式

1. AQMem RoboMME 旧进程应停止；更新磁盘代码不会修复正在运行、已经导入模块的 Python 进程。
2. 将本修复同步到服务器，确认 `EpisodeSource.events` 在增强前执行 uint8 → float32 /255。
3. 用正常 Evo1-GROOT RoboMME Stage1 检查点初始化，使用全新输出目录；不要 `--resume` 受影响的 AQMem/T2MEM Stage2。
4. 无需重新训练正常的 Evo1-GROOT Stage1。当前受影响结果和日志保留为诊断材料，不作为正常架构效果对照。
5. 重开长训练前先跑回归检查、小数据过拟合和短闭环。修复输入错误并不保证解决所有控制或记忆学习问题。

本修复不自动停止远端任务、不修改已有权重，不替用户启动正式重训。代码需同步后远端才能使用修复。

## 本次验证结果

- 新增两个回归用例在修复前均因 uint8/float 不一致失败，修复后通过；相关数据、episode 引擎与 RoboMME 测试 35 passed。
- 全量测试在沙箱内 255 passed、7 skipped；4 项因本地 socket / 构建缓存权限失败。将这 4 项在沙箱外重跑，全部通过（4 passed）。合计 259 个用例通过，7 个按条件跳过。
- 真实 RoboMME 数据：4 个 episode 的 demo（存在时）、首个决策与中点，共 20 张相机图像；经保存的 preprocessor 和 InternVL resize 后，与正确浮点路径逐像素完全一致。平均饱和白色通道比例恢复至 0.0897%。记录：`artifacts/analysis/image-scale-fix-20261016/real-data-result.json`。
- Ruff 与 `git diff --check` 通过。本次真实数据验证不需要 GPU，没有进行新的正式训练或成功率评测。
