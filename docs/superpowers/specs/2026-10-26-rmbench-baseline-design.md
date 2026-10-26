# RMBench Evo1-GROOT baseline 训练与评测设计

状态：设计已获用户批准；训练、数据转换与评测客户端已实现并短测。
真实仿真环境、专家回放和闭环 rollout 尚待验收，详见 2026-10-26 实验记录。

## 目标与已确认范围

在 `feat/rmbench-baseline` 分支中接入本地 Evo1-GROOT 的 RMBench 训练—评测管线。
同时提供单任务、多任务入口，支持把独立实验分发到不同工作站。复用本地模型、
LeRobot 训练能力、processor 和现有 server-client 协议；首轮不引入 TTT。
此次交付包括数据准备、两阶段训练、检查点加载、真实仿真短测和可迁移运行文档。
长时间正式训练由用户分发启动，短测不能作为成功率结论。

用户约束：机器资源可迁移、评测默认 H.264 视频、recipe 优先参考官方、
HF 镜像及清华 PyPI 优先，大文件下载不经过终端代理。

## 实现方式

采用一套公共训练实现、两个用户入口。单任务选择一个 task；多任务选择显式列表
或固定版本的 all 清单。两者共用 schema、chunk 构造、归一化、初始化和评测。
不分别复制两套 trainer，否则修正动作或数据语义时容易发生分歧。
不整体迁入上游训练框架，否则会扩大现有 LeRobot 项目的维护和依赖范围。
上游环境及必要评测控制逻辑固定版本保留，适配层负责与本地策略通信。

## 官方依据与版本约束

截至本次只读调研，依据如下；实施时必须解析不可变 commit/revision 并记录：

- https://github.com/RoboTwin-Platform/RMBench
- https://raw.githubusercontent.com/RoboTwin-Platform/RMBench/main/data/_download.py
- https://raw.githubusercontent.com/RoboTwin-Platform/RMBench/main/assets/_download.py
- https://raw.githubusercontent.com/RoboTwin-Platform/RMBench/main/scripts/eval_policy.py
- https://raw.githubusercontent.com/RoboTwin-Platform/RMBench/main/env_cfg/eval/all_tasks.yml
- https://robotwin-platform.github.io/doc/usage/ACT.html

原始资源来自 `TianxingChen/RMBench` 数据集，训练包为
`data/<task>/demo_clean.zip`，仿真资源包括 `embodiments/**` 和 `objects/**`。
目前上游 all_tasks.yml 列出十个任务，而项目页介绍九个任务。
实现前核对论文任务、实际下载清单和可运行环境，分别记录论文集合和扩展集合；
不得把不一致集合的总成功率直接称为论文复现。all 是本项目版本化清单，不能运行时漂移。

## 数据、观测和动作契约

优先将官方原始 HDF5 转成可复用的 LeRobot 本地数据格式，保留原始轨迹及任务标识。
原始数据只读，派生数据单独存放，记录输入版本、转换配置与产物身份。
使用官方 decode_image_bit 逻辑兼容旧版 JPEG 与 XPL-RGB1，统一得到 RGB；
解码后不能再无条件交换颜色通道。训练与推理执行同一缩放规则。

相机、状态维度、关节排列、夹爪定义、动作时序和控制频率以所选 embodiment 的官方
数据/环境为准，通过真实样本和动作回放核实。不得继承 RoboMME 的 8 维动作假设。
当前 server 只接受 image/image2，需增加 RMBench 独立 schema，同时保留已有基准校验。
读取官方可用相机，不能为了满足旧接口静默丢弃相机。

无记忆 baseline 使用当前观测、机器人状态和任务语言；不使用特权子任务、
专家规划结果或未来观测。是否存在演示前缀须按实际数据核实，不能照搬 RoboMME 处理。
训练 chunk 不跨 episode，末尾 padding 不进入 loss。归一化只使用选定训练数据，
统计随 checkpoint 保存，评测不重新拟合。

## 两种训练入口

拟提供 `train_rmbench_single.sh` 与 `train_rmbench_multi.sh`，调用同一 Python trainer。
共享参数：machine、recipe、run-name、seed、stage、dry-run，以及短测/恢复入口。
任务名称和任务列表保存在 recipe/run manifest；重复和未知任务直接报错。

初始化提案：本地通用 InternVL3 + 随机动作头；Stage1 冻结 VLM，Stage2 加载本次
Stage1 的权重并解冻。两阶段切换重新创建优化器；resume 是同阶段完整续训，二者区分。
不隐式加载 LIBERO/RoboMME 已训练的策略权重。

官方数据处理和控制语义优先；优化器及训练预算参考官方 ACT/DP/VLA 实际配置，
核对 steps/epochs、batch、horizon、执行长度、图像增强和 LR 调度后确定数值。
不能将 ACT 的训练步数直接当作 Evo1-GROOT 已验证配方。
多任务默认提案为先均匀选任务、再采样该任务的执行帧，使用公共归一化统计；
如采用该方案，与官方采样的差异必须明确记录。单任务与多任务的样本曝光预算同时报告。
本设计不预设未经源码核验的数值 recipe。

## 仿真评测与视频

环境独立于 policy Python 环境，使用现有 websocket server-client 协议。
尽量调用固定版本官方评测控制逻辑并提供远程策略 adapter，避免重新定义 episode。
核对并保留官方 seed 策略、无效场景重试、成功判断、动作步数限制及语言生成语义。
环境异常单独记录，不能静默计为策略失败，也不能无限重试。

单任务模型评测对应任务，多任务模型逐任务评测；提供逐 episode、逐任务和整体结果。
支持明确的 episode 分片与合并，检测重复 episode 和 checkpoint/protocol 不匹配。
每次评测记录模型、代码、数据协议、环境版本与 seed，运行期间使用固定代码版本。

默认每个 episode 保存 MP4：libx264、yuv420p、faststart。录制流式写入，
避免整段视频驻留内存；编码成功后才发布最终文件名，失败保留诊断信息。
验证 ffprobe 元数据与实际解码。输出同时包含成功率、耗时、步数和视频路径。

## 可迁移资源和工作站运行

延续 `configs/machine.local.toml`：扩展 RMBench 原始/派生数据、上游源码、
仿真 assets 等明确路径字段；共享 VLM、runs_root、hf_home。模板不包含本机硬编码路径。
recipe 只描述实验；machine 描述资源路径。源数据、权重、缓存和视频不进入 Git。
每个工作站使用同一项目 commit、上游 commit、数据 revision、recipe 和依赖版本。
独立 run-name、输出目录和端口；不同工作站不要求同一绝对路径。

下载默认 HF_ENDPOINT=https://hf-mirror.com；清华源用于 Python 包。
下载子进程清除大小写 HTTP/HTTPS/ALL_PROXY，不改全局代理配置；
如发生重定向也核对实际传输路径。大文件先列清单和体积，支持断点续传与完整性验证。
资源准备完成后训练/评测离线，缺资源时报错，不隐式联网。
独立 RMBench 环境按官方兼容版本锁定，不修改正在使用的主训练环境。

## 验收与交付顺序

1. 固定上游与资源清单，明确官方 recipe 和数据/动作协议；记录下载体积。
2. 数据转换测试：RGB、维度、时间对齐、边界 mask、统计只来自训练数据。
3. 单/多任务采样测试及 dry-run；确认不同机器仅需替换 machine 文件。
4. 真实数据两阶段短训：有限 loss、正确冻结/解冻、保存与重载、resume。
5. 官方演示动作回放，核实观测和控制映射；偏差与环境版本限制如实记录。
6. 本地模型通过 server-client 真实 rollout；H.264 可解码，统计和视频对应。
7. 运行已有 LIBERO/RoboMME 相关回归测试，交付两种正式命令与实验记录。

管线验收不要求随机/短训策略成功；正式研究验收另需已训练模型的任务成功率。
如某个工作站缺 GPU/仿真依赖，将未完成验证明确列出，不以 mock 代替真实结果。
