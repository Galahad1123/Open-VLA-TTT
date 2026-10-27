# 跨机器训练快速开始

适用于 LIBERO 的 T²Mem / AQMem / MAC / MAG / MAL，以及 RoboMME 的 T²Mem episode-parallel 训练。
RMBench 直接看 [复制数据、配置路径、启动训练](#rmbench复制数据配置路径启动训练)。
每台机器只改 `configs/machine.local.toml` 中的路径，所有机器使用同一代码版本和 recipe。

**当前 LIBERO episode-parallel 训练协议仅支持 `HuggingFaceVLA/libero` 的下述固定版本。
`lerobot/libero` 是另一个仓库，不是可直接替换的下载源。** 两个导出的存储和元数据结构可能不同；
已确认目标机器的 `lerobot/libero` video-backed 导出缺少本入口需要的 episode 级 `tasks`。
不通过猜测任务标签或改写元数据来绕过这一差异，以保持与共同 Stage1 和 baseline 的数据协议一致。

## 1. 安装并准备资源

以下三种安装方法任选其一。

### 方法一：按原锁文件从官方源安装

检出团队约定的 Git commit/tag，在仓库根目录执行：

```bash
uv sync --frozen
cp configs/machine.example.toml configs/machine.local.toml
```

主环境使用 Python 3.12 或 3.13；依赖按 `uv.lock` 安装，不复制其他机器的 `.venv`。

### 方法二：通过清华 PyPI 镜像安装导出的依赖清单

`uv.lock` 包含官方源的文件直链；给 `uv sync --frozen` 设置镜像不能可靠地替换这些下载地址。
仓库提供从主锁文件导出的 `requirements-training.txt`，包含默认训练环境和 dev 依赖的
精确版本、平台条件及 SHA256，不包含可编辑的本地项目，也不包含独立的 LIBERO/ROBOMME 评测环境。
此方法保留 `uv.lock` 作为唯一依赖版本来源，不手改下载 URL、不重新解析版本。

在新机器的仓库根目录执行（已有环境正在训练时不要执行同步安装）：

```bash
uv venv --python 3.12 .venv
uv pip sync --python .venv/bin/python requirements-training.txt \
  --default-index https://pypi.tuna.tsinghua.edu.cn/simple \
  --require-hashes
uv pip install --python .venv/bin/python --no-deps \
  --default-index https://pypi.tuna.tsinghua.edu.cn/simple \
  -e packages/eval-client -e .
uv pip check --python .venv/bin/python
cp configs/machine.example.toml configs/machine.local.toml
```

第三方依赖必须匹配锁定的版本和文件哈希；镜像缺少文件或校验不符时安装会失败，
不要删除版本约束或关闭校验。本地两个 editable 包使用当前检出的源码，
其隔离构建依赖（如 hatchling）不在这份运行依赖哈希清单内。
Python 解释器下载不走 PyPI 镜像；不同系统、Python 版本或架构会按条件选择不同依赖和 wheel。
跨机器复现建议统一使用 Python 3.12、Linux x86_64 和同一 Git commit。

安装后，下文命令中的 `uv run --frozen` 改为 `uv run --no-sync`，避免再次按照官方源锁文件同步。
例如：

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run --no-sync python -m lerobot_policy_vla_ttt.training.train_episode_parallel \
  --machine configs/machine.local.toml --run-name t2mem-libero-seed1000-run01 \
  --architecture t2mem --recipe configs/ttt/t2mem/libero/recipe.json
```

每次团队更新 `uv.lock`，同步重新生成并提交清单；不要单独维护第二套版本：

```bash
uv export --frozen --no-emit-local -o requirements-training.txt
```

相关说明：[uv 导出与安装命令](https://docs.astral.sh/uv/reference/cli/)、
[uv 自动同步](https://docs.astral.sh/uv/concepts/projects/sync/)。

### 方法三：备份锁文件后替换下载域名

以下命令已由使用者在目标机器验证可行。在仓库根目录执行；首次备份前确认
`uv.lock` 是仓库原版，且没有需要保留的同名 `uv.lock.backup`，避免重复执行覆盖原始备份：

```bash
cp uv.lock uv.lock.backup
sed -i 's#https://files.pythonhosted.org#https://pypi.tuna.tsinghua.edu.cn#g' uv.lock
uv sync --frozen
```

这是 Linux/GNU sed 的写法。替换只改变下载域名，保留包版本、文件路径和 SHA256；
`--frozen` 不重新解析依赖，也不关闭文件哈希校验。
但此方法依赖镜像提供与官方相同的文件路径，不能保证所有文件或未来镜像布局都适用。
它也没有修改索引地址或构建依赖的来源，因此不保证所有网络请求都走清华源。
若下载失败或校验不符，不要关闭校验，恢复原锁文件后改用方法一或方法二。

安装完成后恢复仓库原锁文件（不会修改已安装的环境）：

```bash
cp uv.lock.backup uv.lock
uv pip check --python .venv/bin/python
```

后续训练使用 `uv run --no-sync`，参数同方法二，避免重新按官方直链同步。
不要提交替换后的锁文件或 `uv.lock.backup`；团队仍以原版 `uv.lock` 为版本依据。
机器路径配置按下文填写，已有环境正在训练时不要执行上述安装命令。

### 检查 CUDA 并准备模型、数据

当前主锁文件包含 Torch 2.11 / CUDA 13 相关运行库，目标 GPU 和驱动需要兼容。
可以先检查 CUDA，再进行下方短程训练验收：

```bash
uv run --frozen python -c 'import torch; print(torch.__version__, torch.version.cuda); print(torch.ones(1, device="cuda") + 1)'
```

提前复制或下载以下完整资源，资源本身不进 Git：

| 配置字段 | 内容 |
| --- | --- |
| `assets.initial_checkpoint` | LIBERO 初始化 `pretrained_model/`（T²Mem 默认共同 Stage1 5k；AQMem 联合实验选择 Stage2 20k/30k），包含权重、config、全部 processor/归一化文件及 `train_config.json` |
| `assets.robomme_initial_checkpoint` | RoboMME Stage1 的 `pretrained_model/`，需单独复制，不可使用 LIBERO 检查点 |
| `assets.robomme_dataset` | 完整 `lerobot/robomme`，revision `92796abbc67db4ea6447e5031d04bf47aa3770bb`，保留 demo 和执行帧 |
| `assets.vlm` | 完整 `OpenGVLab/InternVL3-1B-hf` 快照，revision `014c0583a0d4bedf29fbe2dbff4f865eb998e171`，包括 tokenizer、配置和权重 |
| `assets.libero_dataset` | 完整 `HuggingFaceVLA/libero`，revision `86958911c0f959db2bbbdb107eb3e17c5f9c798e`，包含 `meta/` 和全部数据分片 |
| `storage.runs_root` | 可写的训练产物根目录 |
| `storage.hf_home` | 本机 Hugging Face 缓存目录 |

模板全部使用绝对路径示例，替换为目标机器的实际路径。
复制 HF snapshot 时必须保留可用的 blobs 链接，或者复制链接所指的真实文件。
数据集不能使用早期 smoke 的部分下载；资源目录在训练期间保持只读，不在其中放日志。

### 通过 Hugging Face 镜像下载数据集和 VLM

在目标机器执行。下载工具可以安装在独立的下载环境（例如 conda 的 base 环境）中；
不要在已按锁文件安装的训练 `.venv` 中执行 `pip install -U`，以免改变训练依赖版本。

```bash
pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
```

`HF_ENDPOINT` 只影响当前终端及其子进程的 Hugging Face 请求，与前面的清华 PyPI 源是两件事。
以下命令使用 `~/datasets/LIBERO` 和 `~/models/InternVL3-1B-hf`，可以按磁盘位置修改。
为与共同训练资源保持一致，命令显式指定上表中的 revision；省略 `--revision` 会下载仓库当前默认分支，
不保证与本机已有资源相同。

先完整下载 **HuggingFaceVLA/libero** 数据集：

```bash
hf download \
  HuggingFaceVLA/libero \
  --repo-type dataset \
  --revision 86958911c0f959db2bbbdb107eb3e17c5f9c798e \
  --local-dir ~/datasets/LIBERO
```

再完整下载 **OpenGVLab/InternVL3-1B-hf** 模型（包括 tokenizer、配置和权重）：

```bash
hf download \
  OpenGVLab/InternVL3-1B-hf \
  --revision 014c0583a0d4bedf29fbe2dbff4f865eb998e171 \
  --local-dir ~/models/InternVL3-1B-hf
```

先查看下载清单和体积时，在同一命令末尾加 `--dry-run`；确认后去掉它执行下载。
下载中断后重新运行同一命令即可继续，保留下载目录中的 `.cache/huggingface` 元数据。
若镜像不可用，可在当前终端执行 `unset HF_ENDPOINT`，再运行相同命令从官方源下载。
请使用一个专用于该仓库的新目录，不要覆盖已有 `lerobot/libero` 目录，以免残留文件混入。
目录名 `LIBERO` 不决定数据来源，仓库 ID 必须是 `HuggingFaceVLA/libero`。

共同 Stage1 5k 的 `pretrained_model/` 仍需从已有训练机器复制；上述 VLM 下载不能替代该检查点。
然后更新 `configs/machine.local.toml` 的对应字段（以下假设用户名为 `hello`，请改成真实绝对路径）：

```toml
[assets]
libero_dataset = "/home/hello/datasets/LIBERO"
vlm = "/home/hello/models/InternVL3-1B-hf"
```

这只是字段示例，保留本机配置中其他模型和存储字段。
入口默认的 `--dataset-repo` 和 `--revision` 已对应上面的仓库和版本，不要改成 `lerobot/libero`。
这两个参数记录所声明的来源；它们不会把本地目录中的另一份数据自动替换为正确版本。

若此前因 `KeyError: 'tasks'` 失败，换成正确数据目录后重新执行两步检查，
并使用新名称，例如 `--run-name t2mem-seed1000-smoke-v2`。该初始化失败的运行不能用于 `--resume`。

### RoboMME 数据下载与机器配置

使用上文安装的 `hf` 工具，在目标机器完整下载固定版本：

```bash
export HF_ENDPOINT=https://hf-mirror.com
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
hf download lerobot/robomme \
  --repo-type dataset \
  --revision 92796abbc67db4ea6447e5031d04bf47aa3770bb \
  --local-dir ~/datasets/robomme
```

本机核验的格式为 LeRobot v3.0，1600 episodes、768897 帧、10 FPS。
不要删除 demo 帧或使用 smoke 子集。RoboMME 的来源是 `lerobot/robomme`，
与 LIBERO 使用 `HuggingFaceVLA/libero` 的约定不同。

将以下字段加入现有 `configs/machine.local.toml` 的 `[assets]` 段，勿重复创建该段：

```toml
robomme_dataset = "/home/hello/datasets/robomme"
robomme_initial_checkpoint = "/home/hello/models/robomme-stage1-5k/pretrained_model"
```

路径按本机修改。RoboMME Stage1 检查点需单独复制，必须与 RoboMME 的 8 维动作匹配；
共享的 `assets.vlm` 指向上文下载的 InternVL 模型。
`--benchmark robomme` 自动选择这两个字段，LIBERO 继续使用 `libero_dataset` 和 `initial_checkpoint`。
显式 `--dataset-root`、`--checkpoint` 优先；RoboMME 缺少配置时不会回退到 LIBERO 资源。
输出、日志和 HF 缓存仍使用同一 `[storage]` 约定。

在仓库根目录做配置检查（方法二、三安装的环境使用 `--no-sync` 替换 `--frozen`）：

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run --frozen python -m lerobot_policy_vla_ttt.training.train_episode_parallel \
  --machine configs/machine.local.toml --run-name robomme-t2mem-seed1000-check \
  --benchmark robomme --architecture t2mem \
  --recipe configs/training/robomme-t2mem-episode.json --dry-run
```

将 `--dry-run` 改为 `--stop-after 2` 并使用新的 run name 做两步检查；
正式训练去掉两者并另取 run name。恢复时用 `--resume /绝对路径/step-XXXXXXXX`
替代初始化检查点，仍使用新的 run name；入口从检查点恢复 benchmark，按当前机器配置定位数据和 VLM。
详细协议与验证范围见 [RoboMME 训练记录](experiments/2026-10-13-robomme-observation-memory-training.md)。

## 2. 检查配置

```bash
uv run --frozen python -m lerobot_policy_vla_ttt.training.train_episode_parallel \
  --machine configs/machine.local.toml --run-name t2mem-seed1000-check \
  --architecture t2mem --recipe configs/ttt/t2mem/libero/recipe.json \
  --dry-run
```

这一步只读取配置和小型元数据，不加载模型、初始化 CUDA、创建运行目录或读取全部权重/数据。
输出 `asset_validation=metadata_only`，不代表资源完整或 GPU 训练已经通过。
`recipe.json` 显式保存当前约定的 TBPTT=8 默认值。
`recipe-72gb.json` 是已记录资源验收的 TBPTT=64 候选，需主动选择，不替代默认协议。
跨机器对照应显式使用同一 recipe，不根据显存自动改变实验协议。

## 3. 短程检查与正式训练

单 GPU 两步检查（保留完整训练预算与 scheduler）：

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run --frozen python -m lerobot_policy_vla_ttt.training.train_episode_parallel \
  --machine configs/machine.local.toml --run-name t2mem-seed1000-smoke \
  --architecture t2mem --recipe configs/ttt/t2mem/libero/recipe.json \
  --stop-after 2
```

正式训练使用新的 `--run-name` 并去掉 `--stop-after 2`。
两步检查不覆盖第 500 步的 Memory→Policy 切换；新硬件仍需检查双方 optimizer 状态常驻后的显存。

两张 GPU 用相同参数启动：

```bash
CUDA_VISIBLE_DEVICES=0,1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run --frozen python -m accelerate.commands.launch \
  --multi_gpu --num_processes=2 --mixed_precision=no \
  -m lerobot_policy_vla_ttt.training.train_episode_parallel \
  --machine configs/machine.local.toml --run-name t2mem-seed1000-two-gpus \
  --architecture t2mem --recipe configs/ttt/t2mem/libero/recipe.json
```

默认全局 batch=4，可用 1/2/4 个进程；每卡 microbatch=1，不自动扩大 batch 或 LR。
不要启用 FSDP/DeepSpeed；policy 自行使用 BF16 AMP。多节点运行时资源和输出目录必须对各 rank 可见，
本指南的多卡命令针对单机，不能据此视为完成多节点验收。

## 4. 产物位置与恢复

```text
<storage.runs_root>/<run-name>/
  run.json                  # recipe、资源身份、实际路径、代码和运行环境
  metrics.jsonl             # 主进程逐步指标
  logs/rank-0.log           # 各进程的控制台输出与运行异常
  logs/rank-1.log
  step-00000250/
    COMPLETE
    training.pt
    pretrained_model/
```

目录已存在就拒绝启动；每次启动（包括恢复）使用新的名称。进入运行阶段后才创建日志，
参数解析/资源校验错误直接报告到终端。日志同时显示在控制台，不需要额外 `tee`。

恢复时指定完整 `step-*` 目录，recipe 默认从 checkpoint 恢复：

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run --frozen python -m lerobot_policy_vla_ttt.training.train_episode_parallel \
  --machine configs/machine.local.toml --run-name t2mem-seed1000-continued \
  --resume /srv/experiments/vla-ttt/t2mem-seed1000-smoke/step-00000002
```

恢复必须保持 world size、recipe、结构和数据/增强语义相同。
本版新建运行按数据集、VLM 的完整内容 SHA256 校验，允许相同资源换目录；初始化来源也记录内容哈希。
正式启动会流式读取完整资源进行校验，每个 rank 都会校验，不需要 GPU，但可能耗时较长。
忽略隐藏缓存文件；不要往资源目录添加无关文件。完整哈希用于拒绝恢复时的资源变化，
首次运行仍需要操作者按上表准备正确版本，它不替代发布方的资源清单。

旧版 checkpoint 没有完整内容身份，继续使用原来的严格路径校验；不会自动放宽校验迁移。
若旧路径无法保留，可用 `--checkpoint .../pretrained_model` 开始新阶段，明确属于 warm start，
不恢复 optimizer/进度。新 checkpoint 的异构 GPU 恢复也不保证逐位数值一致。

## 配置规则与评测

- CLI 显式 `--checkpoint`、`--dataset-root`、`--vlm-path` 优先于 TOML；`--resume` 不使用默认初始化 checkpoint。
- `--output /absolute/path` 可代替 `--run-name`，两者不能同时指定。
- TOML 支持 `~`；相对路径相对于 TOML 文件目录，CLI 相对路径相对于当前目录。
- 未知字段、无效路径会报错；本机配置被 Git 忽略，recipe 和示例模板提交 Git。
- `hf_home` 在必要时通过一次进程重新启动于库导入前生效；不移动已有缓存。
- 当前机器配置用于新 episode-parallel 入口和 policy server，旧 baseline/序列训练入口保持原 CLI。

评测服务端也接受相同 VLM 配置，checkpoint 必须显式选择：

```bash
uv run --frozen vla-ttt-serve --machine configs/machine.local.toml \
  --checkpoint /srv/experiments/vla-ttt/t2mem-seed1000-continued/step-00002000/pretrained_model \
  --implementation t2mem --benchmark libero
```

仿真端另按 [baseline 指南](baselines.md) 安装 LIBERO 环境并启动 client。
训练语义、验收范围见 [episode-parallel 指南](episode-parallel-training.md)。

## AQMem + LIBERO 联合训练

[AQMem recipe 与命令](../configs/ttt/aqmem/libero/README.md) 使用同一机器配置和产物约定。
本实验从已训练的 Evo1-GR00T LIBERO Stage2 20k（可选 30k）初始化，
不能沿用 T²Mem 的 Stage1 默认检查点而宣称完成了同一实验。
将 `assets.initial_checkpoint` 指向所选完整 Stage2 `pretrained_model/`，
或者通过 `--checkpoint` 显式覆盖；其他资源、缓存、日志、内容哈希与恢复规则保持一致。
联合模式保持 VLM 全量训练，`global_batch_size=8`，混合任务采样；
使用 `configs/ttt/aqmem/libero/recipe.json` 和 `memory.json`。
单 GPU 实测及 CPU/Gloo 证据见 recipe 文档，多 GPU CUDA/NCCL 仍待目标机器验收。

## RMBench：复制数据、配置路径、启动训练

同一份完整 LeRobot 数据支持单任务、任意任务组合和全部九任务。**不用按任务重新转换或复制数据。**
以下假设 5090 已有本项目 master 和训练环境；直接在原项目目录切换分支，继续使用原 `.venv`。

### 1. 同步代码

本机执行（推送当前 RMBench 分支）：

```bash
cd /home/hello/lizk/vlas/vla-ttt/.worktrees/rmbench-baseline
git push -u origin feat/rmbench-baseline
```

5090 执行（进入已有项目目录）：

```bash
cd /你的项目路径/vla-ttt
git fetch origin
git switch feat/rmbench-baseline
git pull --ff-only origin feat/rmbench-baseline
uv sync --frozen
```

首次切换会跟踪远端同名分支；已有这个分支也可以执行同一组命令更新。
`uv sync` 在已有环境中同步锁定依赖，包含 RMBench 所需的 h5py。评测环境不参与训练。

### 2. 复制数据

本机完整训练数据在：

```text
/home/hello/lizk/datasets/rmbench/lerobot-all
```

在 **5090 上**执行，替换 `本机IP`；本机 SSH 用户为 `hello`：

```bash
mkdir -p "$HOME/datasets/rmbench"
scp -r hello@本机IP:/home/hello/lizk/datasets/rmbench/lerobot-all "$HOME/datasets/rmbench/"
```

这一个目录包含九任务全部数据和元数据，必须完整复制。训练不需要原始 ZIP、HDF5、RMBench 官方仓库或仿真资产。

VLM 直接使用 5090 上原有的完整 InternVL3-1B-hf 目录。如果没有，本机模型位于：

```text
/home/hello/.cache/huggingface/hub/models--OpenGVLab--InternVL3-1B-hf/snapshots/014c0583a0d4bedf29fbe2dbff4f865eb998e171
```

### 3. 配置路径

在 5090 项目下新建 `configs/machine.rmbench.local.toml`，只填以下路径（换成该机的绝对路径）：

```toml
[assets]
vlm = "/home/你的用户名/models/InternVL3-1B-hf"
rmbench_dataset = "/home/你的用户名/datasets/rmbench/lerobot-all"

[storage]
runs_root = "/home/你的用户名/experiments/rmbench"
```

`vlm` 指向已有模型目录，里面应有 `config.json` 和权重文件。训练默认离线，不会重新下载模型或数据。

### 4. 开始训练

单任务（两阶段依次执行，每 5,000 步保存）：

```bash
bash scripts/train_rmbench_single.sh \
  --machine configs/machine.rmbench.local.toml \
  --recipe configs/training/rmbench/rearrange-blocks-grouped.json \
  --tasks rearrange_blocks \
  --run-name rearrange-blocks-grouped-seed1000
```

这个 recipe 虽沿用首个实验的文件名，也适用于其他任务；单任务 Stage1 为 5,000 步，Stage2 为 80,000 步。
换任务只改 `--tasks` 和 `--run-name`，不用改数据路径。每个新实验使用新的 run-name。

部分任务联合训练：

```bash
bash scripts/train_rmbench_multi.sh \
  --machine configs/machine.rmbench.local.toml \
  --recipe configs/training/rmbench/rearrange-blocks-grouped.json \
  --tasks rearrange_blocks put_back_block \
  --run-name two-tasks-grouped-seed1000
```

全部任务：将上一条的任务参数换成 `--tasks all`，并换一个 run-name。
多任务步数按任务数增加：两任务 10,000 + 160,000 步，九任务 45,000 + 720,000 步。
加 `--dry-run` 只查看配置；加 `--smoke-steps 2` 则每阶段试跑两步（使用新的 run-name）。

九个任务名：`observe_and_pickup`、`rearrange_blocks`、`put_back_block`、`swap_blocks`、`swap_T`、
`battery_try`、`blocks_ranking_try`、`cover_blocks`、`press_button`。

所有选择共享整份数据的归一化统计；只从选中任务抽取训练样本。
此前运行中的单任务数据和实验保持原样，其统计量与新九任务数据不同，比较实验时应记录这一差异。
断点恢复和评测见 [RMBench baseline](rmbench-baseline.md)；需要仿真评测时再配置
[独立评测环境](../environments/rmbench/README.md)，默认保存 H.264 视频。
