# 跨机器训练快速开始

适用于 LIBERO 的 T²Mem / AQMem / MAC / MAG / MAL，以及 RoboMME 的 T²Mem episode-parallel 训练。
RMBench 的 Evo1-GROOT 单/多任务 baseline 见文末独立章节，其启动和恢复规则不同。
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

## RMBench：Evo1-GROOT baseline 跨机器快速开始

本节使用 `train_rmbench_single.sh` / `train_rmbench_multi.sh`，**不是上文的
`train_episode_parallel`**。启动参数、产物和恢复规则以本节为准。
检出包含本节的 `feat/rmbench-baseline` 分支或团队约定的后续合并 commit。
训练不需要安装仿真环境；只有闭环评测需要 `environments/rmbench/.venv`。
不要在已有训练运行的机器上执行依赖同步或覆盖正在读取的数据。

### 1. 新旧工作站统一安装训练依赖

RMBench 训练所需的 h5py 已加入默认依赖，`uv.lock` 和 `requirements-training.txt`
同步维护。新机器首次安装、已有 master 环境更新，都在检出同一代码版本后执行：

```bash
uv sync --frozen
uv pip check --python .venv/bin/python
```

无需额外指定 `--extra rmbench`，旧 extra 保留为空兼容别名。
已有环境会按锁文件同步，不必删除 `.venv` 或手动逐项补包。
这些命令只配置训练环境，不安装 SAPIEN、CuRobo 或仿真 CUDA 编译工具。
有训练运行时不要同步它正在使用的环境；新 worktree 应有自己的 `.venv`。

需要清华镜像时，直接使用本文方法二和已更新的 `requirements-training.txt`，
不再临时导出 RMBench 专用 requirements。锁文件中的官方文件直链不会仅因设置
`UV_INDEX_URL` 就全部改走清华；镜像安装和标准 `uv sync` 对应同一份锁定依赖。
大 wheel 下载时清除该命令的大小写 HTTP_PROXY/HTTPS_PROXY/ALL_PROXY，避免经终端代理传输。

后续直接使用 `.venv/bin/python` 和仓库 shell 入口，不在启动训练时重新同步依赖。
VLM 使用上文固定的本地 InternVL3 快照；本 baseline 不需要 LIBERO/RoboMME 策略权重。

### 2. 下载固定版本的完整九任务数据

论文任务集合固定为 `battery_try`、`blocks_ranking_try`、`cover_blocks`、
`observe_and_pickup`、`press_button`、`put_back_block`、`rearrange_blocks`、
`swap_T`、`swap_blocks`。`all` 仅指这九个任务，不含额外的 `place_block_mat`。

原始训练 zip 共 **12,633,491,826 字节，约11.77 GiB**；这不是包含所有衍生版本的
整个 HF 仓库。解压文件、LeRobot 派生数据、环境和检查点另外占空间。
数据固定 `TianxingChen/RMBench@5d8ff21038fba32ea549e9eee8497aaba9aab5cf`；
仓库清单 `configs/datasets/rmbench-resources.json` 保存路径、大小和 LFS SHA256，
避免镜像元数据 API 限流。保留下载包可用于跨工作站分发和重新校验。

```bash
RMBENCH_ARCHIVES="$HOME/datasets/rmbench/archives"
# 先输出所选文件及总大小。
.venv/bin/python scripts/download_rmbench.py \
  --kind data --tasks all \
  --manifest configs/datasets/rmbench-resources.json \
  --output "$RMBENCH_ARCHIVES"
# 执行完整下载；重跑会复用校验合格的文件并续传 .partial。
.venv/bin/python scripts/download_rmbench.py \
  --kind data --tasks all \
  --manifest configs/datasets/rmbench-resources.json \
  --output "$RMBENCH_ARCHIVES" --download
```

默认源是 `https://hf-mirror.com`，下载器的大文件正文及重定向后的传输绕过代理。
不要同时让两个下载器写同一个 `.partial`。需要官方源回退时显式传入
`--endpoint https://huggingface.co`；仍保留 revision 和 SHA256 校验。
`--manifest` 接受本仓库的 revision/files 对象；下载结束的文件列表记录不是同一种输入格式。

把包解压到**新的**原始数据根目录，避免与早期 smoke 样本混合。
下面同时处理已知的新旧两种包内目录，并校验文件长度和SHA256；遇到其他布局直接报错。
`RMBENCH_RAW` 必须尚不存在，执行中断后的残留目录应另名保留再重试，不覆盖训练数据。

```bash
export RMBENCH_ARCHIVES
export RMBENCH_RAW="$HOME/datasets/rmbench/raw-full"
.venv/bin/python - <<'PY'
import hashlib, json, os, zipfile
from pathlib import Path
manifest = json.loads(Path('configs/datasets/rmbench-resources.json').read_text())
archives, raw = Path(os.environ['RMBENCH_ARCHIVES']), Path(os.environ['RMBENCH_RAW'])
raw.mkdir(parents=True, exist_ok=False)
for row in manifest['files']:
    path = archives / row['path']
    with path.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == row['sha256'], path
    assert path.stat().st_size == row['size'], path
    task = Path(row['path']).parts[1]
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in names)
        files = [n for n in names if not n.endswith('/')]
        if files and all(n.startswith(task + '/aloha_agilex/') for n in files):
            destination = raw / 'demo_clean'
        elif files and all(n.startswith('demo_clean/') for n in files):
            destination = raw / 'data' / task
        else:
            raise ValueError(f'Inspect unexpected ZIP layout: {path}')
        archive.extractall(destination)
    print('Extracted:', task, flush=True)
PY
```

新版布局为 `raw-full/demo_clean/<task>/aloha_agilex/data/episode_0000000.hdf5`；
旧版为 `raw-full/data/<task>/demo_clean/data/episode0.hdf5`。保留 instruction(s) 等附属文件。

### 3. 机器配置与数据转换

准备固定上游源码，数据转换调用其中的官方图像 codec，不需要初始化 XPolicyLab 子模块：

```bash
git clone https://github.com/RoboTwin-Platform/RMBench "$HOME/sources/RMBench"
git -C "$HOME/sources/RMBench" checkout 7d1cdfc7661c273a7a5c5f6126e2d1a7f0a3f4b7
cp configs/machine.example.toml configs/machine.local.toml
```

修改现有 TOML 的字段（下面是路径示例，不要重复声明 `[assets]` / `[storage]`）：

```toml
[assets]
vlm = "/home/hello/models/InternVL3-1B-hf"
rmbench_source = "/home/hello/sources/RMBench"
rmbench_raw = "/home/hello/datasets/rmbench/raw-full"
rmbench_dataset = "/home/hello/datasets/rmbench/lerobot-rearrange-blocks"
rmbench_assets = "/home/hello/datasets/rmbench/assets"
[storage]
runs_root = "/home/hello/experiments/rmbench"
```

完整单任务转换（正式训练不使用 `--max-episodes`）：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
.venv/bin/python -m lerobot_policy_vla_ttt.baselines.rmbench_data \
  --machine configs/machine.local.toml --tasks rearrange_blocks \
  --output "$HOME/datasets/rmbench/lerobot-rearrange-blocks"
```

九任务联合训练时另建派生目录，将 `--tasks` 改为 `all`、`--output` 改为
`$HOME/datasets/rmbench/lerobot-all`。不同任务组合必须分别计算统计，不能用联合数据集
冒充单任务训练。输出必须是新目录，包含 `meta/`、`data/`、`preparation.json`；正式入口拒绝
`partial=true`。可直接复制已完整转换的目录以节省时间；不要只复制 Parquet 而漏掉图像。
跨机器恢复时原样复制 `preparation.json`，其字节哈希用于检查数据身份。

### 4. 单任务两阶段训练

复现当前 Rearrange Blocks 实验：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_rmbench_single.sh \
  --machine configs/machine.local.toml \
  --recipe configs/training/rmbench/rearrange-blocks-grouped.json \
  --tasks rearrange_blocks --run-name rearrange-blocks-evo1-groot-grouped-seed1000 \
  --seed 1000 --dry-run
```

去掉 `--dry-run` 后正式训练。`--smoke-steps 2` 是另一个短测选项，会将 batch 改为1、
每阶段只跑2步并缩短warmup，必须使用新 run-name；它不同于上文 episode-parallel 的 `--stop-after`。
入口自动设置离线模式；Stage1冻结通用VLM，随机初始化动作头，Stage2从本次Stage1权重初始化并
解冻VLM、重建优化器。当前每任务Stage1 batch8×5000步，Stage2 batch4×80000步，
动作头/VLM LR分别1e-4/1e-5，chunk50/执行10。该专用recipe每5000步保存。
全部50条演示参与训练，**不划20%验证集，不自动进行闭环评测**；保存检查点不等于完成验证。

其他任务单独训练可复用同一数值recipe，但必须替换 `--tasks`、`--run-name` 和
`--dataset-root /本任务的完整派生数据集`。recipe文件名不决定训练任务，实际任务由CLI和数据清单核对。

九任务入口示例，仅预览预算：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_rmbench_multi.sh \
  --machine configs/machine.local.toml \
  --recipe configs/training/rmbench/grouped.json \
  --tasks all --dataset-root "$HOME/datasets/rmbench/lerobot-all" \
  --run-name all-evo1-groot-grouped-seed1000 --seed 1000 --dry-run
```

**同一每任务预算乘以9后，实际是Stage1 45000步、Stage2 720000步**，不是全任务合计5k/80k。
通用 `grouped.json` 未设 `save_freq`，默认每10000步保存；若要求5000，复制recipe并添加
`"save_freq": 5000` 后显式选择。多任务先均匀选任务再均匀选有效帧。正式联合预算需另行确定。
当前支持多个工作站各跑独立单卡实验，不使用上文 episode-parallel 的多卡启动命令。

产物位置：

```text
<runs_root>/<run-name>-stage1.rmbench.json
<runs_root>/<run-name>-stage2.rmbench.json
<runs_root>/<run-name>-stage1/checkpoints/005000/{pretrained_model,training_state}/
<runs_root>/<run-name>-stage2/checkpoints/005000/{pretrained_model,training_state}/
```

入口输出到终端，后台日志用本机的 tmux/重定向保存。阶段切换用
`--stage stage2 --checkpoint <stage1的pretrained_model目录>`；同阶段恢复用
`--stage stage2 --resume <stage2的pretrained_model目录>`，其他任务/recipe/seed/run-name参数保留。
**RMBench恢复保持原run-name**，与上文episode-parallel规则不同。
迁移续训应复制完整checkpoint目录（含training_state）、旁侧对应 `.rmbench.json`、原样派生数据和
同一VLM快照，并将产物放在新 `runs_root` 下相同相对位置；只改machine资源路径。

### 5. 独立环境的闭环评测

仿真资产单独下载（约1.26GiB），不能用训练zip替代：

```bash
.venv/bin/python scripts/download_rmbench.py --kind assets \
  --metadata-endpoint https://huggingface.co --output "$HOME/datasets/rmbench/assets" --download
bash environments/rmbench/setup.sh
environments/rmbench/.venv/bin/python environments/rmbench/prepare_runtime.py \
  --machine configs/machine.local.toml --runtime environments/rmbench/runtime
```

安装脚本使用清华源、独立Python3.11环境和用户目录CUDA12.8工具链；当前Blackwell组合已实测，
其他硬件仍需运行 [环境说明](../environments/rmbench/README.md) 中的专家preflight。
评测时两个终端分别启动：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/vla-ttt-serve \
  --machine configs/machine.local.toml --implementation evo1_groot --benchmark rmbench \
  --checkpoint /完整路径/checkpoints/040000/pretrained_model --port 8000
```

```bash
environments/rmbench/.venv/bin/python -m vla_ttt_eval.rmbench_runner \
  --source environments/rmbench/runtime --task rearrange_blocks \
  --uri ws://127.0.0.1:8000 --output artifacts/rmbench-rearrange-eval \
  --seed 0 --episodes 100
```

默认保存H.264/yuv420p视频，`progress.json`查看成功率，`episodes.jsonl`查看逐次结果。
不同工作站可分发不同任务或不同配方，资源路径由各自machine配置管理。
正式协议和分片规则见 [RMBench baseline](rmbench-baseline.md)，
当前单任务训练记录见 [Rearrange Blocks实验](experiments/2026-10-26-rmbench-rearrange-blocks-training.md)。

### 5090：一次复制全部任务，按需训练其中一部分

完整复制的是九任务的**原始训练包**（11.77 GiB、450条演示），不是把所有任务都加入一次训练。
目前本机仅 Rearrange Blocks 已转换成 LeRobot；其余八个任务的完整ZIP均已下载、校验。
先将全部原始包复制到5090，以后换任务无需重新联网下载。

**现有入口支持单任务、任意明确的任务组合以及全部九任务。**
但训练时 `--tasks` 必须与派生数据目录中的任务集合完全一致；当前不支持在已转换的
`lerobot-all` 中仅通过训练参数筛选子集，因为归一化统计也必须来自所选任务。
因此采用“完整原始资源 → 所选任务专用的派生目录 → 训练”的流程。

1. 在5090原来的仓库中切到团队已推送的分支并复用原 `.venv`，不用新建worktree。
先确认该目录没有正在运行的训练、没有未处理的源码修改；已存在本地分支时用
`git switch feat/rmbench-baseline`，不重复创建：

```bash
git fetch origin
git switch -c feat/rmbench-baseline --track origin/feat/rmbench-baseline
uv sync --frozen
```

2. 在5090上一次复制所有九任务的ZIP。`SOURCE_HOST` 替换成数据准备机器的SSH地址。
文件列表来自已提交的固定清单，只传训练ZIP，不传仿真资产或本机的smoke目录：

```bash
export RMBENCH_ARCHIVES="$HOME/datasets/rmbench/archives"
mkdir -p "$RMBENCH_ARCHIVES"
.venv/bin/python - <<'PYLIST'
import json
from pathlib import Path
m = json.loads(Path('configs/datasets/rmbench-resources.json').read_text())
Path('/tmp/rmbench-archive-files.txt').write_text(''.join(x['path'] + '\n' for x in m['files']))
PYLIST
rsync -a --info=progress2 --files-from=/tmp/rmbench-archive-files.txt \
  SOURCE_HOST:/home/hello/lizk/datasets/rmbench/raw/ \
  "$RMBENCH_ARCHIVES/"
```

随后执行本节第2步中的SHA256校验/解压代码，令 `RMBENCH_RAW="$HOME/datasets/rmbench/raw-full"`；
再按第3步准备固定版本RMBench源码及machine字段 `rmbench_source`、`rmbench_raw`、
`rmbench_dataset`、`vlm` 和 `runs_root`。源码仅提供转换codec，不需要安装仿真依赖。

3. 选择任务并转换。以下例子选择两任务；任务名称用空格分隔，不使用逗号：

```bash
TASKS=(rearrange_blocks swap_T)
DATASET_DIR="$HOME/datasets/rmbench/lerobot-rearrange-swap-t"
RUN_NAME=rmbench-rearrange-swap-t-seed1000

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
.venv/bin/python -m lerobot_policy_vla_ttt.baselines.rmbench_data \
  --machine configs/machine.local.toml --tasks "${TASKS[@]}" \
  --output "$DATASET_DIR"
```

转换读取完整原始数据中的这两个任务，输出独立的统计及 `preparation.json`。
如目录已存在且已完整转换，直接复用，不再执行转换。不要使用 `--max-episodes` 做正式数据。

4. 用相同任务列表预览训练配置：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_rmbench_multi.sh \
  --machine configs/machine.local.toml \
  --recipe configs/training/rmbench/rearrange-blocks-grouped.json \
  --tasks "${TASKS[@]}" --dataset-root "$DATASET_DIR" \
  --run-name "$RUN_NAME" --seed 1000 --dry-run
```

这里复用每5000步保存的数值recipe，文件名不限制任务。上述**两任务**实际为
Stage1 10000步、Stage2 160000步，batch8/4；这是按任务数扩大样本预算，不是5k/80k总预算。
确认预算后去掉 `--dry-run` 正式启动；先短测则替换为 `--smoke-steps 2` 并用不同run-name。
这只是操作示例，不表示已经决定启动两任务或九任务长训练。

| 训练范围 | 转换时的 `--tasks` | 训练入口 | 本recipe的Stage1 / Stage2步数 |
| --- | --- | --- | --- |
| 一个任务 | `rearrange_blocks` | `train_rmbench_single.sh` | 5000 / 80000 |
| 指定两个任务 | `rearrange_blocks swap_T` | `train_rmbench_multi.sh` | 10000 / 160000 |
| 指定三个任务 | `rearrange_blocks swap_T swap_blocks` | `train_rmbench_multi.sh` | 15000 / 240000 |
| 全部九任务 | `all` | `train_rmbench_multi.sh` | 45000 / 720000 |

换任务组合时同时换派生目录和run-name；训练传入与转换相同的 `--tasks`。
这些多任务入口训练一个联合模型。若希望每个任务独立一个模型，应逐任务转换并使用单任务入口，
可把独立任务分发到不同工作站。

### 5090：只复用已转换单任务数据的快捷方式

适用于已有同版本 InternVL3、本机只做训练的工作站；无需下载原始ZIP、克隆仿真源码或
安装评测环境。先检出团队提供的 RMBench commit，执行上面的 `uv sync --frozen`。

1. 从数据准备机器复制完整 `lerobot-rearrange-blocks/` 到目标机器，例如：

```bash
mkdir -p "$HOME/datasets/rmbench"
# SOURCE_HOST 换为数据准备机器的SSH地址；在5090工作站执行。
rsync -a --info=progress2   SOURCE_HOST:/home/hello/lizk/datasets/rmbench/lerobot-rearrange-blocks/   "$HOME/datasets/rmbench/lerobot-rearrange-blocks/"
```

2. 配置 `configs/machine.local.toml`，以下最小配置足够读取已转换数据进行训练；
替换为5090机器的实际绝对路径，已有配置保留其他字段：

```toml
[assets]
vlm = "/home/hello/models/InternVL3-1B-hf"
rmbench_dataset = "/home/hello/datasets/rmbench/lerobot-rearrange-blocks"
[storage]
runs_root = "/home/hello/experiments/rmbench"
```

3. 两阶段短测：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_rmbench_single.sh   --machine configs/machine.local.toml   --recipe configs/training/rmbench/rearrange-blocks-grouped.json   --tasks rearrange_blocks --run-name rearrange-5090-smoke --smoke-steps 2
```

短测将batch改为1，用于检查两阶段加载/训练/保存链路，不证明正式batch8/4一定适合目标GPU。
本机72GB显卡的显存结果不能代替5090验收；正式启动后检查显存、loss与吞吐，避免隐式改变配方。

4. 使用新名字正式训练：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_rmbench_single.sh   --machine configs/machine.local.toml   --recipe configs/training/rmbench/rearrange-blocks-grouped.json   --tasks rearrange_blocks --run-name rearrange-5090-grouped-seed1000 --seed 1000
```

两阶段分别batch8/4、5000/80000更新，每5000更新保存。跨机器比较使用同一recipe、
数据、VLM版本和seed；这里的新run-name是新的独立训练，不是恢复已有任务。
