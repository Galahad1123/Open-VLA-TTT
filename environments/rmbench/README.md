# RMBench 仿真环境

policy 使用项目 `.venv`；仿真必须独立，不能在 policy 环境执行官方安装脚本。
固定上游 `7d1cdfc7661c273a7a5c5f6126e2d1a7f0a3f4b7`，CuRobo `v0.7.8`。

官方安装说明及脚本位于上游 `INSTALLATION.md`、`scripts/_install.sh`。
它需要 Python 3.10、SAPIEN 3.0.0b1、mplib 0.2.1、Open3D 0.18.0、
PyTorch3D、CuRobo，以及 CUDA 编译器。GPU 驱动不包含 `nvcc`。
官方 PyTorch 2.4.1 配方不能直接视为已验证支持 Blackwell；在该显卡上应先验证
兼容的 PyTorch/CUDA/CuRobo 组合，再锁定环境。当前没有经过真实验证的替代锁文件。

准备独立 checkout 和 Python 环境（路径可自行选择）：

```bash
git clone https://github.com/RoboTwin-Platform/RMBench "$RMBENCH_RUNTIME"
git -C "$RMBENCH_RUNTIME" checkout 7d1cdfc7661c273a7a5c5f6126e2d1a7f0a3f4b7
uv venv --python 3.10 "$RMBENCH_VENV"
source "$RMBENCH_VENV/bin/activate"
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
command -v nvcc
```

先确保 CUDA 工具链和显卡架构相容，再在该环境按官方脚本安装。
大体积 wheel 下载请清除 HTTP_PROXY/HTTPS_PROXY/ALL_PROXY 及小写变量，
避免终端代理传输大文件。不要把缺少 CuRobo 时跳过专家筛选作为正式协议。
将下载资源的 `embodiments`、`objects` 目录链接到 runtime 的 `assets/` 下；
不要修改共享的上游源码。每个运行进程使用专属 runtime，避免规划器的临时文件冲突。

最后在仿真环境安装本仓库的轻量客户端（不依赖 LeRobot）：

```bash
uv pip install --python "$RMBENCH_VENV/bin/python" -e "$VLA_TTT_ROOT/packages/eval-client"
"$RMBENCH_VENV/bin/python" -m vla_ttt_eval.rmbench_runner --help
```

实际安装完成后保存 `pip freeze`、CUDA 版本及上游 git/submodule commit。
当前实现交付不包含已通过真实 rollout 的仿真环境；详见实验记录。
