#!/usr/bin/env bash
# User-local Linux x86_64 / Blackwell environment. Does not alter drivers or system CUDA.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
RMBENCH_ENV_ROOT="$PWD"
export UV_PROJECT_ENVIRONMENT="$RMBENCH_ENV_ROOT/.venv"
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_INDEX_URL="$UV_INDEX_URL"
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  uv sync --frozen --inexact --python 3.11
.venv/bin/python install_cuda.py
.venv/bin/python patch_dependencies.py
.venv/bin/python install_oidn.py
export CUDA_HOME="$RMBENCH_ENV_ROOT/toolkit/cuda-12.8"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$(.venv/bin/python -c 'import torch; a,b = torch.cuda.get_device_capability(0); print(f"{a}.{b}")')}"
export MAX_JOBS="${MAX_JOBS:-2}"
export SETUPTOOLS_SCM_PRETEND_VERSION=0.7.8
mkdir -p vendor
if [[ ! -d vendor/curobo-code/.git ]]; then
  git clone --filter=blob:none --no-checkout --depth 1 --branch v0.7.8 \
    https://github.com/NVlabs/curobo.git vendor/curobo-code
fi
[[ "$(git -C vendor/curobo-code rev-parse HEAD)" == d64c4b005459db10c5dd867d8b30a87d5bda9bdb ]]
git -C vendor/curobo-code config core.sparseCheckout true
git -C vendor/curobo-code config core.sparseCheckoutCone false
cat > vendor/curobo-code/.git/info/sparse-checkout <<'EOF'
/setup.py
/setup.cfg
/pyproject.toml
/MANIFEST.in
/LICENSE
/README.md
/src/curobo/**
!/src/curobo/content/assets/**
EOF
git -C vendor/curobo-code read-tree -mu HEAD
# Building inside the sparse repo makes setuptools_scm archive all upstream assets.
RMBENCH_BUILD_DIR=$(mktemp -d /tmp/vla-ttt-curobo-build.XXXXXX)
trap 'rm -rf -- "$RMBENCH_BUILD_DIR"' EXIT
.venv/bin/python - "$RMBENCH_BUILD_DIR" <<'PY'
import shutil, sys
shutil.copytree('vendor/curobo-code', sys.argv[1], dirs_exist_ok=True,
                ignore=shutil.ignore_patterns('.git', 'build', '*.egg-info', '__pycache__'))
PY
uv pip install --python .venv/bin/python --no-build-isolation --no-deps "$RMBENCH_BUILD_DIR"
.venv/bin/python -c 'from curobo.wrap.reacher.motion_gen import MotionGen; print("CuRobo import OK")'
uv pip freeze --python .venv/bin/python > installed-freeze.txt
