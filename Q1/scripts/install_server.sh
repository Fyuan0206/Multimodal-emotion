#!/usr/bin/env bash
# Platform-specific Linux aarch64 / Python 3.10 / CUDA 12.1 environment.
set -euo pipefail
cd "$(dirname "$0")/.."
export TMPDIR="$(cd .. && pwd)/tmp"
export UV_HTTP_TIMEOUT=60
export CMAKE_BUILD_PARALLEL_LEVEL=4
mkdir -p "$TMPDIR" logs
UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"
BASE_PYTHON="${BASE_PYTHON:-/home/bingxing2/home/scx6706/miniconda3/envs/py310/bin/python}"
TORCH_WHEEL="${TORCH_WHEEL:-/home/bingxing2/apps/package/pytorch/2.5.1-cu121_cp310/torch-2.5.1+cu121-cp310-cp310-linux_aarch64.whl}"
if [[ "$(uname -m)" != aarch64 ]]; then
    echo 'This installer targets the confirmed ARM server only.' >&2
    exit 1
fi
if [[ ! -x .venv/bin/python ]]; then
    "$UV_BIN" venv --python "$BASE_PYTHON" .venv
fi
"$UV_BIN" pip install --python .venv/bin/python --no-deps "$TORCH_WHEEL"
"$UV_BIN" pip install --python .venv/bin/python --index-url https://pypi.tuna.tsinghua.edu.cn/simple -r requirements-server.txt
"$UV_BIN" pip check --python .venv/bin/python
"$UV_BIN" pip freeze --python .venv/bin/python > requirements-server-lock.txt
