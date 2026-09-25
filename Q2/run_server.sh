#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

if ! type module >/dev/null 2>&1; then
    source /etc/profile
fi
module load compilers/cuda/12.1 compilers/gcc/11.3.0 cudnn/8.8.1.3_cuda12.x

# On this ARM server, scikit-learn's OpenMP library must load before PyTorch.
libgomp=("$project_root"/Q1/.venv/lib/python*/site-packages/scikit_learn.libs/libgomp-*.so.*)
if [[ ! -f "${libgomp[0]}" ]]; then
    echo "Q1 environment scikit-learn libgomp was not found" >&2
    exit 1
fi
export LD_PRELOAD="${libgomp[0]}${LD_PRELOAD:+:$LD_PRELOAD}"

"$project_root/Q1/.venv/bin/python" -m unittest Q2/test_q2.py

exec "$project_root/Q1/.venv/bin/python" "$project_root/Q2/run_q2.py" \
    --data-root "$project_root/data" --output "$project_root/Q2" \
    --epochs 30 --device cuda
