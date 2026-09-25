#!/usr/bin/env bash
# Run on the allocated GPU compute node, from any working directory.
set -euo pipefail
cd "$(dirname "$0")/.."
export TMPDIR="$(cd .. && pwd)/tmp"
export MPLBACKEND=Agg
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p "$TMPDIR" logs
module load compilers/cuda/12.1 compilers/gcc/11.3.0 cudnn/8.8.1.3_cuda12.x
PYTHON=.venv/bin/python
# ARM/glibc: reserve TLS for the scikit-learn OpenMP runtime before torch imports.
Q1_GOMP=$("$PYTHON" -c 'import glob; p=glob.glob(".venv/lib/python*/site-packages/scikit_learn.libs/libgomp*.so*"); assert len(p)==1,p; print(p[0])')
export LD_PRELOAD="$(pwd)/$Q1_GOMP${LD_PRELOAD:+:$LD_PRELOAD}"

DATA_ROOT="${1:-../data/附件1-数据集原始多模态样本/MOSEI数据集部分原始视频-100条}"
OUT="${2:-outputs/server_a100}"
mkdir -p "$OUT"
nvidia-smi > "$OUT/nvidia-smi.txt"
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0)); x=torch.ones((64,64),device="cuda"); print((x@x).sum().item()); torch.cuda.synchronize()'
"$PYTHON" -m pytest -q tests | tee logs/server_tests.log
"$PYTHON" scripts/audit_independent.py --data-root "$DATA_ROOT" --out "$OUT"
"$PYTHON" src/pipeline.py --data-root "$DATA_ROOT" --device cuda --out "$OUT" --audit-only
"$PYTHON" src/pipeline.py --data-root "$DATA_ROOT" --device cuda --out "$OUT" --pilot
"$PYTHON" src/pipeline.py --data-root "$DATA_ROOT" --device cuda --out "$OUT" --resume
"$PYTHON" scripts/validate_outputs.py --out "$OUT" | tee logs/server_validation.log
"$PYTHON" src/pipeline.py --data-root "$DATA_ROOT" --device cuda --limit 5 --out "$OUT/repeat5"
"$PYTHON" scripts/check_reproducibility.py --out "$OUT" | tee logs/server_reproducibility.log
"$PYTHON" scripts/evaluate_human_review.py --out "$OUT"
"$PYTHON" scripts/plot_paper.py --data-root "$DATA_ROOT" --out "$OUT" --figures figures/server_a100
"${UV_BIN:-$HOME/.local/bin/uv}" pip freeze --python "$PYTHON" > "$OUT/requirements-freeze.txt"
