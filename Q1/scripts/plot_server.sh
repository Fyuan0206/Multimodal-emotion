#!/usr/bin/env bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
test -f src/pipeline.py
export MPLBACKEND=Agg
module load compilers/cuda/12.1 compilers/gcc/11.3.0 cudnn/8.8.1.3_cuda12.x
Q1_GOMP=$(.venv/bin/python -c 'import glob; print(glob.glob(".venv/lib/python*/site-packages/scikit_learn.libs/libgomp*.so*")[0])')
export LD_PRELOAD="$(pwd)/$Q1_GOMP${LD_PRELOAD:+:$LD_PRELOAD}"
.venv/bin/python scripts/plot_paper.py --data-root '../data/附件1-数据集原始多模态样本/MOSEI数据集部分原始视频-100条' --out outputs/server_a100 --figures figures/server_a100
