#!/usr/bin/env bash
set -euo pipefail
cd /home/bingxing2/home/scx6706/xbmu-CCQ/Q1
TASK=视觉增强专项
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
Q1_GOMP=$(.venv/bin/python -c 'import glob; print(glob.glob(".venv/lib/python*/site-packages/scikit_learn.libs/libgomp*.so*")[0])')
export LD_PRELOAD="$(pwd)/$Q1_GOMP${LD_PRELOAD:+:$LD_PRELOAD}"
mkdir -p "$TASK/results"
trap 'printf "failed exit=%s time=%s\n" "$?" "$(date -Iseconds)" > "$TASK/results/job_failed.txt"' ERR
.venv/bin/python "$TASK/extract_expression.py" --data-root '../data/附件1-数据集原始多模态样本/MOSEI数据集部分原始视频-100条' --q1 outputs/server_a100 --out "$TASK/results"
.venv/bin/python "$TASK/evaluate_expression.py" --expression "$TASK/results/features" --features outputs/server_a100/features --labels '../data/附件1-数据集原始多模态样本/MOSEI数据集部分原始视频-100条/label-100.xlsx' --out "$TASK/results/evaluation"
.venv/bin/python "$TASK/verify_results.py"
printf 'completed %s\n' "$(date -Iseconds)" > "$TASK/results/SUCCESS.txt"
