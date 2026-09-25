#!/usr/bin/env bash
set -euo pipefail
cd /home/bingxing2/home/scx6706/xbmu-CCQ/Q1
Q1_GOMP=$(.venv/bin/python -c 'import glob; print(glob.glob(".venv/lib/python*/site-packages/scikit_learn.libs/libgomp*.so*")[0])')
export LD_PRELOAD="$(pwd)/$Q1_GOMP${LD_PRELOAD:+:$LD_PRELOAD}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
.venv/bin/python 独立验证专项/evaluate_visual.py --features outputs/server_a100/features --labels '../data/附件1-数据集原始多模态样本/MOSEI数据集部分原始视频-100条/label-100.xlsx' --out 独立验证专项/visual_results
