#!/usr/bin/env bash
set -euo pipefail
cd /home/bingxing2/home/scx6706/xbmu-CCQ/Q1
module load compilers/cuda/12.1 compilers/gcc/11.3.0 cudnn/8.8.1.3_cuda12.x
Q1_GOMP=$(.venv/bin/python -c 'import glob; print(glob.glob(".venv/lib/python*/site-packages/scikit_learn.libs/libgomp*.so*")[0])')
export LD_PRELOAD="$(pwd)/$Q1_GOMP${LD_PRELOAD:+:$LD_PRELOAD}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
DATA='../data/附件1-数据集原始多模态样本/MOSEI数据集部分原始视频-100条'
.venv/bin/python scripts/verify_silence_release.py --data-root "$DATA" --out outputs/server_a100 --previous archive/pre_silence_fix/outputs_server_a100
.venv/bin/python scripts/build_audible_review.py --data-root "$DATA" --out outputs/server_a100 --destination 有声回听专项
cp outputs/server_a100/04_token_feature_map.xlsx figures/server_a100/04_token_feature_map.xlsx
