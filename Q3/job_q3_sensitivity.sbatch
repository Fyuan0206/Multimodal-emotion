#!/usr/bin/env bash
#SBATCH --job-name=q3-sensitivity
#SBATCH --partition=scx6706_test
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --output=Q3/logs/slurm-sensitivity-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
if ! type module >/dev/null 2>&1; then
    source /etc/profile
fi
module load compilers/cuda/12.1 compilers/gcc/11.3.0 cudnn/8.8.1.3_cuda12.x
libgomp=("$PWD"/Q1/.venv/lib/python*/site-packages/scikit_learn.libs/libgomp-*.so.*)
test -f "${libgomp[0]}"
export LD_PRELOAD="${libgomp[0]}${LD_PRELOAD:+:$LD_PRELOAD}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
"$PWD/Q1/.venv/bin/python" Q3/stress_explanation_q3.py \
  --attachment2 "$PWD/data/附件2-数据集特征文件/aligned_50.pkl" \
  --run "$PWD/Q3/outputs/run_1530329" --device cuda
