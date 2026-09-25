#!/usr/bin/env bash
#SBATCH --job-name=q2-v2-paired
#SBATCH --partition=scx6706_test
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --output=Q2/slurm-v2-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
if ! type module >/dev/null 2>&1; then
    source /etc/profile
fi
module load compilers/cuda/12.1 compilers/gcc/11.3.0 cudnn/8.8.1.3_cuda12.x
libgomp=("$PWD"/Q1/.venv/lib/python*/site-packages/scikit_learn.libs/libgomp-*.so.*)
test -f "${libgomp[0]}"
export LD_PRELOAD="${libgomp[0]}${LD_PRELOAD:+:$LD_PRELOAD}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
python="$PWD/Q1/.venv/bin/python"
output="$PWD/Q2/outputs/v2_${SLURM_JOB_ID}"
mkdir -p "$output"
trap 'rc=$?; echo "$rc" > "$output/exit.status"' EXIT
"$python" -c 'import importlib.metadata as m; print("\n".join(sorted(str(d.metadata["Name"])+"=="+d.version for d in m.distributions())))' > "$output/environment_packages.txt"
"$python" -m unittest Q2/test_q2_v2.py
"$python" Q2/run_q2_v2.py --data-root "$PWD/data" --output "$output/smoke" --device cuda --smoke
"$python" Q2/run_q2_v2.py --data-root "$PWD/data" --output "$output" --device cuda
if [[ -f Q2/report_q2_v2.py ]]; then
    "$python" Q2/report_q2_v2.py --output "$output"
fi
