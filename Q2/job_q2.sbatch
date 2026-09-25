#!/usr/bin/env bash
#SBATCH --job-name=q2-robust
#SBATCH --partition=scx6706_test
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --output=Q2/logs/slurm-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
bash Q2/launch_server.sh
