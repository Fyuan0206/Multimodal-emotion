#!/usr/bin/env bash
set -u

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_id="${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S)"
log_file="$project_root/Q2/train_${run_id}.log"
status_file="$project_root/Q2/train_${run_id}.status"

printf 'RUNNING\n' > "$status_file"
bash "$project_root/Q2/run_server.sh" > "$log_file" 2>&1
status=$?
printf '%s\n' "$status" > "$status_file"
exit "$status"
