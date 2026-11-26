#!/usr/bin/env bash
#SBATCH --job-name=three-gpu-cache-preflight
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --no-requeue
#SBATCH --output=/dev/null
set -euo pipefail
umask 077

: "${THREE_GPU_ROOT:?Set fresh experiment root}"
: "${VENV_PATH:?Set pinned Qwen environment}"
: "${HF_HOME:?Set pinned checkpoint cache}"
: "${SLURM_JOB_ID:?Run under Slurm}"
attempt="$THREE_GPU_ROOT/cpu/job_${SLURM_JOB_ID}_restart_${SLURM_RESTART_COUNT:-0}"
mkdir -p "$THREE_GPU_ROOT/cpu"
mkdir "$attempt"
exec >"$attempt/slurm.out" 2>&1
cd "${SLURM_SUBMIT_DIR:?Submit from the committed checkout}"
"$VENV_PATH/bin/python" -m agent_support.three_gpu_overnight cache \
  --output-dir "$attempt/cache" --hf-home "$HF_HOME"
