#!/usr/bin/env bash
#SBATCH --job-name=three-gpu-overnight
#SBATCH --gpus=3
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --output=/dev/null
set -euo pipefail
umask 077
: "${THREE_GPU_ROOT:?Set fresh experiment root}"
: "${SLURM_JOB_ID:?Run under Slurm}"
mkdir -p "$THREE_GPU_ROOT/gpu"
export THREE_GPU_ATTEMPT="$THREE_GPU_ROOT/gpu/job_${SLURM_JOB_ID}_restart_${SLURM_RESTART_COUNT:-0}"
mkdir "$THREE_GPU_ATTEMPT"
exec >"$THREE_GPU_ATTEMPT/slurm.out" 2>&1

export THREE_GPU_PHASE=overnight
export AGENT_SUPPORT_GPTOSS_CHUNKED_EAGER=1
export AGENT_SUPPORT_GPTOSS_MODEL_LOAD_DIAGNOSTICS=1
export AGENT_SUPPORT_GPTOSS_INFERENCE_SAMPLING=1
export AGENT_SUPPORT_GPTOSS_ALLOW_PROFILER=1
export AGENT_SUPPORT_GPTOSS_STARTUP_TIMEOUT_SECONDS=5400
source "${SLURM_SUBMIT_DIR:?Submit from the committed checkout}/scripts/three_gpu_mxfp4.sbatch"
