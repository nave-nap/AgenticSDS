#!/usr/bin/env bash
#SBATCH --job-name=three-gpu-chunked-eager
#SBATCH --gpus=3
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:30:00
set -euo pipefail

export AGENT_SUPPORT_GPTOSS_CHUNKED_EAGER=1
export AGENT_SUPPORT_GPTOSS_WORKER_TRACEBACKS=1
source "${SLURM_SUBMIT_DIR:?Submit from the committed checkout}/scripts/three_gpu_mxfp4.sbatch"
