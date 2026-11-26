#!/usr/bin/env bash
#SBATCH --job-name=three-gpu-load-diagnostic
#SBATCH --gpus=3
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:30:00
set -euo pipefail

export AGENT_SUPPORT_GPTOSS_MODEL_LOAD_DIAGNOSTICS=1
source "${SLURM_SUBMIT_DIR:?Submit from the committed checkout}/scripts/three_gpu_chunked_eager_startup.sbatch"
