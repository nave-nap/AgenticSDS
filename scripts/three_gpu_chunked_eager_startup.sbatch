#!/usr/bin/env bash
#SBATCH --job-name=three-gpu-chunked-startup
#SBATCH --gpus=3
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:30:00
set -euo pipefail

# Allow a slow shared-cache model load without extending the 1,800-second
# generation RPC deadline frozen in the evaluation configuration.
export AGENT_SUPPORT_GPTOSS_STARTUP_TIMEOUT_SECONDS=5400
source "${SLURM_SUBMIT_DIR:?Submit from the committed checkout}/scripts/three_gpu_chunked_eager.sbatch"
