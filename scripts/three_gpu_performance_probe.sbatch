#!/usr/bin/env bash
#SBATCH --job-name=three-gpu-performance-probe
#SBATCH --gpus=3
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:30:00
set -euo pipefail

export THREE_GPU_PHASE=performance
source "${SLURM_SUBMIT_DIR:?Submit from the committed checkout}/scripts/three_gpu_chunked_eager_load_diagnostic.sbatch"
