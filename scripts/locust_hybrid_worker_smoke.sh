#!/usr/bin/env bash
#SBATCH --job-name=hybrid-workers-smoke
set -euo pipefail
umask 077

: "${AGENT_SUPPORT_CONFIG:?Set the hybrid smoke config path}"
: "${AGENT_SUPPORT_OUTPUT_ROOT:?Set a fresh durable output root}"
: "${AGENT_SUPPORT_QWEN_PYTHON:?Set the existing Qwen interpreter}"
: "${AGENT_SUPPORT_GPTOSS_PYTHON:?Set the existing GPT-OSS interpreter}"
: "${SLURM_JOB_ID:?Submit this script with sbatch}"

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PYTHONHASHSEED=0
exec "$AGENT_SUPPORT_QWEN_PYTHON" -m agent_support.hybrid_worker_smoke \
  --config "$AGENT_SUPPORT_CONFIG" \
  --output "$AGENT_SUPPORT_OUTPUT_ROOT/worker_smoke_${SLURM_JOB_ID}.json"
