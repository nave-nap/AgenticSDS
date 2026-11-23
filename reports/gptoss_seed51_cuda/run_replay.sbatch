#!/usr/bin/env bash
#SBATCH --job-name=gptoss-seed51-replay
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=01:00:00
set -euo pipefail
umask 077

: "${DEPLOY_REPO:?Set the isolated diagnostic worktree}"
: "${DIAG_DIR:?Set the separate diagnostic directory}"
: "${DIAG_OUTPUT_DIR:?Set one new output directory}"
: "${REPLAY_BACKEND:?Set flex_attention or default}"
: "${VENV_PATH:?Set the pinned evaluator Python environment}"
: "${AGENT_SUPPORT_GPTOSS_PYTHON:?Set the pinned GPT-OSS Python}"
: "${AGENT_SUPPORT_GPTOSS_HF_HOME:?Set the pinned GPT-OSS cache}"
test ! -e "$DIAG_OUTPUT_DIR"
test -x "$VENV_PATH/bin/python"
test -x "$AGENT_SUPPORT_GPTOSS_PYTHON"
test -d "$AGENT_SUPPORT_GPTOSS_HF_HOME"
case "$REPLAY_BACKEND" in flex_attention|default) ;; *) exit 2 ;; esac

export PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_LAUNCH_BLOCKING=1
export AGENT_SUPPORT_GPTOSS_WORKER_TRACEBACKS=1
export AGENT_SUPPORT_GPU_MEMORY_DIAGNOSTICS=1
export AGENT_SUPPORT_GPTOSS_CUDA_CACHE_RELEASE=0
export PYTHONPATH="$DEPLOY_REPO${PYTHONPATH:+:$PYTHONPATH}"
cd "$DEPLOY_REPO"
"$VENV_PATH/bin/python" -m reports.gptoss_seed51_cuda.replay \
  --plan "$DIAG_DIR/source_audit/replay_plan.json" \
  --backend "$REPLAY_BACKEND" --output-dir "$DIAG_OUTPUT_DIR"
