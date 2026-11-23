#!/usr/bin/env bash
#SBATCH --job-name=gptoss-bounded-replay
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=02:00:00
set -euo pipefail
umask 077

: "${DEPLOY_REPO:?Set the isolated repair worktree}"
: "${DIAG_DIR:?Set the new diagnostic directory containing replay.py and the plan}"
: "${TOKEN_MANIFEST:?Set the verified pinned-tokenizer manifest}"
: "${DIAG_OUTPUT_DIR:?Set a unique output directory}"
: "${REPLAY_BACKEND:?Set eager or flex_attention}"
: "${VENV_PATH:?Set the pinned evaluator venv}"
: "${AGENT_SUPPORT_GPTOSS_PYTHON:?Set the pinned GPT-OSS worker Python}"
: "${AGENT_SUPPORT_GPTOSS_HF_HOME:?Set the pinned GPT-OSS model cache}"
: "${HF_HOME:?Set the pinned Qwen model cache}"
test ! -e "$DIAG_OUTPUT_DIR"
test -x "$VENV_PATH/bin/python"
test -x "$AGENT_SUPPORT_GPTOSS_PYTHON"
case "$REPLAY_BACKEND" in eager|flex_attention) ;; *) exit 2 ;; esac

export PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_LAUNCH_BLOCKING=1
export AGENT_SUPPORT_GPTOSS_WORKER_TRACEBACKS=1
export AGENT_SUPPORT_GPTOSS_PHASE_TRACE=1
export AGENT_SUPPORT_GPU_MEMORY_DIAGNOSTICS=1
export AGENT_SUPPORT_GPTOSS_CUDA_CACHE_RELEASE=0
export PYTHONPATH="$DEPLOY_REPO${PYTHONPATH:+:$PYTHONPATH}"
cd "$DEPLOY_REPO"
"$VENV_PATH/bin/python" "$DIAG_DIR/replay.py" \
  --plan "$DIAG_DIR/archived_replay_plan.json" \
  --token-manifest "$TOKEN_MANIFEST" \
  --backend "$REPLAY_BACKEND" --mode "${REPLAY_MODE:-sequence}" \
  --output-dir "$DIAG_OUTPUT_DIR"
