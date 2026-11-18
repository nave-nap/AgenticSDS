#!/usr/bin/env bash
#SBATCH --job-name=heldout50
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=02:00:00
set -euo pipefail
umask 077

repo_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
: "${AGENT_SUPPORT_DATA:?Set AGENT_SUPPORT_DATA to the external evaluation data directory}"
: "${EVAL_ARM:?Set EVAL_ARM to autonomous_v3 or qwen_standalone}"
: "${EVAL_BATCH:?Set EVAL_BATCH to a nonnegative batch number}"
venv_path="${VENV_PATH:-$repo_root/.venv}"
test -x "$venv_path/bin/python" || { echo "Missing Qwen Python: $venv_path/bin/python" >&2; exit 1; }
export PYTHONHASHSEED=0
export HF_HOME="${HF_HOME:-$AGENT_SUPPORT_DATA/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export AGENT_SUPPORT_M2_200_ARTIFACT_DIR="${AGENT_SUPPORT_M2_200_ARTIFACT_DIR:-/shared_data0/evanpan/agent_support/gptoss_memory_v1/memory_scaling_v1/artifacts}"
if [[ "$EVAL_ARM" == "autonomous_v3" ]]; then
  : "${AGENT_SUPPORT_GPTOSS_PYTHON:?Set AGENT_SUPPORT_GPTOSS_PYTHON to the pinned GPT-OSS Python executable}"
  test -x "$AGENT_SUPPORT_GPTOSS_PYTHON"
  export AGENT_SUPPORT_GPTOSS_HF_HOME="${AGENT_SUPPORT_GPTOSS_HF_HOME:-$HF_HOME}"
elif [[ "$EVAL_ARM" != "qwen_standalone" ]]; then
  echo "Unsupported held-out50 arm: $EVAL_ARM" >&2
  exit 1
fi

evaluation_root="$AGENT_SUPPORT_DATA/autonomous_support_evaluation/heldout50_v1"
cd "$repo_root"
arguments=(run-batch --root "$evaluation_root" --arm "$EVAL_ARM" --batch "$EVAL_BATCH")
if [[ -n "${EVAL_MAX_TASKS_PER_ATTEMPT:-}" ]]; then
  arguments+=(--max-tasks-per-attempt "$EVAL_MAX_TASKS_PER_ATTEMPT")
fi
"$venv_path/bin/python" -m agent_support.heldout50_evaluation \
  "${arguments[@]}"
