#!/usr/bin/env bash
#SBATCH --job-name=textcraft-control-smoke
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=02:00:00
set -euo pipefail
umask 077
repo_root="${SLURM_SUBMIT_DIR:?Submit from the control checkout}"
: "${CONTROL_ROOT:?Set a new control output root}"
: "${CONTROL_ARM:?Set gptoss_standalone_v1 or gptoss_review_only_v1}"
: "${AGENT_SUPPORT_GPTOSS_PYTHON:?Set the pinned GPT-OSS Python executable}"
: "${AGENT_SUPPORT_GPTOSS_HF_HOME:?Set the pinned GPT-OSS cache}"
venv_path="${VENV_PATH:-$repo_root/.venv}"
test -x "$venv_path/bin/python"
test -x "$AGENT_SUPPORT_GPTOSS_PYTHON"
export PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd "$repo_root"
"$venv_path/bin/python" -m agent_support.control_experiments smoke \
  --root "$CONTROL_ROOT" --arm "$CONTROL_ARM" --task-index "${CONTROL_DEV_INDEX:-0}"
