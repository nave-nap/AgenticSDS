#!/usr/bin/env bash
#SBATCH --job-name=recovery-v4-smoke
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=02:30:00
set -euo pipefail
umask 077
repo_root="${SLURM_SUBMIT_DIR:?Submit from the v4 worktree}"
: "${RECOVERY_ROOT:?Set isolated v4 output root}"
: "${RECOVERY_ARM:?Set expanded_v3_v1 or autonomous_recovery_v4}"
: "${AGENT_SUPPORT_GPTOSS_PYTHON:?Set pinned GPT-OSS Python}"
: "${AGENT_SUPPORT_GPTOSS_HF_HOME:?Set pinned GPT-OSS cache}"
: "${AGENT_SUPPORT_M2_200_ARTIFACT_DIR:?Set read-only M2-200 artifact directory}"
venv_path="${VENV_PATH:-$repo_root/.venv}"
test -x "$venv_path/bin/python"
test -x "$AGENT_SUPPORT_GPTOSS_PYTHON"
test -d "$AGENT_SUPPORT_GPTOSS_HF_HOME"
test -r "$AGENT_SUPPORT_M2_200_ARTIFACT_DIR/corpus_200.json"
test -r "$AGENT_SUPPORT_M2_200_ARTIFACT_DIR/index_200.json"
export PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv
cd "$repo_root"
"$venv_path/bin/python" -m agent_support.recovery_v4_eval smoke \
  --root "$RECOVERY_ROOT" --arm "$RECOVERY_ARM" --count "${RECOVERY_SMOKE_COUNT:-2}"
