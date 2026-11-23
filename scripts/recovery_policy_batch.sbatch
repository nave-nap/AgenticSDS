#!/usr/bin/env bash
#SBATCH --job-name=recovery-policy-batch
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=02:30:00
set -euo pipefail
umask 077
repo_root="${SLURM_SUBMIT_DIR:?Submit from the policy evaluation worktree}"
: "${RECOVERY_ROOT:?Set initialized policy output root}"
: "${RECOVERY_POLICY:?Set original_v4, state_aware_v1, accuracy_first_v1, or accuracy_first_terminal_v1}"
: "${RECOVERY_BATCH:?Set batch 0 through 9}"
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
export AGENT_SUPPORT_GPTOSS_FLEX_ATTENTION=0
export AGENT_SUPPORT_GPTOSS_WORKER_TRACEBACKS=1
export AGENT_SUPPORT_GPU_MEMORY_DIAGNOSTICS=1
export AGENT_SUPPORT_GPTOSS_CUDA_CACHE_RELEASE=0
export CUDA_LAUNCH_BLOCKING=1
cd "$repo_root"
"$venv_path/bin/python" -m agent_support.recovery_policy_eval run-batch \
  --root "$RECOVERY_ROOT" --policy "$RECOVERY_POLICY" --batch "$RECOVERY_BATCH" \
  --max-tasks-per-attempt "${RECOVERY_MAX_TASKS_PER_ATTEMPT:-3}"
