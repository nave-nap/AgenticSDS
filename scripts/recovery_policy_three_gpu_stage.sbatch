#!/usr/bin/env bash
#SBATCH --job-name=recovery-policy-three-gpu
#SBATCH --gpus=3
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --no-requeue
set -euo pipefail
umask 077
: "${POLICY_ROOT:?Set a fresh four-policy root}"
: "${POLICY_STAGE:?Set smoke, pilot, or batch}"
: "${POLICY_FEASIBILITY_RESULT:?Set the valid full feasibility result}"
: "${POLICY_LIST:?Set the exact space-separated policy IDs}"
: "${VENV_PATH:?Set pinned Qwen environment}"
: "${AGENT_SUPPORT_GPTOSS_PYTHON:?Set pinned GPT-OSS environment}"
: "${AGENT_SUPPORT_GPTOSS_HF_HOME:?Set pinned GPT-OSS cache}"
: "${AGENT_SUPPORT_M2_200_ARTIFACT_DIR:?Set pinned memory artifacts}"
: "${CUDA_VISIBLE_DEVICES:?Slurm must expose three allocated GPUs}"
repo_root="${SLURM_SUBMIT_DIR:?Submit from a clean committed checkout}"
cd "$repo_root"
test -z "$(git status --porcelain)"
test -x "$VENV_PATH/bin/python"
test -x "$AGENT_SUPPORT_GPTOSS_PYTHON"
IFS=',' read -r -a allocated <<< "$CUDA_VISIBLE_DEVICES"
test "${#allocated[@]}" -eq 3
test "${allocated[0]}" != "${allocated[1]}"
test "${allocated[0]}" != "${allocated[2]}"
test "${allocated[1]}" != "${allocated[2]}"
export AGENT_SUPPORT_GPTOSS_CUDA_VISIBLE_DEVICES="${allocated[1]},${allocated[2]}"
export CUDA_VISIBLE_DEVICES="${allocated[0]}"
export AGENT_SUPPORT_GPTOSS_TWO_GPU=1
export AGENT_SUPPORT_GPTOSS_CHUNKED_EAGER=1
export AGENT_SUPPORT_GPU_MEMORY_DIAGNOSTICS=1
export AGENT_SUPPORT_GPTOSS_WORKER_TRACEBACKS=1
export AGENT_SUPPORT_GPTOSS_STARTUP_TIMEOUT_SECONDS=5400
export AGENT_SUPPORT_GPTOSS_CUDA_CACHE_RELEASE=0
export PYTHONHASHSEED=0
export HF_HOME="$AGENT_SUPPORT_GPTOSS_HF_HOME"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.free --format=csv
"$VENV_PATH/bin/python" -c 'import torch; assert torch.cuda.device_count() == 1 and "L40S" in torch.cuda.get_device_name(0)'
"$VENV_PATH/bin/python" -m agent_support.recovery_policy_eval feasibility-gate \
  --result "$POLICY_FEASIBILITY_RESULT"
read -r -a policies <<< "$POLICY_LIST"
for policy in "${policies[@]}"; do
  "$VENV_PATH/bin/python" -m agent_support.recovery_policy_eval preflight \
    --policy "$policy" --strict-runtime
done
if [[ "$POLICY_STAGE" == batch ]]; then
  : "${POLICY_BATCH:?Set batch 0 through 9}"
  "$VENV_PATH/bin/python" -m agent_support.recovery_policy_eval matched-batch \
    --root "$POLICY_ROOT" --batch "$POLICY_BATCH" --policies "${policies[@]}"
else
  "$VENV_PATH/bin/python" -m agent_support.recovery_policy_eval matched-stage \
    --root "$POLICY_ROOT" --stage "$POLICY_STAGE" --policies "${policies[@]}"
fi
