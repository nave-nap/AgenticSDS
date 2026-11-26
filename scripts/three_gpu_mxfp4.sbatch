#!/usr/bin/env bash
#SBATCH --job-name=three-gpu-mxfp4
#SBATCH --gpus=3
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:30:00
set -euo pipefail
umask 077
launch_stage=validate_environment
trap 'status=$?; echo "Three-GPU launcher failed during ${launch_stage} at line ${LINENO} (exit ${status})" >&2' ERR

: "${THREE_GPU_PHASE:?Set feasibility, smoke, or batch}"
: "${THREE_GPU_ROOT:?Set a fresh three-GPU experiment root}"
: "${VENV_PATH:?Set the pinned Qwen environment}"
: "${AGENT_SUPPORT_GPTOSS_PYTHON:?Set the pinned GPT-OSS Python}"
: "${AGENT_SUPPORT_GPTOSS_HF_HOME:?Set the pinned GPT-OSS cache}"
: "${HF_HOME:?Set the pinned Qwen cache}"
: "${AGENT_SUPPORT_M2_200_ARTIFACT_DIR:?Set the pinned memory artifacts}"
: "${CUDA_VISIBLE_DEVICES:?Slurm must expose three allocated devices}"

IFS=',' read -r -a allocated <<< "$CUDA_VISIBLE_DEVICES"
test "${#allocated[@]}" -eq 3
test "${allocated[0]}" != "${allocated[1]}"
test "${allocated[0]}" != "${allocated[2]}"
test "${allocated[1]}" != "${allocated[2]}"
test -x "$VENV_PATH/bin/python"
test -x "$AGENT_SUPPORT_GPTOSS_PYTHON"
test -d "$AGENT_SUPPORT_GPTOSS_HF_HOME"
test -d "$HF_HOME"
test -r "$AGENT_SUPPORT_M2_200_ARTIFACT_DIR/corpus_200.json"
test -r "$AGENT_SUPPORT_M2_200_ARTIFACT_DIR/index_200.json"

export AGENT_SUPPORT_GPTOSS_CUDA_VISIBLE_DEVICES="${allocated[1]},${allocated[2]}"
export CUDA_VISIBLE_DEVICES="${allocated[0]}"
export AGENT_SUPPORT_GPTOSS_TWO_GPU=1
export AGENT_SUPPORT_GPU_MEMORY_DIAGNOSTICS=1
export AGENT_SUPPORT_GPTOSS_CUDA_CACHE_RELEASE=0
export PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
repo_root="${SLURM_SUBMIT_DIR:?Submit from the committed checkout}"
cd "$repo_root"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.free --format=csv

launch_stage=dispatch_phase
case "$THREE_GPU_PHASE" in
  startup_check)
    if [[ -e "$THREE_GPU_ROOT/startup_check" ]]; then
      echo "Startup-check output already exists; refusing to reuse: $THREE_GPU_ROOT/startup_check" >&2
      exit 1
    fi
    echo "Feasibility startup check: job=${SLURM_JOB_ID:-unset} root=$THREE_GPU_ROOT parent_gpu=$CUDA_VISIBLE_DEVICES worker_gpus=$AGENT_SUPPORT_GPTOSS_CUDA_VISIBLE_DEVICES" >&2
    launch_stage=startup_check_python
    "$VENV_PATH/bin/python" -u -m agent_support.three_gpu_feasibility_entry startup-check \
      --output-dir "$THREE_GPU_ROOT/startup_check"
    ;;
  overnight)
    : "${THREE_GPU_CACHE_JOB_ID:?Set the completed CPU cache job ID}"
    attempt="${THREE_GPU_ATTEMPT:?Overnight wrapper must create its unique attempt directory}"
    mapfile -t cache_results < <(find "$THREE_GPU_ROOT/cpu" -path "*/job_${THREE_GPU_CACHE_JOB_ID}_restart_*/cache/cache_result.json" -print | sort -V)
    test "${#cache_results[@]}" -gt 0
    cache_result="${cache_results[${#cache_results[@]}-1]}"
    args=(--output-dir "$attempt/run" --cache-result "$cache_result")
    if [[ -n "${THREE_GPU_MESSAGES_FILE:-}" ]]; then
      args+=(--messages-file "$THREE_GPU_MESSAGES_FILE")
    fi
    "$VENV_PATH/bin/python" -m agent_support.three_gpu_overnight gpu "${args[@]}"
    ;;
  performance)
    test ! -e "$THREE_GPU_ROOT/performance_probe"
    args=(--output-dir "$THREE_GPU_ROOT/performance_probe")
    if [[ -n "${THREE_GPU_MESSAGES_FILE:-}" ]]; then
      args+=(--messages-file "$THREE_GPU_MESSAGES_FILE")
    fi
    "$VENV_PATH/bin/python" -m agent_support.three_gpu_performance_probe "${args[@]}"
    ;;
  feasibility)
    if [[ -e "$THREE_GPU_ROOT/feasibility_8887" ]]; then
      echo "Feasibility output already exists; refusing to reuse: $THREE_GPU_ROOT/feasibility_8887" >&2
      exit 1
    fi
    echo "Feasibility launch: job=${SLURM_JOB_ID:-unset} root=$THREE_GPU_ROOT parent_gpu=$CUDA_VISIBLE_DEVICES worker_gpus=$AGENT_SUPPORT_GPTOSS_CUDA_VISIBLE_DEVICES python=$VENV_PATH/bin/python" >&2
    args=(--output-dir "$THREE_GPU_ROOT/feasibility_8887")
    if [[ -n "${THREE_GPU_MESSAGES_FILE:-}" ]]; then
      args+=(--messages-file "$THREE_GPU_MESSAGES_FILE")
    fi
    launch_stage=feasibility_python
    "$VENV_PATH/bin/python" -u -m agent_support.three_gpu_feasibility_entry test "${args[@]}"
    ;;
  smoke|batch)
    "$VENV_PATH/bin/python" -m agent_support.three_gpu_feasibility gate \
      --result "$THREE_GPU_ROOT/feasibility_8887/feasibility_result.json"
    export RECOVERY_ROOT="$THREE_GPU_ROOT/evaluation"
    : "${RECOVERY_ARM:?Set expanded_v3_v1 or autonomous_recovery_v4}"
    if [[ "$THREE_GPU_PHASE" == smoke ]]; then
      "$VENV_PATH/bin/python" -m agent_support.recovery_v4_eval smoke \
        --root "$RECOVERY_ROOT" --arm "$RECOVERY_ARM" --count 2
    else
      : "${RECOVERY_BATCH:?Set batch 0 through 9}"
      "$VENV_PATH/bin/python" -m agent_support.recovery_v4_eval run-batch \
        --root "$RECOVERY_ROOT" --arm "$RECOVERY_ARM" --batch "$RECOVERY_BATCH" \
        --max-tasks-per-attempt 3
    fi
    ;;
  *) echo "Unknown THREE_GPU_PHASE: $THREE_GPU_PHASE" >&2; exit 2 ;;
esac
