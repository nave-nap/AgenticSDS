#!/usr/bin/env bash
#SBATCH --job-name=autonomous-support-eval
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=02:00:00
set -euo pipefail
umask 077

repo_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
: "${AGENT_SUPPORT_DATA:?Set AGENT_SUPPORT_DATA to the external evaluation data directory}"
: "${EVAL_ARM:?Set EVAL_ARM to an evaluation arm}"
: "${EVAL_STAGE:?Set EVAL_STAGE to smoke, pilot, or full}"
venv_path="${VENV_PATH:-$repo_root/.venv}"
test -x "$venv_path/bin/python" || { echo "Missing Qwen Python: $venv_path/bin/python" >&2; exit 1; }
export PYTHONHASHSEED=0
export HF_HOME="${HF_HOME:-$AGENT_SUPPORT_DATA/hf_cache}"
export AGENT_SUPPORT_M2_200_ARTIFACT_DIR="${AGENT_SUPPORT_M2_200_ARTIFACT_DIR:-/shared_data0/evanpan/agent_support/gptoss_memory_v1/memory_scaling_v1/artifacts}"
if [[ "$EVAL_STAGE" == "smoke" ]]; then
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
fi
if [[ "$EVAL_ARM" == "random_support" || "$EVAL_ARM" == "autonomous" || "$EVAL_ARM" == "fixed_reasoning_2" ]]; then
  : "${AGENT_SUPPORT_GPTOSS_PYTHON:?Set AGENT_SUPPORT_GPTOSS_PYTHON to the pinned GPT-OSS Python executable}"
  test -x "$AGENT_SUPPORT_GPTOSS_PYTHON" || { echo "GPT-OSS Python is not executable: $AGENT_SUPPORT_GPTOSS_PYTHON" >&2; exit 1; }
  export AGENT_SUPPORT_GPTOSS_HF_HOME="${AGENT_SUPPORT_GPTOSS_HF_HOME:-$HF_HOME}"
fi
mkdir -p "$HF_HOME" "$AGENT_SUPPORT_DATA/autonomous_support_evaluation/runs"
cd "$repo_root"

telemetry_pid=""
if [[ "$EVAL_STAGE" == "smoke" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  telemetry_dir="$AGENT_SUPPORT_DATA/autonomous_support_evaluation/telemetry"
  mkdir -p "$telemetry_dir"
  telemetry_file="$telemetry_dir/${EVAL_ARM}-smoke-${SLURM_JOB_ID:-manual}.csv"
  nvidia-smi --query-gpu=timestamp,name,memory.total,memory.used --format=csv -l 2 > "$telemetry_file" 2>&1 &
  telemetry_pid="$!"
  echo "GPU telemetry: $telemetry_file"
fi
cleanup_telemetry() {
  if [[ -n "$telemetry_pid" ]]; then
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
  fi
}
trap cleanup_telemetry EXIT

arguments=(run --arm "$EVAL_ARM" --stage "$EVAL_STAGE"
           --output-root "$AGENT_SUPPORT_DATA/autonomous_support_evaluation/runs")
if [[ -n "${EVAL_SMOKE_RUN:-}" ]]; then
  arguments+=(--smoke-run "$EVAL_SMOKE_RUN")
fi
if [[ -n "${EVAL_GATE_FILE:-}" ]]; then
  arguments+=(--gate-file "$EVAL_GATE_FILE")
fi
"$venv_path/bin/python" -m agent_support.autonomous_evaluation "${arguments[@]}"
