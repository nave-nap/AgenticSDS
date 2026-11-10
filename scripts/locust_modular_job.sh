#!/usr/bin/env bash
#SBATCH --job-name=agentic-modular
#SBATCH --requeue
#SBATCH --signal=B:TERM@60
set -euo pipefail
umask 077

: "${AGENT_SUPPORT_DATA:?Set the durable data directory before sbatch}"
: "${AGENT_SUPPORT_CONFIG:?Set the schema-v2 config path before sbatch}"
: "${SLURM_JOB_ID:?This script must run inside a Slurm job}"

repo_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
venv_path="${VENV_PATH:-$repo_root/.venv}"
result_root="$AGENT_SUPPORT_DATA/modular_results"
run_key="slurm_${SLURM_JOB_ID}"
cd "$repo_root"
benchmark="$("$venv_path/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["benchmark"])' "$AGENT_SUPPORT_CONFIG")"
run_directory="$result_root/${benchmark}_$run_key"
export PYTHONHASHSEED=0
export HF_HOME="${HF_HOME:-$AGENT_SUPPORT_DATA/hf_cache}"
mkdir -p "$result_root" "$HF_HOME"

resume_args=()
if [[ -e "$run_directory" ]]; then
  resume_args=(--resume)
fi
"$venv_path/bin/python" -m agent_support.runner \
  --config "$AGENT_SUPPORT_CONFIG" \
  --output-root "$result_root" \
  --run-key "$run_key" \
  "${resume_args[@]}"
