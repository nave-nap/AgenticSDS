#!/usr/bin/env bash
#SBATCH --job-name=adapt-textcraft-dev
#SBATCH --output=slurm-%j.out
set -euo pipefail
umask 077

if [[ $# -ne 2 || ( "$1" != "smoke" && "$1" != "full" ) ||
      ( "$2" != "baseline" && "$2" != "adapt_dynamic_replanning_v1" ) ]]; then
  echo "usage: locust_adapt_dev.sh {smoke|full} {baseline|adapt_dynamic_replanning_v1}" >&2
  exit 2
fi

mode="$1"
condition="$2"
repo_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
: "${AGENT_SUPPORT_DATA:?Export AGENT_SUPPORT_DATA to the transferred data directory}"
venv_path="${VENV_PATH:-$repo_root/.venv}"
export PYTHONHASHSEED=0
export HF_HOME="${HF_HOME:-$AGENT_SUPPORT_DATA/hf_cache}"
cd "$repo_root"
"$venv_path/bin/python" -m agent_support.adapt_development preflight --data-root "$AGENT_SUPPORT_DATA"
args=(--config "$AGENT_SUPPORT_DATA/adapt_development/configs/$condition.json"
      --manifest "$AGENT_SUPPORT_DATA/manifests/textcraft/development.json"
      --output-root "$AGENT_SUPPORT_DATA/adapt_development/results")
if [[ "$mode" == "smoke" ]]; then
  args+=(--limit 3)
fi
"$venv_path/bin/python" -m agent_support.runner "${args[@]}"
