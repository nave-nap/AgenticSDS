#!/usr/bin/env bash
#SBATCH --job-name=agent-support
#SBATCH --output=slurm-%j.out
set -euo pipefail
umask 077

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${AGENT_SUPPORT_DATA:?Export AGENT_SUPPORT_DATA to the transferred data directory}"
venv_path="${VENV_PATH:-$repo_root/.venv}"
export PYTHONHASHSEED=0
export ALFWORLD_DATA="${ALFWORLD_DATA:-$HOME/.cache/alfworld}"
export HF_HOME="${HF_HOME:-$AGENT_SUPPORT_DATA/hf_cache}"
mkdir -p "$HF_HOME"
cd "$repo_root"
"$venv_path/bin/python" -m agent_support.cluster_workflow "$@"
