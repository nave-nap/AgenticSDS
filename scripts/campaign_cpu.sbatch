#!/usr/bin/env bash
#SBATCH --job-name=support-campaign-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
set -euo pipefail
umask 077
: "${CAMPAIGN_CPU_ROOT:?Set a fresh CPU campaign output root}"
: "${CAMPAIGN_CPU_PHASE:?Set integrity, policy_source, or trajectory}"
: "${VENV_PATH:?Set the pinned Qwen Python environment}"
repo_root="${SLURM_SUBMIT_DIR:?Submit from a clean campaign checkout}"
test -x "$VENV_PATH/bin/python"
cd "$repo_root"
test -z "$(git status --porcelain)"
attempt="$CAMPAIGN_CPU_ROOT/${CAMPAIGN_CPU_PHASE}_job_${SLURM_JOB_ID:?}_restart_${SLURM_RESTART_COUNT:-0}"
mkdir "$attempt"
case "$CAMPAIGN_CPU_PHASE" in
  integrity)
    "$VENV_PATH/bin/python" -m agent_support.campaign_offline --output "$attempt/result.json" integrity
    ;;
  policy_source)
    "$VENV_PATH/bin/python" -m agent_support.campaign_offline --output "$attempt/result.json" policy-source
    ;;
  trajectory)
    : "${CAMPAIGN_TRAJECTORY_ROOT:?Set a read-only prior-run root}"
    "$VENV_PATH/bin/python" -m agent_support.campaign_offline --output "$attempt/result.json" \
      trajectory --root "$CAMPAIGN_TRAJECTORY_ROOT"
    ;;
  *) echo "Unknown CPU campaign phase: $CAMPAIGN_CPU_PHASE" >&2; exit 2 ;;
esac
"$VENV_PATH/bin/python" -c 'import json,sys; x=json.load(open(sys.argv[1])); print(x["kind"], x["scientific_status"])' "$attempt/result.json"
