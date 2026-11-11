# Experiment 1: simple TextCraft support-selection policies

This is a development-only, paired 20-task comparison on the committed
`manifests/textcraft/development.json`. It does not read held-out outcomes.
The four committed configs are `configs/experiment1_textcraft_dev_{never,always,random,failure_triggered}.json`.
They differ only in `selection_policy`; all pin the same Qwen revision,
agent seed, greedy generation, reconsideration support version, task order,
and per-episode limits. Random uses probability 0.25 and policy seed 20261110.
Failure-triggered uses one prior failed environment action and zero cooldown.
The support-call ceiling is 10 **per task**, so realized calls may differ
across policies even when their configured budgets match.

## Gate: reconcile the completed failure-triggered smoke run

The attached `slurm_142728` smoke trace was audited: 50 selected decisions,
30 support calls, 107 model calls, and 27 `failure_threshold_not_met`
decisions. All 20 selected-but-not-invoked calls were marked
`support_call_budget`: 12 for task seed 51, one for seed 126, and seven for
seed 137. Each task invoked support exactly ten times. The saved config and
task hashes, call counts, and summary totals agreed. The
trace audit below independently requires any selected-but-not-invoked calls to
be accounted for by `support_call_budget` or `model_call_budget` blocks. A
`support_token_budget` block happens after a support invocation and cannot
explain that difference. The audit also checks each selection against only
the previous environment action's failed status.

After fast-forwarding the separate Locust framework checkout to the protocol
commit below, set the existing data and virtual-environment paths. Discover
the unique completed smoke run matching
the supplied totals, then audit it. **Do not submit Experiment 1 if the audit
fails or finds more than one candidate.**

```bash
export AGENT_SUPPORT_DATA="$HOME/agent_support_data"
export VENV_PATH="$HOME/AgenticSDS/.venv"
SMOKE_RUN="$("$VENV_PATH/bin/python" - <<'PY'
import os
from pathlib import Path
from agent_support.common import read_json
root = Path(os.environ["AGENT_SUPPORT_DATA"]) / "modular_results"
matches = []
for path in root.glob("textcraft_slurm_*"):
    try:
        run, config, summary = (read_json(path / name) for name in
                                ("run.json", "config.json", "summary.json"))
    except (FileNotFoundError, ValueError):
        continue
    if (run.get("status") == "complete" and run.get("limit") == 3
            and config.get("selection_policy", {}).get("id") == "failure_triggered_v1"
            and summary.get("support_calls") == 30 and summary.get("model_calls") == 107):
        matches.append(path)
if len(matches) != 1:
    raise SystemExit(f"Expected one failure smoke run, found {len(matches)}: {matches}")
print(matches[0])
PY
)"
"$VENV_PATH/bin/python" -m agent_support.experiment1_report audit-smoke "$SMOKE_RUN" \
  --expect-selections 50 --expect-calls 30 --expect-model-calls 107 --expect-not-met 27
```

## Preflight and sequential submission

First push this protocol commit to `modular_framework`, then fast-forward the
separate Locust checkout. Inspect `sinfo -N -h -p standby -o '%N %G %f'` and
the allocation for a typed L40S GRES. The commands below use the historical
`hassani`/`standby` allocation and `gpu:l40s:1`; use the exact account,
partition, QoS, and GRES names reported by your cluster if different. Check
the partition's `MaxTime` and choose a time sufficient for 20 tasks based on
the completed smoke durations. No job is submitted by reading this document.

```bash
cd "$HOME/AgenticSDS-framework"
git switch modular_framework
git pull --ff-only origin modular_framework
test -z "$(git status --porcelain)"
export AGENT_SUPPORT_DATA="$HOME/agent_support_data"
export HF_HOME="$AGENT_SUPPORT_DATA/hf_cache"
export VENV_PATH="$HOME/AgenticSDS/.venv"
export AGENT_SUPPORT_OUTPUT_ROOT="$AGENT_SUPPORT_DATA/experiment1_textcraft_development"
unset AGENT_SUPPORT_LIMIT
mkdir -p "$AGENT_SUPPORT_OUTPUT_ROOT" "$AGENT_SUPPORT_DATA/experiment1_logs"
sinfo -N -h -p standby -o '%N %G %f' | grep -i l40s
sinfo -N -h -p standby -o '%G' | grep -Eiq 'gpu:l40s:[0-9]+' || {
  echo 'No typed L40S GRES found on standby; inspect allocation before submission' >&2
  exit 1
}
scontrol show partition standby | grep -E 'MaxTime|AllowAccounts|AllowQos'
```

Run the smoke audit above **after** the checkout has the experiment report
command and **before** submission. Then validate the four configs:

```bash
"$VENV_PATH/bin/python" - <<'PY'
from agent_support.common import read_json, validate_config, validate_manifest
from agent_support.run_store import audit_partitions
base = read_json('configs/experiment1_textcraft_dev_never.json')
for name in ('never', 'always', 'random', 'failure_triggered'):
    config = read_json(f'configs/experiment1_textcraft_dev_{name}.json')
    manifest = read_json(config['manifest'])
    validate_manifest(manifest)
    validate_config(config, manifest['benchmark'])
    audit_partitions(config, manifest)
    assert len(manifest['tasks']) == 20 and manifest['partition'] == 'development'
    assert {k: v for k, v in config.items() if k != 'selection_policy'} == {
        k: v for k, v in base.items() if k != 'selection_policy'}
    print(name, config['selection_policy'])
PY
```

The following sequence requests **one typed L40S GPU per job**. Slurm's
`afterok` dependency starts each job only after the prior job succeeds. A
failed predecessor stops the chain for inspection. Set `WALLTIME` only after
checking the partition limit and smoke durations. Each run gets a unique
Slurm job-ID directory under the isolated experiment root.

```bash
export WALLTIME='04:00:00'
submit_experiment1() {
  local policy="$1" predecessor="${2:-}"
  export AGENT_SUPPORT_CONFIG="$PWD/configs/experiment1_textcraft_dev_${policy}.json"
  if [ -n "$predecessor" ]; then
    sbatch --parsable --export=ALL --dependency="afterok:$predecessor" \
      --account=hassani --partition=standby --qos=standby \
      --gres=gpu:l40s:1 --cpus-per-task=4 --mem=32G --time="$WALLTIME" \
      --output="$AGENT_SUPPORT_DATA/experiment1_logs/%j.out" --open-mode=append \
      scripts/locust_modular_job.sh
  else
    sbatch --parsable --export=ALL \
      --account=hassani --partition=standby --qos=standby \
      --gres=gpu:l40s:1 --cpus-per-task=4 --mem=32G --time="$WALLTIME" \
      --output="$AGENT_SUPPORT_DATA/experiment1_logs/%j.out" --open-mode=append \
      scripts/locust_modular_job.sh
  fi
}
NEVER_JOB="$(submit_experiment1 never)"; NEVER_JOB="${NEVER_JOB%%;*}"
ALWAYS_JOB="$(submit_experiment1 always "$NEVER_JOB")"; ALWAYS_JOB="${ALWAYS_JOB%%;*}"
RANDOM_JOB="$(submit_experiment1 random "$ALWAYS_JOB")"; RANDOM_JOB="${RANDOM_JOB%%;*}"
FAILURE_JOB="$(submit_experiment1 failure_triggered "$RANDOM_JOB")"; FAILURE_JOB="${FAILURE_JOB%%;*}"
printf 'never=%s always=%s random=%s failure=%s\n' \
  "$NEVER_JOB" "$ALWAYS_JOB" "$RANDOM_JOB" "$FAILURE_JOB"
printf 'EXPERIMENT1_COMMIT=%s\nNEVER_JOB=%s\nALWAYS_JOB=%s\nRANDOM_JOB=%s\nFAILURE_JOB=%s\n' \
  "$(git rev-parse HEAD)" "$NEVER_JOB" "$ALWAYS_JOB" "$RANDOM_JOB" "$FAILURE_JOB" \
  > "$AGENT_SUPPORT_DATA/experiment1_job_ids.env"
chmod 600 "$AGENT_SUPPORT_DATA/experiment1_job_ids.env"
```

## Analysis and report

After all four runs finish, use their exact job IDs. The command rejects
incomplete, limited, dirty, mismatched, or changed artifacts. It writes a
new `report.json`, `report.md`, and `task_comparisons.csv` without changing
the run directories. Use a fresh report directory on each analysis attempt.

```bash
source "$AGENT_SUPPORT_DATA/experiment1_job_ids.env"
ROOT="$AGENT_SUPPORT_OUTPUT_ROOT"
"$VENV_PATH/bin/python" -m agent_support.experiment1_report report \
  --never "$ROOT/textcraft_slurm_${NEVER_JOB}" \
  --always "$ROOT/textcraft_slurm_${ALWAYS_JOB}" \
  --random "$ROOT/textcraft_slurm_${RANDOM_JOB}" \
  --failure-triggered "$ROOT/textcraft_slurm_${FAILURE_JOB}" \
  --output-dir "$AGENT_SUPPORT_DATA/experiment1_reports/first_20_tasks"
```

The Markdown report has one row per policy with success, selections, actual
calls, blocked selections, model calls, tokens, failed actions, actions
changed by support, and inference seconds. A second table gives paired
helped/hurt counts and changes from never support. JSON contains 95% Wilson
intervals for each success rate, deterministic paired task-bootstrap 95%
intervals for success and compute differences, and per-task compute intervals.
The CSV has all 20 tasks under each policy with baseline-aligned differences.
The intervals are descriptive; 20 tasks offer limited statistical power.
