# Matched three-policy recovery evaluation protocol

## What is locked

The new CLI is `python -m agent_support.recovery_policy_eval`. It is separate
from the frozen `recovery_v4_eval` CLI and always requires an explicit
`--policy` for execution and reports. Its registry contains `original_v4`
(passes `policy=None` to the unchanged v4 episode path), `state_aware_v1`, and
`accuracy_first_v1`. The original v4 config file, 30-task manifest, recipe
order, M2-200 corpus and index, model revisions, decoding settings, dependency
pins, and limits are reused without modification. There is no cost term in the
success score.

Each run stores `run.json`, `config.json`, `policy.json`, and `manifest.json`.
The run metadata records the implementation commit, policy ID/configuration
and digest, model settings, runtime package versions, memory digests/paths,
manifest digest, recipe order, limits, indices, and worker GPU metadata. Every
episode and checkpoint has `decision_telemetry`: visible observation/history,
failure signals, policy interventions, support requests/results, candidate
actions and verification, final action, environment outcome/progress evidence,
controller errors and fallbacks, and fatal/infrastructure signals. The raw v4
trace remains available. Controller parse fallback is recorded separately
from a policy intervention and from a backend failure.

The 30 development tasks are paired by task ID and seed. All policies use the
same initial observation, environment-action limit (150), Qwen/GPT-OSS
weights, revisions, decoding, memory, and runtime image. The one-task smoke
uses index 0; the matched pilot uses indices 0–2. A new full-run plan is
initialized only after all three smoke runs and all three pilot runs pass the
CPU-only stage gate. Full runs use 10 batches of 3 tasks per policy. Smoke,
pilot, and full outputs live under separate subdirectories and policy IDs.

The primary metric is environment-verified success across all 30 tasks. The
`compare` command refuses an incomplete or infrastructure-contaminated arm.
It reports success differences, paired wins/losses, and deterministic paired
bootstrap intervals. The exploratory development gate is at least two more
successes for accuracy-first than **each** comparator. Secondary measures are
success after at least one failed action, observed `Got`/`Crafted` progress
actions, model `progressed` judgments, decisions with policy interventions,
strategy changes, and replanning. Support calls, model calls, tokens, and
elapsed/inference seconds are reported independently of success. These 30
development tasks do not establish held-out generalization.

Fatal CUDA OOM/unavailable-worker statuses stop the current run. Any worker
fatality, backend error, support backend error, model generation failure,
or environment error marks the episode infrastructure-invalid and stops the
new evaluator after saving that episode. Ordinary controller parse fallbacks
remain separately counted. A batch with a recorded infrastructure stop or
runner failure cannot be resumed silently. Repair the runtime, document
the changed commit/protocol, and repeat matched arms before comparing.

## Local CPU checks (no GPU job)

```bash
cd /private/tmp/AgenticSDS-accuracy-eval
/Users/evanpan/AgenticSDS/.venv/bin/python -m pytest -q tests
bash -n scripts/recovery_policy_stage.sbatch scripts/recovery_policy_batch.sbatch
git diff --check
for policy in original_v4 state_aware_v1 accuracy_first_v1; do
  /Users/evanpan/AgenticSDS/.venv/bin/python -m agent_support.recovery_policy_eval preflight --policy "$policy"
done
```

`preflight` only checks files, manifests, parser protocol, and installed
packages. On the Mac, missing Qwen/GPT-OSS GPU dependencies or worker paths
appear as false runtime checks; that does not launch or validate inference.
All runtime checks, parser checks, and pinned memory checks must pass on
Locust before the first stage run.

## Locust deployment (commands for the operator; none submitted here)

First publish this committed branch from the Mac, then create a clean detached
Locust worktree at that exact commit:

```bash
cd /private/tmp/AgenticSDS-accuracy-eval
git status --porcelain
export POLICY_COMMIT="$(git rev-parse HEAD)"
git push origin accuracy-first-eval
export LOCUST_HOST=locust-login.seas.upenn.edu
ssh "$LOCUST_HOST" "cd /home/evanpan/AgenticSDS && git fetch origin accuracy-first-eval && test ! -e /home/evanpan/AgenticSDS-accuracy-first-eval && git worktree add --detach /home/evanpan/AgenticSDS-accuracy-first-eval $POLICY_COMMIT"
ssh "$LOCUST_HOST"
```

On Locust:

```bash
export DEPLOY_REPO=/home/evanpan/AgenticSDS-accuracy-first-eval
export RECOVERY_ROOT=/shared_data0/evanpan/agent_support/recovery_policy30_v1
export VENV_PATH=/home/evanpan/AgenticSDS/.venv
export AGENT_SUPPORT_GPTOSS_PYTHON=/shared_data0/evanpan/agent_support/venvs/gptoss20b/bin/python
export AGENT_SUPPORT_GPTOSS_HF_HOME=/shared_data0/evanpan/agent_support/hf_cache
export HF_HOME="$AGENT_SUPPORT_GPTOSS_HF_HOME"
export AGENT_SUPPORT_M2_200_ARTIFACT_DIR=/shared_data0/evanpan/agent_support/gptoss_memory_v1/memory_scaling_v1/artifacts
export GPU_PARTITION="$(sacct -j 163640 -X -n -o Partition | awk 'NF {print $1; exit}')"
test -n "$GPU_PARTITION"
test -x "$VENV_PATH/bin/python"
test -x "$AGENT_SUPPORT_GPTOSS_PYTHON"
test -r "$AGENT_SUPPORT_M2_200_ARTIFACT_DIR/corpus_200.json"
test -r "$AGENT_SUPPORT_M2_200_ARTIFACT_DIR/index_200.json"
cd "$DEPLOY_REPO"
git status --porcelain
for policy in original_v4 state_aware_v1 accuracy_first_v1; do
  "$VENV_PATH/bin/python" -m agent_support.recovery_policy_eval preflight --policy "$policy"
done
```

If Slurm accounting no longer has job 163640, use `sinfo -o '%P %G %m %l'`
and set `GPU_PARTITION` to the verified L40S partition. Inspect all preflight
JSON records: parser/runtime checks must be true; memory must be available
with overlap zero. Do not continue if the GPT-OSS worker stability diagnostic
is unresolved.

### One-task smoke, then matched three-task pilot

Run one stage at a time and inspect all three results before advancing. The
stage gate checks exact indices, commit, settings, snapshots, and absence of
infrastructure-invalid episodes. Task failure alone does not block the gate.

```bash
mkdir -p "$RECOVERY_ROOT/logs"
for policy in original_v4 state_aware_v1 accuracy_first_v1; do
  sbatch --wait --partition="$GPU_PARTITION" --export=ALL,RECOVERY_POLICY="$policy",RECOVERY_STAGE=smoke \
    --output="$RECOVERY_ROOT/logs/smoke-$policy-%j.out" scripts/recovery_policy_stage.sbatch
done
# Wait for all three jobs, inspect worker logs, run.json, episodes, and checkpoints.
"$VENV_PATH/bin/python" -m agent_support.recovery_policy_eval stage-report \
  --root "$RECOVERY_ROOT" --stage smoke
for policy in original_v4 state_aware_v1 accuracy_first_v1; do
  sbatch --wait --partition="$GPU_PARTITION" --export=ALL,RECOVERY_POLICY="$policy",RECOVERY_STAGE=pilot \
    --output="$RECOVERY_ROOT/logs/pilot-$policy-%j.out" scripts/recovery_policy_stage.sbatch
done
# Wait for all three jobs and inspect traces and model/support accounting.
"$VENV_PATH/bin/python" -m agent_support.recovery_policy_eval stage-report \
  --root "$RECOVERY_ROOT" --stage pilot
```

`--wait` runs one stage job at a time, avoiding GPU contention among arms.
Inspect `sacct -j <job-id> --format=JobID,Elapsed,MaxRSS,AllocTRES,State` and
each run's `workers/` logs. If a stage fails, use a new output root and repeat
the matched stage for all three policies after the runtime issue is resolved.

### Initialize and run the matched 30-task comparison

```bash
test ! -e "$RECOVERY_ROOT/plan.json"
"$VENV_PATH/bin/python" -m agent_support.recovery_policy_eval init --root "$RECOVERY_ROOT"
for batch in 0 1 2 3 4 5 6 7 8 9; do
  for policy in original_v4 state_aware_v1 accuracy_first_v1; do
    "$VENV_PATH/bin/python" -m agent_support.recovery_policy_eval status \
      --root "$RECOVERY_ROOT" --policy "$policy" --batch "$batch"
    sbatch --wait --partition="$GPU_PARTITION" \
      --export=ALL,RECOVERY_POLICY="$policy",RECOVERY_BATCH="$batch" \
      --output="$RECOVERY_ROOT/logs/batch-$batch-$policy-%j.out" \
      scripts/recovery_policy_batch.sbatch
  done
done
```

Submit only batches with nonempty `missing_indices`. Each batch has a file
lock; completed task snapshots are never rerun. An interrupted task restarts
from its initial observation, while its partial checkpoint is kept. After all
runs, write reports to new files:

```bash
mkdir -p "$RECOVERY_ROOT/reports"
for policy in original_v4 state_aware_v1 accuracy_first_v1; do
  "$VENV_PATH/bin/python" -m agent_support.recovery_policy_eval aggregate \
    --root "$RECOVERY_ROOT" --policy "$policy" \
    --output "$RECOVERY_ROOT/reports/$policy.json"
done
"$VENV_PATH/bin/python" -m agent_support.recovery_policy_eval compare \
  --root "$RECOVERY_ROOT" --output "$RECOVERY_ROOT/reports/paired_three_policy.json"
```

No commands in this protocol were used to submit GPU work for this change.
