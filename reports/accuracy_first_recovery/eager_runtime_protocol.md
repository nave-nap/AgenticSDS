# Four-policy eager-attention evaluation

This branch changes only the four-policy evaluator's runtime selection,
runtime checks, Slurm environment, and offline tests. The protocol is
`textcraft_recovery_policy30_eager_terminal_review_v1`. The frozen manifest,
four selectors, memory corpus, Qwen and GPT-OSS revisions and decoding,
environment-action limits, and scoring remain as in commit `701c25d`.
The old FlexAttention experiment roots and artifacts are not inputs to this
protocol and must remain untouched.

The paired bounded diagnostic reported FlexAttention 9/10 generations with
an illegal access in block-mask compilation and eager 10/10, with Qwen
co-resident. Eager reserved about 34.2 GiB at peak versus 13.85 GiB for
FlexAttention, so the first real episode is a memory feasibility gate, not
evidence of long-run stability. This branch does not modify GPU kernels.

The Slurm scripts explicitly set `AGENT_SUPPORT_GPTOSS_FLEX_ATTENTION=0`,
`AGENT_SUPPORT_GPTOSS_CUDA_CACHE_RELEASE=0`,
`AGENT_SUPPORT_GPU_MEMORY_DIAGNOSTICS=1`,
`AGENT_SUPPORT_GPTOSS_WORKER_TRACEBACKS=1`, and
`CUDA_LAUNCH_BLOCKING=1` for every arm. CPU preflight requires those exact
settings. The evaluator requires the GPT-OSS worker's active backend to be
`eager` before loading Qwen or starting a task; run metadata and the locked
plan record the request and active backend. A mismatch is a run failure.

## Deployment and CPU-only Locust preflight

The implementation commit must first be made available in the Locust clone
through an authorized transfer. The following commands assume the branch
has been published there; they do not publish code or submit a GPU job.
From the Mac:

```bash
cd /private/tmp/AgenticSDS-accuracy-eager
test -z "$(git status --porcelain)"
export EAGER_COMMIT="$(git rev-parse HEAD)"
export LOCUST_HOST=locust-login.seas.upenn.edu
ssh "$LOCUST_HOST" "cd /home/evanpan/AgenticSDS && git fetch origin accuracy-first-eval-eager && git cat-file -e '$EAGER_COMMIT^{commit}' && test ! -e /home/evanpan/AgenticSDS-accuracy-eager && git worktree add --detach /home/evanpan/AgenticSDS-accuracy-eager $EAGER_COMMIT"
ssh "$LOCUST_HOST"
```

On the Locust login node:

```bash
set -euo pipefail
export DEPLOY_REPO=/home/evanpan/AgenticSDS-accuracy-eager
export RECOVERY_ROOT=/shared_data0/evanpan/agent_support/recovery_policy30_eager_terminal_review_v1
export VENV_PATH=/home/evanpan/AgenticSDS/.venv
export AGENT_SUPPORT_GPTOSS_PYTHON=/shared_data0/evanpan/agent_support/venvs/gptoss20b/bin/python
export AGENT_SUPPORT_GPTOSS_HF_HOME=/shared_data0/evanpan/agent_support/hf_cache
export HF_HOME="$AGENT_SUPPORT_GPTOSS_HF_HOME"
export AGENT_SUPPORT_M2_200_ARTIFACT_DIR=/shared_data0/evanpan/agent_support/gptoss_memory_v1/memory_scaling_v1/artifacts
export AGENT_SUPPORT_GPTOSS_FLEX_ATTENTION=0
export AGENT_SUPPORT_GPTOSS_CUDA_CACHE_RELEASE=0
export AGENT_SUPPORT_GPU_MEMORY_DIAGNOSTICS=1
export AGENT_SUPPORT_GPTOSS_WORKER_TRACEBACKS=1
export CUDA_LAUNCH_BLOCKING=1
export GPU_PARTITION="$(sacct -j 164919 -X -n -o Partition | awk 'NF {print $1; exit}')"
test -n "$GPU_PARTITION"
test -d "$DEPLOY_REPO"
test ! -e "$RECOVERY_ROOT"
test -x "$VENV_PATH/bin/python"
test -x "$AGENT_SUPPORT_GPTOSS_PYTHON"
test -r "$AGENT_SUPPORT_M2_200_ARTIFACT_DIR/corpus_200.json"
test -r "$AGENT_SUPPORT_M2_200_ARTIFACT_DIR/index_200.json"
cd "$DEPLOY_REPO"
test -z "$(git status --porcelain)"
bash -n scripts/recovery_policy_stage.sbatch scripts/recovery_policy_batch.sbatch
for policy in original_v4 state_aware_v1 accuracy_first_v1 accuracy_first_terminal_v1; do
  "$VENV_PATH/bin/python" -m agent_support.recovery_policy_eval preflight \
    --policy "$policy" --strict-runtime
done
```

If `sacct` no longer identifies an L40S partition, inspect
`sinfo -o '%P %G %m %l'` and set `GPU_PARTITION` to a verified L40S
partition. Preflight loads no GPU models.

## First real-model gate: one terminal-policy episode

Only after authorization to submit GPU jobs, run this first. The frozen
smoke task is TextCraft seed 51. `runtime-gate` verifies a complete,
infrastructure-valid episode and both GPT-OSS and Qwen memory logs. It
reports peak reserved bytes, minimum observed global free bytes, and
whether terminal review appeared in decision telemetry. It does not require
task success or assume terminal review was triggered by every stochastic
controller trajectory.

```bash
mkdir -p "$RECOVERY_ROOT/logs"
sbatch --wait --partition="$GPU_PARTITION" \
  --export=ALL,RECOVERY_POLICY=accuracy_first_terminal_v1,RECOVERY_STAGE=smoke \
  --output="$RECOVERY_ROOT/logs/gate-terminal-%j.out" \
  scripts/recovery_policy_stage.sbatch
"$VENV_PATH/bin/python" -m agent_support.recovery_policy_eval runtime-gate \
  --root "$RECOVERY_ROOT" | tee "$RECOVERY_ROOT/logs/runtime-gate.json"
```

Inspect the gate run's `run.json`, episode, `terminal_reviews`, worker log,
and memory JSONL. A false `terminal_review_observed` means this episode did
not exercise that path; the offline terminal-review regression tests still
apply, but do not claim real-model terminal-review validation. If the gate
OOMs or is otherwise infrastructure-invalid, stop. Preserve the failed root,
report the failing request, token counts, worker traceback, and memory
snapshots. The next workaround to test is GPT-OSS and Qwen on separate GPUs
in a new protocol, without further FlexAttention kernel debugging.

## Matched smoke, pilot, and 30-task evaluation

Proceed only after the gate is infrastructure-valid. Each `stage-report`
checks one arm immediately; the final call checks all four. A fatal worker
or backend failure invalidates the stage. Use a new root for a rerun, and
never retry a fatal CUDA request in the same worker.

```bash
for policy in original_v4 state_aware_v1 accuracy_first_v1; do
  sbatch --wait --partition="$GPU_PARTITION" \
    --export=ALL,RECOVERY_POLICY="$policy",RECOVERY_STAGE=smoke \
    --output="$RECOVERY_ROOT/logs/smoke-$policy-%j.out" \
    scripts/recovery_policy_stage.sbatch
  "$VENV_PATH/bin/python" -m agent_support.recovery_policy_eval stage-report \
    --root "$RECOVERY_ROOT" --stage smoke --policy "$policy"
done
"$VENV_PATH/bin/python" -m agent_support.recovery_policy_eval stage-report \
  --root "$RECOVERY_ROOT" --stage smoke

for policy in original_v4 state_aware_v1 accuracy_first_v1 accuracy_first_terminal_v1; do
  sbatch --wait --partition="$GPU_PARTITION" \
    --export=ALL,RECOVERY_POLICY="$policy",RECOVERY_STAGE=pilot \
    --output="$RECOVERY_ROOT/logs/pilot-$policy-%j.out" \
    scripts/recovery_policy_stage.sbatch
  "$VENV_PATH/bin/python" -m agent_support.recovery_policy_eval stage-report \
    --root "$RECOVERY_ROOT" --stage pilot --policy "$policy"
done
"$VENV_PATH/bin/python" -m agent_support.recovery_policy_eval stage-report \
  --root "$RECOVERY_ROOT" --stage pilot

"$VENV_PATH/bin/python" -m agent_support.recovery_policy_eval init \
  --root "$RECOVERY_ROOT"
for batch in 0 1 2 3 4 5 6 7 8 9; do
  for policy in original_v4 state_aware_v1 accuracy_first_v1 accuracy_first_terminal_v1; do
    sbatch --wait --partition="$GPU_PARTITION" \
      --export=ALL,RECOVERY_POLICY="$policy",RECOVERY_BATCH="$batch" \
      --output="$RECOVERY_ROOT/logs/batch-$batch-$policy-%j.out" \
      scripts/recovery_policy_batch.sbatch
    "$VENV_PATH/bin/python" -m agent_support.recovery_policy_eval status \
      --root "$RECOVERY_ROOT" --policy "$policy" --batch "$batch" \
      --require-complete
  done
done
mkdir -p "$RECOVERY_ROOT/reports"
for policy in original_v4 state_aware_v1 accuracy_first_v1 accuracy_first_terminal_v1; do
  "$VENV_PATH/bin/python" -m agent_support.recovery_policy_eval aggregate \
    --root "$RECOVERY_ROOT" --policy "$policy" \
    --output "$RECOVERY_ROOT/reports/$policy.json"
done
"$VENV_PATH/bin/python" -m agent_support.recovery_policy_eval compare \
  --root "$RECOVERY_ROOT" \
  --output "$RECOVERY_ROOT/reports/paired_four_policy.json"
```

All arms use identical task indices and seeds, model weights and decoding,
limits, backend, and success criterion. Primary score is environment-verified
success rate. Report recovery success, observed progress, intervention
frequency, strategy changes, support calls, tokens, and time separately.
