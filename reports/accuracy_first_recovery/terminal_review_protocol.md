# Accuracy-first terminal review: audit and controlled evaluation

## Decision lifecycle audit

In `run_recovery_episode`, Qwen proposes a TextCraft action and the v4
`adaptive_loop` asks GPT-OSS to execute, request support, reflect, or stop.
The original accuracy-first selector was called only for an `execute` choice.
Its state-aware support selection, prerequisite checks, and candidate
inspection therefore did not run when GPT-OSS first returned
`{"kind":"stop","stop_reason":"no_viable_strategy"}`. The `stop` branch
returned immediately with no environment action. The selector's failure
analysis, memory retrieval, and competing candidate generation were also
gated by `_stalled`, which is false before any failed action or progress
assessment. Action arbitration ran only after an `execute` choice. These are
implemented behaviors, not planned capabilities.

The new `accuracy_first_terminal_v1` policy keeps the previous selector and
adds a bounded review of ordinary `no_viable_strategy` claims. It records the
controller's reason and visible evidence, inspects state, retrieves frozen
memory when available, asks the existing reasoning support for two distinct
actions, inspects unique candidates, verifies craft prerequisites, and selects
an evidenced action. If inventory has never been observed, it may use a
checked `inventory` action to gather evidence. It accepts the stop after the
bounded search finds no supported alternative. Equivalent candidates are
deduplicated; investigated actions are tracked by visible-state fingerprint.
The existing seven-support and ten-depth limits remain in force. A search
limited by those safeguards is explicitly marked `accept_at_support_limit`.

The hook is invoked only for this policy's ordinary `no_viable_strategy`.
Unverified state updates attached to a stop claim are not applied while the
review requests support or rescues an action.
`budget_exhausted`, fatal CUDA/worker failures, environment-verified success,
and hard wall/action/model limits keep their existing paths. Parser fallbacks
remain separate from terminal review and infrastructure failures. Review
requests and action rescues appear in `terminal_review` trace events and
`terminal_reviews` decision telemetry.

Offline verification: the complete CPU suite passed 261 tests. `git diff --check`
and shell syntax checks for both Slurm scripts passed. Local CPU
preflight passed the four policy identities, frozen protocol checks, memory
checks, and FlexAttention/traceback opt-in flags. The local strict runtime
check remains false because the pinned Locust model runtimes are absent; run
the strict preflight on Locust before any GPU work.

## Policy isolation and next hypothesis

The new evaluator protocol is `textcraft_recovery_policy30_terminal_review_v1`.
It has four matched arms: `original_v4`, `state_aware_v1`, unchanged
`accuracy_first_v1`, and `accuracy_first_terminal_v1`. The legacy
accuracy-first policy configuration retains its prior digest. Frozen task
manifest, configurations, model revisions, decoding settings, pinned
dependencies, environment limits, and success criteria are unchanged.

Proactive deliberation is a separate, **unimplemented** hypothesis. A future
`accuracy_first_proactive_v1` arm could compare candidates before the first
environment failure when prerequisites or acquisition routes remain uncertain.
It should retain terminal review and add a separately logged proactive trigger.
Compare that arm against `accuracy_first_terminal_v1` on the same instances to
measure its incremental effect. No proactive trigger is enabled here.

## Mac-to-Locust deployment and CPU-only preflight

Run on the Mac after committing this branch. The SSH hostname shown in the
previous protocol did not resolve from the development workspace; set
`LOCUST_HOST` to the working cluster hostname if it differs. These commands do
not submit GPU jobs.

```bash
cd /private/tmp/AgenticSDS-accuracy-terminal-review
test -z "$(git status --porcelain)"
export POLICY_COMMIT="$(git rev-parse HEAD)"
git push origin accuracy-first-terminal-review
export LOCUST_HOST=locust-login.seas.upenn.edu
ssh "$LOCUST_HOST" "cd /home/evanpan/AgenticSDS && git fetch origin accuracy-first-terminal-review && test ! -e /home/evanpan/AgenticSDS-accuracy-first-terminal-review && git worktree add --detach /home/evanpan/AgenticSDS-accuracy-first-terminal-review $POLICY_COMMIT"
ssh "$LOCUST_HOST"
```

On the Locust login node, run the CPU-only checks first. The root is new and
does not touch `recovery_policy30_flex_runtime_v1` or its prior smoke files.

```bash
export DEPLOY_REPO=/home/evanpan/AgenticSDS-accuracy-first-terminal-review
export RECOVERY_ROOT=/shared_data0/evanpan/agent_support/recovery_policy30_terminal_review_v1
export VENV_PATH=/home/evanpan/AgenticSDS/.venv
export AGENT_SUPPORT_GPTOSS_PYTHON=/shared_data0/evanpan/agent_support/venvs/gptoss20b/bin/python
export AGENT_SUPPORT_GPTOSS_HF_HOME=/shared_data0/evanpan/agent_support/hf_cache
export HF_HOME="$AGENT_SUPPORT_GPTOSS_HF_HOME"
export AGENT_SUPPORT_M2_200_ARTIFACT_DIR=/shared_data0/evanpan/agent_support/gptoss_memory_v1/memory_scaling_v1/artifacts
export AGENT_SUPPORT_GPTOSS_FLEX_ATTENTION=1
export AGENT_SUPPORT_GPTOSS_WORKER_TRACEBACKS=1
export GPU_PARTITION="$(sacct -j 163640 -X -n -o Partition | awk 'NF {print $1; exit}')"
test -n "$GPU_PARTITION"
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

If the accounting query no longer identifies the L40S partition, use
`sinfo -o '%P %G %m %l'` to select the verified L40S partition. The worker
checks the active attention backend at model initialization; this CPU-only
preflight checks the requested settings and pinned runtime prerequisites.

## Fresh matched smoke, pilot, and full comparison

These are **prepared commands only**. Submit GPU jobs only after authorization.
Run one job at a time to avoid contention. The smoke uses TextCraft seed 51;
the pilot uses the first three frozen tasks. If any arm has a fatal worker or
backend failure, the stage is invalid and must be rerun for all arms in a new
root after diagnosis. Task failure alone does not invalidate a stage.

```bash
mkdir -p "$RECOVERY_ROOT/logs"
for policy in original_v4 state_aware_v1 accuracy_first_v1 accuracy_first_terminal_v1; do
  sbatch --wait --partition="$GPU_PARTITION" \
    --export=ALL,RECOVERY_POLICY="$policy",RECOVERY_STAGE=smoke \
    --output="$RECOVERY_ROOT/logs/smoke-$policy-%j.out" \
    scripts/recovery_policy_stage.sbatch
done
"$VENV_PATH/bin/python" -m agent_support.recovery_policy_eval stage-report \
  --root "$RECOVERY_ROOT" --stage smoke
for policy in original_v4 state_aware_v1 accuracy_first_v1 accuracy_first_terminal_v1; do
  sbatch --wait --partition="$GPU_PARTITION" \
    --export=ALL,RECOVERY_POLICY="$policy",RECOVERY_STAGE=pilot \
    --output="$RECOVERY_ROOT/logs/pilot-$policy-%j.out" \
    scripts/recovery_policy_stage.sbatch
done
"$VENV_PATH/bin/python" -m agent_support.recovery_policy_eval stage-report \
  --root "$RECOVERY_ROOT" --stage pilot
```

Inspect each run's `run.json`, episode, checkpoint, decision telemetry, and
worker traceback before advancing. Confirm all four arms record the same
manifest hash, model settings, implementation commit, and active
`flex_attention` backend. For the terminal arm, inspect the first decision's
`terminal_reviews`, support outputs, candidate evaluations, and final action.

Only after both stages pass, initialize the 30-task matched comparison:

```bash
"$VENV_PATH/bin/python" -m agent_support.recovery_policy_eval init --root "$RECOVERY_ROOT"
for batch in 0 1 2 3 4 5 6 7 8 9; do
  for policy in original_v4 state_aware_v1 accuracy_first_v1 accuracy_first_terminal_v1; do
    "$VENV_PATH/bin/python" -m agent_support.recovery_policy_eval status \
      --root "$RECOVERY_ROOT" --policy "$policy" --batch "$batch"
    sbatch --wait --partition="$GPU_PARTITION" \
      --export=ALL,RECOVERY_POLICY="$policy",RECOVERY_BATCH="$batch" \
      --output="$RECOVERY_ROOT/logs/batch-$batch-$policy-%j.out" \
      scripts/recovery_policy_batch.sbatch
  done
done
mkdir -p "$RECOVERY_ROOT/reports"
for policy in original_v4 state_aware_v1 accuracy_first_v1 accuracy_first_terminal_v1; do
  "$VENV_PATH/bin/python" -m agent_support.recovery_policy_eval aggregate \
    --root "$RECOVERY_ROOT" --policy "$policy" \
    --output "$RECOVERY_ROOT/reports/$policy.json"
done
"$VENV_PATH/bin/python" -m agent_support.recovery_policy_eval compare \
  --root "$RECOVERY_ROOT" --output "$RECOVERY_ROOT/reports/paired_four_policy.json"
```

The primary score is environment-verified success rate. Compare terminal
review directly against unchanged `accuracy_first_v1`; report recovery
success, progress, interventions, strategy changes, support calls, inference
tokens, and elapsed time separately. Cost does not enter the primary score.
