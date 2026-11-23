# FlexAttention runtime integration for the three-policy evaluation

## Scope and ancestry

This branch starts at `accuracy-first-eval` commit `3e17abb`. Both diagnostic
branches share ancestor `890eb0a` with it. The opt-in attention branch ends at
`dd37f04`; the fatal-CUDA branch adds `1609f85`, a diagnostic replay commit,
and traceback commit `d6efedc`. This change ports the required runtime behavior
without merging the diagnostic branches or their probe/replay utilities.

The selected changes are: opt-in `attn_implementation="flex_attention"` with a
capability check; reporting the active model backend; typed CUDA OOM, illegal
memory access, device assert, and launch failures; worker traceback and failure
phase propagation; skipping CUDA cleanup/telemetry calls after a fatal error;
and terminal episode/stage handling. The policy implementations, registry,
frozen configurations, task manifest, model revisions, decoding settings,
dependency pins, action limits, and success criteria are unchanged. Ordinary
controller parsing fallbacks remain distinct from infrastructure failures.

The protocol is `textcraft_recovery_policy30_flex_runtime_v1`. Every run records
the requested and active attention implementation, policy and implementation
commit, model settings, manifest hash, and tracebacks flag. Model loading and
the evaluator both reject an active backend that differs from the requested
FlexAttention backend. Slurm stage and batch scripts enable the opt-in and
tracebacks for all three arms.

Offline verification on the Mac: `python -m pytest -q` passed all 253 tests;
`git diff --check` and `bash -n` for both Slurm scripts passed. CPU preflight
with both runtime flags set passed policy, parser, manifest, and memory checks.
The local environment lacks the pinned Locust GPT-OSS/Qwen runtimes, so its
strict runtime gate cannot pass there. The Locust gate below remains pending.

**This does not fix the intermittent FlexAttention illegal memory access seen
in earlier diagnostics.** A fatal occurrence invalidates that stage; diagnose
the kernel/runtime before repeating all matched arms in a new output root.

## Deployment and CPU-only Locust preflight

Run on the Mac after this branch is committed. These commands publish source
and create a clean detached Locust worktree; they do not submit GPU jobs. The
hostname below is the one in the prior protocol, but did not resolve from this
workspace. Set `LOCUST_HOST` to the working cluster SSH hostname if needed.

```bash
cd /private/tmp/AgenticSDS-accuracy-flex-runtime
test -z "$(git status --porcelain)"
export POLICY_COMMIT="$(git rev-parse HEAD)"
git push origin accuracy-first-eval-flex-runtime
export LOCUST_HOST=locust-login.seas.upenn.edu
ssh "$LOCUST_HOST" "cd /home/evanpan/AgenticSDS && git fetch origin accuracy-first-eval-flex-runtime && test ! -e /home/evanpan/AgenticSDS-accuracy-first-eval-flex-runtime && git worktree add --detach /home/evanpan/AgenticSDS-accuracy-first-eval-flex-runtime $POLICY_COMMIT"
ssh "$LOCUST_HOST"
```

Run on the Locust login node. The output root is new; the previous
`original_v4` smoke remains untouched.

```bash
export DEPLOY_REPO=/home/evanpan/AgenticSDS-accuracy-first-eval-flex-runtime
export RECOVERY_ROOT=/shared_data0/evanpan/agent_support/recovery_policy30_flex_runtime_v1
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
for policy in original_v4 state_aware_v1 accuracy_first_v1; do
  "$VENV_PATH/bin/python" -m agent_support.recovery_policy_eval preflight \
    --policy "$policy" --strict-runtime
done
```

If the accounting query no longer identifies an L40S partition, inspect
`sinfo -o '%P %G %m %l'` and set `GPU_PARTITION` to the verified partition.
The preflight is CPU-only. It checks the frozen artifacts, installed runtime,
policy identity, and opt-in settings; active GPU backend is checked at worker
initialization in each run.

## Fresh matched smoke and pilot (commands prepared, not submitted)

After successful CPU preflight, run these on Locust with the above exports.
Each job runs alone; the stage gate checks matched task indices, run commit,
backend, settings, and infrastructure validity. A task failure by itself does
not invalidate the stage.

```bash
mkdir -p "$RECOVERY_ROOT/logs"
for policy in original_v4 state_aware_v1 accuracy_first_v1; do
  sbatch --wait --partition="$GPU_PARTITION" \
    --export=ALL,RECOVERY_POLICY="$policy",RECOVERY_STAGE=smoke \
    --output="$RECOVERY_ROOT/logs/smoke-$policy-%j.out" \
    scripts/recovery_policy_stage.sbatch
done
"$VENV_PATH/bin/python" -m agent_support.recovery_policy_eval stage-report \
  --root "$RECOVERY_ROOT" --stage smoke
for policy in original_v4 state_aware_v1 accuracy_first_v1; do
  sbatch --wait --partition="$GPU_PARTITION" \
    --export=ALL,RECOVERY_POLICY="$policy",RECOVERY_STAGE=pilot \
    --output="$RECOVERY_ROOT/logs/pilot-$policy-%j.out" \
    scripts/recovery_policy_stage.sbatch
done
"$VENV_PATH/bin/python" -m agent_support.recovery_policy_eval stage-report \
  --root "$RECOVERY_ROOT" --stage pilot
```

Inspect each run's `run.json`, `episodes/`, `checkpoints/`, and `workers/` log
after smoke and before pilot. If a worker fails, stop the stage and use a new
root after diagnosis; never count the failure as a task outcome.
