# Two-GPU eager evaluation inside Slurm

This runtime variant starts from eager commit `01598f4`. Main-process Qwen
uses allocation GPU 0. The GPT-OSS child sees only allocation GPU 1, mapped
to its local CUDA device 0. GPT-OSS stays eager. The evaluator verifies both
model placements before starting a task and records them in `run.json`.
CUDA memory events identify each process's visible GPU. No policy, frozen
task, memory corpus, revision, decoding, limit, or scoring change.

Protocol: `textcraft_recovery_policy30_eager_two_gpu_v1`.
New root: `/shared_data0/evanpan/agent_support/recovery_policy30_eager_two_gpu_v1`.
Prior roots must remain untouched.

## Deploy

After the commit is available in the Locust clone through an authorized
transfer, create a separate worktree from the Mac terminal:

```bash
export LOCUST_HOST=locust-login.seas.upenn.edu
export TWO_GPU_COMMIT="$(cd /private/tmp/AgenticSDS-accuracy-eager-two-gpu && git rev-parse HEAD)"
ssh "$LOCUST_HOST" "cd /home/evanpan/AgenticSDS && git fetch origin accuracy-first-eval-eager-two-gpu && git cat-file -e '$TWO_GPU_COMMIT^{commit}' && test ! -e /home/evanpan/AgenticSDS-accuracy-eager-two-gpu && git worktree add --detach /home/evanpan/AgenticSDS-accuracy-eager-two-gpu $TWO_GPU_COMMIT"
```

## CPU-only preflight in existing interactive job 165211

These commands do not submit jobs or load GPU models. Keep the main shell's
CUDA visibility unchanged. If its GPU list differs, stop and inspect the
allocation mapping.

```bash
set -euo pipefail
test "$SLURM_JOB_ID" = 165211
test "$(hostname -s)" = node9
test "$CUDA_VISIBLE_DEVICES" = 0,1
export DEPLOY_REPO=/home/evanpan/AgenticSDS-accuracy-eager-two-gpu
export RECOVERY_ROOT=/shared_data0/evanpan/agent_support/recovery_policy30_eager_two_gpu_v1
export AGENT_SUPPORT_GPTOSS_PYTHON=/shared_data0/evanpan/agent_support/venvs/gptoss20b/bin/python
export AGENT_SUPPORT_GPTOSS_HF_HOME=/shared_data0/evanpan/agent_support/hf_cache
export AGENT_SUPPORT_M2_200_ARTIFACT_DIR=/shared_data0/evanpan/agent_support/gptoss_memory_v1/memory_scaling_v1/artifacts
export HF_HOME="$AGENT_SUPPORT_GPTOSS_HF_HOME"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=0
export AGENT_SUPPORT_GPTOSS_FLEX_ATTENTION=0
export AGENT_SUPPORT_GPTOSS_CUDA_CACHE_RELEASE=0
export AGENT_SUPPORT_GPTOSS_WORKER_TRACEBACKS=1
export AGENT_SUPPORT_GPU_MEMORY_DIAGNOSTICS=1
export CUDA_LAUNCH_BLOCKING=1
export AGENT_SUPPORT_QWEN_GPU=0
export AGENT_SUPPORT_GPTOSS_WORKER_GPU=1
test -d "$DEPLOY_REPO"
test ! -e "$RECOVERY_ROOT"
test -x "$AGENT_SUPPORT_GPTOSS_PYTHON"
test -r "$AGENT_SUPPORT_M2_200_ARTIFACT_DIR/corpus_200.json"
test -r "$AGENT_SUPPORT_M2_200_ARTIFACT_DIR/index_200.json"
cd "$DEPLOY_REPO"
test -z "$(git status --porcelain)"
source /home/evanpan/AgenticSDS/.venv/bin/activate
for policy in original_v4 state_aware_v1 accuracy_first_v1 accuracy_first_terminal_v1; do
  python -m agent_support.recovery_policy_eval preflight --policy "$policy" --strict-runtime
done
```

## First real-model episode

Run only after authorization to use the reserved GPUs. This executes the
frozen TextCraft seed-51 terminal-policy smoke directly through `python`.
With `set -euo pipefail`, placement, CUDA, and OOM failures stop the sequence.
The gate verifies the saved episode and both GPU-specific memory logs.

```bash
mkdir -p "$RECOVERY_ROOT/logs"
python -m agent_support.recovery_policy_eval smoke \
  --root "$RECOVERY_ROOT" --policy accuracy_first_terminal_v1 \
  2>&1 | tee "$RECOVERY_ROOT/logs/terminal-smoke.log"
python -m agent_support.recovery_policy_eval runtime-gate \
  --root "$RECOVERY_ROOT" | tee "$RECOVERY_ROOT/logs/runtime-gate.json"
```

Inspect `run.json`, the episode's `terminal_reviews`, the worker log,
and both memory JSONL files. The gate reports whether terminal review was
actually triggered; do not claim real-model verification of that path if it
was not. A task failure is a valid runtime result. An OOM, placement error,
or other infrastructure failure invalidates the stage; preserve the root,
report the traceback and memory evidence, and do not retry in that worker.

## Matched smoke, pilot, and 30 tasks

After a valid gate, run the other smoke arms, the three-task pilot, then
the matched 30-task evaluation. Each command runs inside an allocation with
the same two-GPU environment; no `sbatch` is used. Stop at the first
nonzero status. A failed infrastructure stage requires a new root.

```bash
for policy in original_v4 state_aware_v1 accuracy_first_v1; do
  python -m agent_support.recovery_policy_eval smoke --root "$RECOVERY_ROOT" --policy "$policy"
  python -m agent_support.recovery_policy_eval stage-report --root "$RECOVERY_ROOT" --stage smoke --policy "$policy"
done
python -m agent_support.recovery_policy_eval stage-report --root "$RECOVERY_ROOT" --stage smoke
for policy in original_v4 state_aware_v1 accuracy_first_v1 accuracy_first_terminal_v1; do
  python -m agent_support.recovery_policy_eval pilot --root "$RECOVERY_ROOT" --policy "$policy"
  python -m agent_support.recovery_policy_eval stage-report --root "$RECOVERY_ROOT" --stage pilot --policy "$policy"
done
python -m agent_support.recovery_policy_eval stage-report --root "$RECOVERY_ROOT" --stage pilot
python -m agent_support.recovery_policy_eval init --root "$RECOVERY_ROOT"
for batch in 0 1 2 3 4 5 6 7 8 9; do
  for policy in original_v4 state_aware_v1 accuracy_first_v1 accuracy_first_terminal_v1; do
    python -m agent_support.recovery_policy_eval run-batch --root "$RECOVERY_ROOT" --policy "$policy" --batch "$batch"
    python -m agent_support.recovery_policy_eval status --root "$RECOVERY_ROOT" --policy "$policy" --batch "$batch" --require-complete
  done
done
mkdir -p "$RECOVERY_ROOT/reports"
for policy in original_v4 state_aware_v1 accuracy_first_v1 accuracy_first_terminal_v1; do
  python -m agent_support.recovery_policy_eval aggregate --root "$RECOVERY_ROOT" --policy "$policy" --output "$RECOVERY_ROOT/reports/$policy.json"
done
python -m agent_support.recovery_policy_eval compare --root "$RECOVERY_ROOT" --output "$RECOVERY_ROOT/reports/paired_four_policy.json"
```

The primary metric remains environment-verified success. Report recovery,
progress, interventions, strategy changes, support calls, tokens, and time
separately.
