# Two-GPU eager evaluation inside Slurm

This v2 correction follows two-GPU commit `93ec1f0`, which started from
eager commit `01598f4`. Main-process Qwen
uses allocation GPU 0. The GPT-OSS child sees only allocation GPU 1, mapped
to its local CUDA device 0. GPT-OSS stays eager. The evaluator verifies both
model placements before starting a task and records them in `run.json`.
CUDA memory events identify each process's visible GPU. No policy, frozen
task, memory corpus, revision, decoding, limit, or scoring change.

Protocol: `textcraft_recovery_policy30_eager_two_gpu_v2`.
New root: `/shared_data0/evanpan/agent_support/recovery_policy30_eager_two_gpu_v2`.
Prior roots must remain untouched.

The GPT-OSS loader requests local `device_map={"": 0}`, then verifies all
packed expert storage and ordinary parameters on local `cuda:0`. Its
`hf_device_map` can nevertheless be empty. The validator accepts an empty
map while still requiring worker visibility `1`, one CUDA device, model
device `cuda:0`, and eager attention. A populated map must point only to
local device 0. GPU UUIDs are cross-checked when exposed by the CUDA
runtimes.

## Deploy

After the commit is available in the Locust clone through an authorized
transfer, create a separate worktree from the Mac terminal:

```bash
export LOCUST_HOST=locust-login.seas.upenn.edu
export TWO_GPU_COMMIT="$(cd /private/tmp/AgenticSDS-accuracy-eager-two-gpu && git rev-parse HEAD)"
ssh "$LOCUST_HOST" "cd /home/evanpan/AgenticSDS && git fetch origin accuracy-first-eval-eager-two-gpu && git cat-file -e '$TWO_GPU_COMMIT^{commit}' && test ! -e /home/evanpan/AgenticSDS-accuracy-eager-two-gpu-v2 && git worktree add --detach /home/evanpan/AgenticSDS-accuracy-eager-two-gpu-v2 $TWO_GPU_COMMIT"
```

## Create a new tmux-managed two-GPU allocation

On the Locust login node, recover the prior two-L40S partition and start
tmux. This is a command for the operator, not a job submitted by Codex.

```bash
export GPU_PARTITION="$(sacct -j 165211 -X -n -o Partition | awk 'NF {print $1; exit}')"
test -n "$GPU_PARTITION"
tmux new-session -s accuracy-eager-two-gpu
```

Inside the new tmux login shell, request the compute shell:

```bash
export GPU_PARTITION="$(sacct -j 165211 -X -n -o Partition | awk 'NF {print $1; exit}')"
test -n "$GPU_PARTITION"
srun --partition="$GPU_PARTITION" --nodes=1 --ntasks=1 --gpus=2 \
  --cpus-per-task=6 --mem=64G --time=04:00:00 --pty bash -l
```

The compute-node Bash shell must remain the
parent shell; never `source` an evaluator script or enable `set -e` in it.
Detach from tmux with Ctrl-B, D. When done, exit the compute shell
deliberately to release the allocation.

## CPU-only preflight in the new allocation

These commands do not submit jobs or load GPU models. Keep the compute
shell's CUDA visibility unchanged. Exports are made in the persistent shell,
but all fallible checks run in a child Bash process. A failed child returns
to the interactive shell without releasing the allocation.

```bash
set +e
export DEPLOY_REPO=/home/evanpan/AgenticSDS-accuracy-eager-two-gpu-v2
export RECOVERY_ROOT=/shared_data0/evanpan/agent_support/recovery_policy30_eager_two_gpu_v2
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
export EVAL_PYTHON=/home/evanpan/AgenticSDS/.venv/bin/python
if bash -e -u -o pipefail <<'PREFLIGHT'
test -n "$SLURM_JOB_ID"
test "$CUDA_VISIBLE_DEVICES" = 0,1
test -d "$DEPLOY_REPO"
test ! -e "$RECOVERY_ROOT"
test -x "$EVAL_PYTHON"
test -x "$AGENT_SUPPORT_GPTOSS_PYTHON"
test -r "$AGENT_SUPPORT_M2_200_ARTIFACT_DIR/corpus_200.json"
test -r "$AGENT_SUPPORT_M2_200_ARTIFACT_DIR/index_200.json"
cd "$DEPLOY_REPO"
test -z "$(git status --porcelain)"
for policy in original_v4 state_aware_v1 accuracy_first_v1 accuracy_first_terminal_v1; do
  "$EVAL_PYTHON" -m agent_support.recovery_policy_eval preflight --policy "$policy" --strict-runtime
done
PREFLIGHT
then
  echo "CPU preflight passed; allocation shell is still active"
else
  echo "CPU preflight failed; stop and inspect, allocation shell is still active"
fi
```

## First real-model episode

Run only after authorization to use the reserved GPUs. This executes the
frozen TextCraft seed-51 terminal-policy smoke directly through Python.
The child Bash process stops on placement, CUDA, or OOM failure; its exit
cannot kill the parent allocation shell. The gate verifies the saved
episode and both GPU-specific memory logs.

```bash
if bash -e -u -o pipefail <<'SMOKE'
cd "$DEPLOY_REPO"
mkdir -p "$RECOVERY_ROOT/logs"
"$EVAL_PYTHON" -m agent_support.recovery_policy_eval smoke \
  --root "$RECOVERY_ROOT" --policy accuracy_first_terminal_v1 \
  2>&1 | tee "$RECOVERY_ROOT/logs/terminal-smoke.log"
"$EVAL_PYTHON" -m agent_support.recovery_policy_eval runtime-gate \
  --root "$RECOVERY_ROOT" | tee "$RECOVERY_ROOT/logs/runtime-gate.json"
SMOKE
then
  echo "Smoke gate passed; allocation shell is still active"
else
  echo "Smoke failed; preserve this root and stop, allocation shell is still active"
fi
```

Inspect `run.json`, the episode's `terminal_reviews`, the worker log,
and both memory JSONL files. The gate reports whether terminal review was
actually triggered; do not claim real-model verification of that path if it
was not. A task failure is a valid runtime result. An OOM, placement error,
or other infrastructure failure invalidates the stage; preserve the root,
report the traceback and memory evidence, and do not retry in that worker.

## Matched smoke, pilot, and 30 tasks

After a valid gate, run the other smoke arms, the three-task pilot, then
the matched 30-task evaluation. These direct Python commands run in a child
Bash process with fail-fast behavior. A failure returns to the allocation
shell and requires review; an infrastructure-invalid stage needs a new root.
No `sbatch` is used.

```bash
if bash -e -u -o pipefail <<'EVALUATION'
cd "$DEPLOY_REPO"
for policy in original_v4 state_aware_v1 accuracy_first_v1; do
  "$EVAL_PYTHON" -m agent_support.recovery_policy_eval smoke --root "$RECOVERY_ROOT" --policy "$policy"
  "$EVAL_PYTHON" -m agent_support.recovery_policy_eval stage-report --root "$RECOVERY_ROOT" --stage smoke --policy "$policy"
done
"$EVAL_PYTHON" -m agent_support.recovery_policy_eval stage-report --root "$RECOVERY_ROOT" --stage smoke
for policy in original_v4 state_aware_v1 accuracy_first_v1 accuracy_first_terminal_v1; do
  "$EVAL_PYTHON" -m agent_support.recovery_policy_eval pilot --root "$RECOVERY_ROOT" --policy "$policy"
  "$EVAL_PYTHON" -m agent_support.recovery_policy_eval stage-report --root "$RECOVERY_ROOT" --stage pilot --policy "$policy"
done
"$EVAL_PYTHON" -m agent_support.recovery_policy_eval stage-report --root "$RECOVERY_ROOT" --stage pilot
"$EVAL_PYTHON" -m agent_support.recovery_policy_eval init --root "$RECOVERY_ROOT"
for batch in 0 1 2 3 4 5 6 7 8 9; do
  for policy in original_v4 state_aware_v1 accuracy_first_v1 accuracy_first_terminal_v1; do
    "$EVAL_PYTHON" -m agent_support.recovery_policy_eval run-batch --root "$RECOVERY_ROOT" --policy "$policy" --batch "$batch"
    "$EVAL_PYTHON" -m agent_support.recovery_policy_eval status --root "$RECOVERY_ROOT" --policy "$policy" --batch "$batch" --require-complete
  done
done
mkdir -p "$RECOVERY_ROOT/reports"
for policy in original_v4 state_aware_v1 accuracy_first_v1 accuracy_first_terminal_v1; do
  "$EVAL_PYTHON" -m agent_support.recovery_policy_eval aggregate --root "$RECOVERY_ROOT" --policy "$policy" --output "$RECOVERY_ROOT/reports/$policy.json"
done
"$EVAL_PYTHON" -m agent_support.recovery_policy_eval compare --root "$RECOVERY_ROOT" --output "$RECOVERY_ROOT/reports/paired_four_policy.json"
EVALUATION
then
  echo "Matched evaluation complete; allocation shell is still active"
else
  echo "Evaluation stopped; inspect artifacts, allocation shell is still active"
fi
```

The primary metric remains environment-verified success. Report recovery,
progress, interventions, strategy changes, support calls, tokens, and time
separately.
