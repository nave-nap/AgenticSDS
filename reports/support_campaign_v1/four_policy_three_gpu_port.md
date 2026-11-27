# Four-policy evaluation on the three-L40S runtime

Prepared 26 November 2026. No GPU job was submitted or cancelled. Job 171188
was user-reported pending when this work began. The full 8,887-input-token,
1,024-output-token feasibility gate has not passed in the evidence available
to this checkout.

## Source and prior results

The four frozen policies came from commit
`701c25d4a76903743148210b3f3e780f246d7131`, protocol
`textcraft_recovery_policy30_terminal_review_v1`. The registry, selector
classes, and telemetry module are byte-identical to that commit. The v4
controller `SYSTEM` prompt, strict decision parser, prompt assembly, adaptive
loop, and episode function have matching parsed syntax trees. The policy
configuration digests remain:

| Policy | SHA-256 |
| --- | --- |
| `original_v4` | `3cbd31d336be037890248d921527c043c829f6fafbd469b14447d8de0a35894d` |
| `state_aware_v1` | `b0f4a2a0bb4daf77b74a4464616107b5e3e69d8170c3602f16e38f8877bcb314` |
| `accuracy_first_v1` | `14a19273a3afb2b04b589610014d7981a9fb355d2071dbef964b1150c7ad4733` |
| `accuracy_first_terminal_v1` | `0fe2a50209cbaa8f32e2271264035431fd5eb5f892b638388162bae76134d3ae` |

The frozen 30-task manifest, task seeds, v4 configuration, M2-200 memory
digests, action and support budgets, model revisions, prompts, and evaluation
metrics are unchanged. This port writes a **new** protocol ID,
`textcraft_recovery_policy30_terminal_review_three_gpu_v1`, under a new root.
The old `recovery_v4_eval` two-arm CLI, plan, and output roots remain separate.

The available prior matched-policy record is an **invalid one-policy smoke**:
Slurm job 164919 ran `state_aware_v1` on seed 51 at the frozen commit and hit
a FlexAttention CUDA illegal-memory-access error on GPT-OSS request 10. The
archived report records 5,451 input tokens and a 1,024-token output ceiling
for that request, but no output-token count at failure. Slurm returned
`COMPLETED`/exit 0 after saving the fatal result. The prior root
`/shared_data0/evanpan/agent_support/recovery_policy30_terminal_review_v1`
is infrastructure-invalid and must not be reused. The available consolidated
record explicitly says **no valid matched four-policy pilot or 30-task
comparison exists**. Pilot outcomes therefore provide no policy ranking or
runtime estimate for this port.

## What changed

`recovery_policy_eval.py` now requests the existing
`gptoss_chunked_eager_v1` backend, the two-GPU 12/12 GPT-OSS placement, and
GPU memory diagnostics. The new Slurm launcher splits exactly three allocated
L40S IDs: Qwen sees GPU 0; GPT-OSS sees GPUs 1 and 2 as its CUDA 0 and 1.
It checks the active attention implementation, 12 packed expert modules on
each worker GPU, two L40S worker names, and Qwen's sole GPU-0 device map.
These are execution checks; they do not select a support or revise an action.

`matched-stage` runs the seed-51 smoke for all four policies in fixed registry
order, sharing one Qwen load, one GPT-OSS load, and one frozen memory object.
GPT-OSS is reseeded by task ID before each episode, as in the original
evaluator. Qwen uses greedy decoding. A pilot allocation accepts at most two
policies over the frozen first three tasks; two serial allocations complete
the matched pilot. `matched-batch` similarly shares model loads for up to two
three-task policy batches after the pilot gate. A six-hour allocation may
hold fewer tasks if smoke timing requires it. Incomplete or failed attempts
remain in their own immutable run/attempt directories.

The new feasibility check requires a completed long request and compares
the validated feasibility commit with the policy checkout for model runtime
files, configs, and manifests. Because controller error handling changed for
fatal CUDA classification, **run the full feasibility gate on the port commit**
before submitting the policy smoke. The pending five-minute startup check at
171188 remains useful as launcher validation, but does not replace that full
gate. A prior feasibility result from a different runtime source is rejected.

The evaluator stops a run on worker fatality, backend error, support backend
error, model generation failure, environment error, or malformed accounting.
It records each episode's decision telemetry, policy intervention, support
request/result, action, progress assessment, fallback, tokens, latency,
strategy changes, and replanning. Stage and full comparisons verify exact
indices, frozen digests, runtime metadata, episode snapshots, completion,
and absence of infrastructure-invalid episodes. The allocation file records
the Slurm job ID and reused model load; `sacct -X` plus
`campaign_offline allocation-cost` reports allocated GPU-hours, CPU-hours,
and dollars if site rates are supplied. A successful Slurm exit is not a
scientific validity signal.

## Runtime difference and comparability

The original frozen policies were prepared for single-GPU FlexAttention.
This port uses GPT-OSS packed MXFP4 weights on two L40S cards with explicit
12/12 layer placement and 256-query chunked eager attention. Each query
still attends to the same full key range with the pinned eager masking and
attention sink, but the different kernel and matrix operation order can
produce small numerical differences. GPT-OSS uses sampling, so small logit
changes can change support decisions and later trajectories. Inter-GPU
transfers, model-load time, memory headroom, and latency differ. The new
backend also classifies fatal CUDA errors explicitly. **Compare the four
policies within this new runtime only**; do not pool or treat the invalid
FlexAttention smoke as a matched observation. The development-set success
comparison remains exploratory and does not establish held-out generality.

## Offline validation

```bash
.venv/bin/python -m pytest -q \
  tests/test_recovery_policy_three_gpu_port.py \
  tests/test_recovery_policy_eval.py tests/test_recovery_terminal_review.py \
  tests/test_recovery_accuracy_policy.py tests/test_recovery_v4.py \
  tests/test_recovery_v4_eval.py
.venv/bin/python -m pytest -q \
  tests/test_controller_backend_oom.py tests/test_three_gpu_runtime.py \
  tests/test_three_gpu_feasibility_startup.py
bash -n scripts/recovery_policy_three_gpu_stage.sbatch
git diff --check
```

Results in this checkout: **72 passed** for policy/recovery tests, **27
passed** for worker/three-GPU tests, shell syntax passed, and `git diff
--check` passed. The full offline suite passed **295 tests**. The parity tests check byte-identical policy modules,
controller prompt/parser/decision syntax, all four policy digests, frozen
config and manifest digests, fixed smoke/pilot indices, shared model loads,
and rejection of incomplete full feasibility evidence. These CPU checks do
not prove GPU generation or scientific task success.

## Safe submission procedure, prepared only

First commit and deploy the port to a clean Locust worktree. Use the pinned
Qwen and GPT-OSS environments, same caches and M2-200 files as the feasibility
campaign. Verify 171188's `startup_valid` result. Submit the **full**
six-hour feasibility job on this port commit in a fresh root using the command
in `three_gpu_feasibility_startup_v1/protocol.md`. Inspect its JSON, worker
log, memory JSONL, and Slurm accounting. The following smoke commands are
conditional on `infrastructure_valid`; they do not submit anything by
themselves:

```bash
cd /home/evanpan/AgenticSDS-four-policy-three-gpu
export VENV_PATH=/home/evanpan/AgenticSDS/.venv
export AGENT_SUPPORT_GPTOSS_PYTHON=/shared_data0/evanpan/agent_support/venvs/gptoss20b/bin/python
export AGENT_SUPPORT_GPTOSS_HF_HOME=/shared_data0/evanpan/agent_support/hf_cache
export AGENT_SUPPORT_M2_200_ARTIFACT_DIR=/shared_data0/evanpan/agent_support/gptoss_memory_v1/memory_scaling_v1/artifacts
export GPU_PARTITION="$(sacct -j 170042 -X -n -o Partition | awk 'NF {print $1; exit}')"
export POLICY_FEASIBILITY_RESULT="$THREE_GPU_ROOT/feasibility_8887/feasibility_result.json"
test -n "$GPU_PARTITION"
test -z "$(git status --porcelain)"
"$VENV_PATH/bin/python" -m agent_support.recovery_policy_eval feasibility-gate \
  --result "$POLICY_FEASIBILITY_RESULT"
export POLICY_ROOT="/shared_data0/evanpan/agent_support/recovery_policy30_three_gpu_v1/run_$(date -u +%Y%m%dT%H%M%SZ)"
test ! -e "$POLICY_ROOT"
mkdir -m 700 -p "$POLICY_ROOT/logs"
export POLICY_STAGE=smoke
export POLICY_LIST="original_v4 state_aware_v1 accuracy_first_v1 accuracy_first_terminal_v1"
sbatch --partition="$GPU_PARTITION" --gpus=3 --cpus-per-task=8 --mem=96G \
  --time=06:00:00 --no-requeue --export=ALL \
  --output="$POLICY_ROOT/logs/smoke-%j.out" scripts/recovery_policy_three_gpu_stage.sbatch
```

This is **one** matched smoke allocation, with four seed-51 episodes and one
model load. Resource request: **exactly three L40S, eight CPUs, 96 GiB host
memory, six-hour wall limit**. The frozen 40-minute per-task limit bounds
episode time at 2 h 40 min total, before initialization and setup. If model
loading or task timing exceeds the wall limit, the attempt is incomplete and
cannot pass the stage gate. Do not queue a second GPU allocation while this
one is pending or running.

After the job ends, inspect `allocations/*/allocation.json`, all four
`run.json` files, episodes, checkpoints, worker log, CUDA JSONL, and `sacct`.
Then run the CPU stage gate:

```bash
"$VENV_PATH/bin/python" -m agent_support.recovery_policy_eval stage-report \
  --root "$POLICY_ROOT" --stage smoke
```

Only after that report is `ready: true`, use two **serial** pilot allocations
with `POLICY_STAGE=pilot` and `POLICY_LIST` set first to
`original_v4 state_aware_v1`, then to
`accuracy_first_v1 accuracy_first_terminal_v1`; use the same `sbatch`
resource request and a distinct Slurm log for each. Each allocation runs six
episodes and shares one model load. Check smoke timing first: a 45-minute
model load plus six 40-minute episode maxima is 4 h 45 min, but earlier
diagnostics also showed long Qwen initialization, so six hours is not
guaranteed. If timing is insufficient, run one policy per allocation and
keep the matched task set and fresh attempt roots. Run `stage-report --stage
pilot` only after all four valid pilot arms exist. Then `init` writes the
locked full-run plan; `matched-batch` is available for at most two policies
per allocation. No pilot or 30-task job is submitted by this report.
