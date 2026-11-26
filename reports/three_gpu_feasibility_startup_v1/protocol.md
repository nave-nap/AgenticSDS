# Three-GPU feasibility startup diagnosis

## What job 170811 establishes

The 12-second exit happened before any recorded model-loading or generation.
The successful overnight job 170042 already demonstrated the pinned 12/12
MXFP4 placement, Qwen isolation, chunked eager prefill, and 8,887-token
two-token probe. The full feasibility path differs in its fixed
`$THREE_GPU_ROOT/feasibility_8887` output directory and its 32-token warmup
and 1,024-token long generation. The output ceilings cannot explain a failure
before model loading.

The strongest code-level explanation for **no traceback and no result** is
the launcher's former `test ! -e "$THREE_GPU_ROOT/feasibility_8887"`: Bash
silently returned status 1 after `nvidia-smi` if that path already existed.
An empty worker log at that path does not prove this job launched the worker;
its modification time must be compared with job 170811. This is a hypothesis,
not a confirmed reconstruction, because the original root directory listing
and timestamps are unavailable locally.

Other confirmed reporting gaps were `preflight()` and `require_clean_commit()`
outside the feasibility `try`, a result write that could replace the original
exception if it failed, and a `Popen` before worker-constructor cleanup. These
could also produce missing or misleading artifacts. Their occurrence in job
170811 is unproven. The overnight job passed the same preflight and clean
commit checks, which makes a persistent model or configuration error less
likely. Its separate attempt directory avoids the fixed-path collision.

Read-only forensic check on Locust, using the old root from job 170811:

```bash
export FAILED_ROOT=/absolute/path/to/job_170811_root
stat -c '%y %s %n' "$FAILED_ROOT/feasibility_8887" \
  "$FAILED_ROOT/feasibility_8887/workers/gpt_oss_controller_worker.log"
find "$FAILED_ROOT/feasibility_8887" -maxdepth 2 -type f -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' | sort
```

If the directory or worker log predates job 170811, the silent guard explains
the exit. If they were created during job 170811, inspect the new stage and
traceback diagnostics on a fresh run before attributing the failure.

## Implementation and artifacts

The launcher now prints an explicit collision error and logs the selected
parent and worker GPU IDs before starting Python. Feasibility emits
`startup_events.jsonl` and refreshes `feasibility_result.json` at preflight,
clean checkout, GPU validation, worker init, Qwen init, request preparation,
warmup, long generation, and final validation. Exceptions include the full
traceback, current stage, worker exit code when available, and up to 8 KiB of
worker log. If the atomic result write fails, an exclusive
`feasibility_result_fallback.json` and Slurm stderr retain the error. A
stdlib-only entry wrapper records import failures before the feasibility
module can run. Worker construction preserves the original `Popen` or init
exception if cleanup fails.

The `startup-check` command follows the same Slurm launcher chain, checks
preflight and clean checkout, validates the parent's L40S identity, and
starts the pinned GPT-OSS Python in a separate process with the two worker
GPU IDs. It imports the worker and verifies both L40S identities without
loading models. Its result must be `startup_valid`; it is not a feasibility
result and cannot pass the smoke/pilot gate.

## Offline verification

```bash
.venv/bin/python -m pytest -q tests
bash -n scripts/three_gpu_chunked_eager_startup.sbatch scripts/three_gpu_chunked_eager.sbatch scripts/three_gpu_mxfp4.sbatch
git diff --check
```

## Fresh-root Locust deployment and validation

After review, publish the implementation commit to the existing branch.
No push or Slurm submission is performed by this implementation.

```bash
git push origin HEAD:refs/heads/autonomous-recovery-v4-cache-release
```

On Locust, deploy the pushed branch tip in a clean worktree and set the same
pinned environment paths used by job 170042:

```bash
git fetch origin autonomous-recovery-v4-cache-release
git worktree add --detach /home/evanpan/AgenticSDS-feasibility-startup origin/autonomous-recovery-v4-cache-release
cd /home/evanpan/AgenticSDS-feasibility-startup
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/autonomous-recovery-v4-cache-release)"
test -z "$(git status --porcelain)"

export VENV_PATH=/home/evanpan/AgenticSDS/.venv
export AGENT_SUPPORT_GPTOSS_PYTHON=/shared_data0/evanpan/agent_support/venvs/gptoss20b/bin/python
export HF_HOME=/shared_data0/evanpan/agent_support/hf_cache
export AGENT_SUPPORT_GPTOSS_HF_HOME="$HF_HOME"
export AGENT_SUPPORT_M2_200_ARTIFACT_DIR=/shared_data0/evanpan/agent_support/gptoss_memory_v1/memory_scaling_v1/artifacts
export GPU_PARTITION="$(sacct -j 170042 -X -n -o Partition | awk 'NF {print $1; exit}')"
test -n "$GPU_PARTITION"
"$VENV_PATH/bin/python" -m agent_support.three_gpu_feasibility preflight
```

Submit the short startup gate to a **new** root. The same scripts request
three GPUs and validate that the parent sees one L40S while the worker
subprocess sees the other two:

```bash
export THREE_GPU_ROOT="/shared_data0/evanpan/agent_support/three_gpu_feasibility_startup_v1/check_$(date -u +%Y%m%dT%H%M%SZ)"
test ! -e "$THREE_GPU_ROOT"
mkdir -p "$THREE_GPU_ROOT/logs"
export THREE_GPU_PHASE=startup_check
sbatch --partition="$GPU_PARTITION" --gpus=3 --time=00:05:00 --export=ALL \
  --output="$THREE_GPU_ROOT/logs/startup-%j.out" scripts/three_gpu_chunked_eager_startup.sbatch
```

Proceed only if the startup job completes and
`$THREE_GPU_ROOT/startup_check/feasibility_result.json` has
`status=startup_valid`, `worker_process_check.count=2`, two L40S names, and
the expected `CUDA_VISIBLE_DEVICES` split. Any failed stage or missing result
stops the sequence; inspect the Slurm log, `startup_events.jsonl`, traceback,
and worker-process exit code.

Only then submit the unchanged full feasibility request to another **new**
root. The six-hour Slurm limit overrides the older 2:30 header; the pinned
model's generation RPC deadline and the 1,024-token output ceiling remain
unchanged. Set `THREE_GPU_MESSAGES_FILE` beforehand if the archived exact
8,887-token request is available.

```bash
export THREE_GPU_ROOT="/shared_data0/evanpan/agent_support/three_gpu_feasibility_startup_v1/full_$(date -u +%Y%m%dT%H%M%SZ)"
test ! -e "$THREE_GPU_ROOT"
mkdir -p "$THREE_GPU_ROOT/logs"
export THREE_GPU_PHASE=feasibility
sbatch --partition="$GPU_PARTITION" --gpus=3 --time=06:00:00 --export=ALL \
  --output="$THREE_GPU_ROOT/logs/feasibility-%j.out" scripts/three_gpu_chunked_eager_startup.sbatch
```

Success requires `status=infrastructure_valid`, exactly 8,887 input tokens,
at least one completed output token from the 1,024-token ceiling, all 24
prefill layers crossing the 12/12 boundary, 12 packed expert modules per
worker GPU, Qwen wholly on its GPU, at least 10 GiB sampled free memory per
GPT-OSS GPU, and no OOM. Only then run the existing read-only gate command:

```bash
"$VENV_PATH/bin/python" -m agent_support.three_gpu_feasibility gate \
  --result "$THREE_GPU_ROOT/feasibility_8887/feasibility_result.json"
```

Any failure or missing artifact ends validation. Do not launch smoke, pilot,
or the 30-task benchmark from this procedure.
