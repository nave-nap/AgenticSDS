# Job 171188 startup-probe repair

27 November 2026. This repair is on the four-policy three-L40S runtime branch,
whose policy port begins at commit `7f99c9a`. No GPU job was submitted or
cancelled here.

## Evidence and diagnosis

The supplied job 171188 artifact identifies commit `fa2c57e`, three correctly
isolated L40S GPUs, `worker_process_check`, and `subprocess.TimeoutExpired`
after the old 60-second bound. The whole job lasted 76 seconds. The subprocess
combined Python startup, `torch` import, CUDA enumeration, and worker-module
import in one command. It produced no stage timing before timeout. The actual
GPT-OSS worker was never launched, so the missing worker log is expected.

Static import inspection found no model loading at
`agent_support.gpt_oss_controller_worker` import time. Its top-level imports
are Python standard-library modules plus the local GPU-diagnostics and
two-GPU helpers; `GptOss20BModel` is imported only on an `init` request. A
local CPU import probe completed successfully. These observations point to
the combined Python/torch/CUDA startup path as the leading cause of the
60-second timeout, but the job 171188 artifact cannot distinguish torch
import from CUDA enumeration or filesystem latency. The new probe records
those phases separately before another GPU attempt.

## Repair and bounds

The startup check now runs two separate subprocesses in the pinned GPT-OSS
Python environment with the same two allocated GPU IDs:

1. `cuda_identity` records Python entry, completed torch import, and completed
   CUDA enumeration, then requires exactly two L40S names. Its total bound is
   **120 seconds**. The longer bound applies to torch/CUDA initialization,
   which was previously combined with worker import and timed out at 60
   seconds. The result records each phase's measured elapsed time.
2. `worker_import` runs only after valid CUDA identity. It records Python
   entry and completion of the lightweight worker-module import. Its bound
   remains **30 seconds**. No model is loaded and no TextCraft task advances.

Each subprocess writes `*_events.jsonl` immediately after a phase completes
and `*_result.json` on exit or timeout. The parent result includes captured
stdout/stderr tails, elapsed time, timeout value, and the last recorded
phase. A missing `python_started` event identifies a stall before the child
entered its module. A timeout leaves `status=failed` in
`feasibility_result.json`; the `startup-check` CLI exits nonzero. Malformed
GPU counts, names, JSON, or worker-import acknowledgment cannot pass.

`startup_valid` remains a launcher check only. The existing full gate still
requires `status=infrastructure_valid` from a completed request with exactly
8,887 input tokens, at least one output token under the frozen 1,024-token
ceiling, all 24 layers on the designated GPUs, memory headroom, and no OOM.
The four-policy evaluator also compares model-runtime source against the full
feasibility commit. Therefore the full feasibility job must use the **new
repair commit**, and a startup result cannot authorize a matched smoke.

No frozen policy, task assignment, support budget, prompt, model revision,
sharding map, attention implementation, or evaluation definition changed.
The launcher still requests exactly three L40S GPUs, eight CPUs, and 96 GiB
host memory.

## Offline validation

```bash
.venv/bin/python -m pytest -q tests/test_three_gpu_feasibility_startup.py \
  tests/test_recovery_policy_three_gpu_port.py
bash -n scripts/three_gpu_chunked_eager_startup.sbatch \
  scripts/three_gpu_chunked_eager.sbatch scripts/three_gpu_mxfp4.sbatch
git diff --check
```

Tests cover identity and worker-import timeouts, malformed GPU evidence,
failure-artifact persistence, nonzero startup exit, and rejection of a
startup-only result by the full gate. The local worker-module import completes
without loading either model. GPU execution remains untested for this repair.

## Bounded Locust retry, prepared only

After the repair commit is published, deploy it to a **new clean worktree**.
Use the pinned environments and cache from the prior three-L40S campaign. If
the partition query is unavailable, select the verified L40S partition with
`sinfo`. The command below is a fresh five-minute startup probe, not the full
generation test:

```bash
cd /home/evanpan/AgenticSDS-three-gpu-startup-repair
test -z "$(git status --porcelain)"
export VENV_PATH=/home/evanpan/AgenticSDS/.venv
export AGENT_SUPPORT_GPTOSS_PYTHON=/shared_data0/evanpan/agent_support/venvs/gptoss20b/bin/python
export HF_HOME=/shared_data0/evanpan/agent_support/hf_cache
export AGENT_SUPPORT_GPTOSS_HF_HOME="$HF_HOME"
export AGENT_SUPPORT_M2_200_ARTIFACT_DIR=/shared_data0/evanpan/agent_support/gptoss_memory_v1/memory_scaling_v1/artifacts
export GPU_PARTITION="$(sacct -j 171188 -X -n -o Partition | awk 'NF {print $1; exit}')"
test -n "$GPU_PARTITION"
export THREE_GPU_ROOT="/shared_data0/evanpan/agent_support/three_gpu_startup_repair/check_$(date -u +%Y%m%dT%H%M%SZ)"
test ! -e "$THREE_GPU_ROOT"
mkdir -m 700 -p "$THREE_GPU_ROOT/logs"
export THREE_GPU_PHASE=startup_check
sbatch --partition="$GPU_PARTITION" --gpus=3 --cpus-per-task=8 --mem=96G \
  --time=00:05:00 --no-requeue --export=ALL \
  --output="$THREE_GPU_ROOT/logs/startup-%j.out" scripts/three_gpu_chunked_eager_startup.sbatch
```

Inspect `startup_check/feasibility_result.json`, both probe result JSON files,
both event JSONL files, and the Slurm log. Require `startup_valid`, two L40S
names, `worker_module_imported: true`, and plausible phase durations. A
failure or wall timeout stops the sequence; keep its root immutable.

Only then submit **one** full six-hour feasibility attempt from the same clean
repair commit under another fresh root. Do not make this an automatic `afterok`
dependency because Slurm completion alone does not validate the startup
evidence:

```bash
export THREE_GPU_ROOT="/shared_data0/evanpan/agent_support/three_gpu_startup_repair/full_$(date -u +%Y%m%dT%H%M%SZ)"
test ! -e "$THREE_GPU_ROOT"
mkdir -m 700 -p "$THREE_GPU_ROOT/logs"
export THREE_GPU_PHASE=feasibility
sbatch --partition="$GPU_PARTITION" --gpus=3 --cpus-per-task=8 --mem=96G \
  --time=06:00:00 --no-requeue --export=ALL \
  --output="$THREE_GPU_ROOT/logs/feasibility-%j.out" scripts/three_gpu_chunked_eager_startup.sbatch
```

If the exact archived 8,887-token request is available, set
`THREE_GPU_MESSAGES_FILE` before submission. Otherwise the driver labels its
synthetic exact-length request. After completion, inspect the result, worker
log, CUDA JSONL, and `sacct`, then run:

```bash
"$VENV_PATH/bin/python" -m agent_support.three_gpu_feasibility gate \
  --result "$THREE_GPU_ROOT/feasibility_8887/feasibility_result.json"
```

Only a passed full gate can precede the four-policy matched smoke. No smoke,
pilot, or full policy evaluation is submitted by this procedure.
