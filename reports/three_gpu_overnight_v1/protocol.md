# Three-GPU overnight diagnostic campaign v1

This protocol runs one read-only CPU cache probe and, only after that succeeds,
one six-hour three-L40S allocation. GPT-OSS is loaded once on physical GPUs 1
and 2; Qwen loads once on physical GPU 0. The GPT-OSS worker keeps its pinned
MXFP4 weights, explicit 12/12 layer map, 256-query chunked eager attention,
prompts, sampling parameters, and tokenizer. The driver sends only diagnostic
requests. It does not run benchmark tasks or change any evaluation settings.

## Gates and limits

| Stage | Prompt | Output ceiling | Soft latency gate | Hard RPC limit |
| --- | --- | ---: | ---: | ---: |
| Pinned checkpoint cache | Up to 512 MiB of safetensors, read only | — | At least 128 MiB and 8 MiB/s | 15 min Slurm wall |
| GPT-OSS initialization | Pinned checkpoint, one load | — | — | 5,400 s startup timeout |
| Qwen initialization | Pinned checkpoint, one load | — | — | 5,400 s alarm |
| First short generation | `Reply briefly.` | 8 | 600 s | 1,200 s |
| Repeat short generation | Same prompt | 8 | 120 s | 300 s |
| Medium prefill | Exact 2,048-token synthetic note | 2 | 360 s | 900 s |
| Long prefill | Existing exact 8,887-token diagnostic request | 2 | 900 s | 1,200 s |

After each successful generation, the driver requires at least two generated
tokens, all 24 prefill layers executing on their designated worker GPU, chunked
attention records on both devices, and 10 GiB sampled global free memory on
each GPT-OSS GPU. A completed but slow stage runs one short, two-token CUDA
profiler trace (prompt at most 128 tokens) and then stops. A timed-out or
invalid stage stops immediately; it never advances to medium or long context.
The full 1,024-token feasibility request is intentionally excluded until this
campaign passes. The profiler is diagnostic and never used as a latency gate.

`gpt_oss_model_load_*.jsonl` records process reads, CPU, RSS, GPU utilization,
weight progress, and stalls. `gpt_oss_inference_*.jsonl` samples the same
signals every 15 seconds during each request. Existing CUDA JSONL records
per-device memory peaks, free-memory minima, prefill layers, and attention
timing. `campaign_result.json` and `summary.md` are refreshed after every
stage. The one-page summary includes load durations, stage latency, GPU
memory, utilization, and failure. These files are per attempt; Slurm job ID
and restart count create a new directory even if a site forces requeue.
`--no-requeue` requests no normal restart. A preempted attempt stays available
for read-only review; a forced restart begins a new attempt and loads models
again. No stage is resumed from a partial result.

The cache job reads the checkpoint on a CPU node, so its throughput is a
screening signal, not proof of GPU-node filesystem performance. `read_bytes`
in `/proc/self/io` reflects storage reads; `rchar` also counts cache-served
reads. Sustained high `read_bytes` with low throughput suggests storage
contention. Low CPU use and long weight-progress gaps with little read
activity suggest blocking or cache metadata stalls. High CPU use with little
storage I/O and low GPU utilization suggests CPU tensor preparation. High GPU
utilization or increasing device memory with low CPU and I/O suggests GPU
transfer or initialization. First-generation slowness that disappears on the
repeat suggests one-time kernel setup. If repeat latency stays high, inspect
the optional profiler's CUDA kernels and CPU gaps before changing attention.
These signals are observational and do not by themselves isolate a kernel.

## Offline verification

```bash
.venv/bin/python -m pytest -q tests/test_three_gpu_overnight.py tests/test_three_gpu_performance_probe.py tests/test_gpt_oss_load_diagnostics.py tests/test_controller_backend_oom.py
bash -n scripts/three_gpu_mxfp4.sbatch scripts/three_gpu_overnight_cache.sbatch scripts/three_gpu_overnight_gpu.sbatch
```

## Locust deployment and submission (run only after authorization)

First publish the implementation commit to the existing branch from the
local checkout. This command is for the user to run after review; the
implementation does not push or submit anything.

```bash
git push origin HEAD:refs/heads/autonomous-recovery-v4-cache-release
```

Then on Locust, fetch the commit and use a clean detached
worktree. Replace `IMPLEMENTATION_COMMIT` with the commit hash reported with
this implementation.

```bash
git fetch origin autonomous-recovery-v4-cache-release
git worktree add --detach /home/evanpan/AgenticSDS-three-gpu-overnight IMPLEMENTATION_COMMIT
cd /home/evanpan/AgenticSDS-three-gpu-overnight
test "$(git rev-parse HEAD)" = "IMPLEMENTATION_COMMIT"
test -z "$(git status --porcelain)"

export VENV_PATH=/home/evanpan/AgenticSDS/.venv
export AGENT_SUPPORT_GPTOSS_PYTHON=/shared_data0/evanpan/agent_support/venvs/gptoss20b/bin/python
export HF_HOME=/shared_data0/evanpan/agent_support/hf_cache
export AGENT_SUPPORT_GPTOSS_HF_HOME="$HF_HOME"
export AGENT_SUPPORT_M2_200_ARTIFACT_DIR=/shared_data0/evanpan/agent_support/gptoss_memory_v1/memory_scaling_v1/artifacts
export THREE_GPU_ROOT="/shared_data0/evanpan/agent_support/three_gpu_overnight_v1/run_$(date -u +%Y%m%dT%H%M%SZ)"
export GPU_PARTITION="$(sacct -j 169675 -X -n -o Partition | awk 'NF {print $1; exit}')"
export CPU_PARTITION="$(sinfo -h -o '%P %G' | awk '$2=="(null)" {gsub(/\*/, "", $1); print $1; exit}')"
test -n "$GPU_PARTITION" && test -n "$CPU_PARTITION"
test ! -e "$THREE_GPU_ROOT"
scontrol show partition "$GPU_PARTITION" | tr ' ' '\n' | grep '^MaxTime='
mkdir -m 700 -p "$THREE_GPU_ROOT"

cpu_submission="$(sbatch --parsable --partition="$CPU_PARTITION" --export=ALL scripts/three_gpu_overnight_cache.sbatch)"
cpu_job="${cpu_submission%%;*}"
export THREE_GPU_CACHE_JOB_ID="$cpu_job"
gpu_submission="$(sbatch --parsable --partition="$GPU_PARTITION" --dependency="afterok:$cpu_job" --kill-on-invalid-dep=yes --export=ALL scripts/three_gpu_overnight_gpu.sbatch)"
gpu_job="${gpu_submission%%;*}"
printf 'CPU cache job: %s\nGPU diagnostic job: %s\nRoot: %s\n' "$cpu_job" "$gpu_job" "$THREE_GPU_ROOT"
```

Check that `MaxTime` permits six hours before running `sbatch`. If the local
CPU partition does not advertise `(null)` GRES, set `CPU_PARTITION` to the
site's CPU-only partition. Optionally set `THREE_GPU_MESSAGES_FILE` to an
archived exact 8,887-token JSON request before submission; otherwise the
driver constructs and records an exact-length synthetic note. The GPU job
is held by `afterok` and cancelled on invalid dependency if the cache job
fails. Neither job is submitted by this implementation.

Afterward, inspect `cpu/job_<id>_restart_<n>/cache/summary.md` and
`gpu/job_<id>_restart_<n>/run/summary.md`. If Slurm stops during a stage,
regenerate the latest summary from partial JSONL using:

```bash
"$VENV_PATH/bin/python" -m agent_support.three_gpu_overnight summary \
  --output-dir "$THREE_GPU_ROOT/gpu/job_${gpu_job}_restart_0/run"
```

Only an observed `performance_valid` result warrants a separate, newly
authorized full 8,887-token feasibility job. This protocol never launches
the 30-task benchmark.
