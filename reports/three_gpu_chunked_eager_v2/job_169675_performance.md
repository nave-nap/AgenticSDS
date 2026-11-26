# Job 169675 performance diagnosis and bounded retry

The copied CUDA JSONL, worker log, and Slurm stdout are preserved under
`debug_logs/job_169675/`. The job used the pre-performance-instrumentation
runtime. It has no model-load JSONL or final feasibility result because Slurm
ended it at the 2:30 wall limit. Its CUDA JSONL contains no OOM.

| Phase | UTC interval or duration | Evidence |
| --- | --- | --- |
| GPT-OSS load | 04:22:07–05:07:25, 45m 18s | `model_load_start` to `model_loaded`; 411-weight progress bar took 40m 37s. |
| After GPT-OSS load | 05:07:25–06:01:30, 54m 05s | Qwen loading and exact-request preparation occurred here; their individual durations were not recorded. |
| 70-token warmup | 06:01:30–06:18:37, 17m 07s | Prefill crossed all 24 layers in 90.69s; 31 one-token decode steps followed. |
| 8,887-token prefill | 06:18:38–06:20:03, 84.70s | All 24 layers completed, including layers 12–23 on worker `cuda:1`. |
| Long-request decode | 06:20:03–06:27:36, at least 7m 33s | 22 layer-0 decode starts; final event was `attention_chunk_start` on layer 8, query length 1, key length 128. |

The 48 `layer_complete` events are two full prefills: 24 at query length 70
and 24 at 8,887. Each prefill has 12 entries on each worker GPU. They are
not 48 decode layers; the hook intentionally omits one-token steps. The
final long-request event is in decoding after the full long prefill. The
log's 1,329 attention starts versus 1,328 ends reflects Slurm stopping during
that final layer-8 operation, not a recorded CUDA error.

For 1,256 completed one-token attention calls, the median interval from
`attention_chunk_start` to `attention_chunk_end` is 0.308s and their intervals
sum to about 666s. The 70-token warmup had a median 22.6s between successive
layer-0 decode starts; the long request had a median 18.1s. This is far too
much time to attribute to query-block looping: one-token decoding uses one
block per layer, and the 70-token prefill also uses one block per layer.

The old chunk logger took a CUDA memory snapshot and appended a JSONL row at
both edges of **every decode layer**. A full decode token thus made 48
snapshots and shared-filesystem writes. The job recorded 2,513 one-token
start/end rows before termination. Similar per-call intervals for short and
long keys point to this diagnostic path as a strong cause of decode latency;
the logs do not isolate filesystem write time from CUDA synchronization or
kernel time. The runtime-only correction suppresses one-token chunk logging
while retaining prefill and generation-level memory telemetry. It does not
alter eager attention or the 256-query prefill blocks.

At the end of the observed long request, sampled allocations were at most
7.20 GiB on worker `cuda:0` and 7.01 GiB on `cuda:1`; minimum observed global
free memory was about 35.61 and 35.80 GiB. These are observed samples, not
final peaks, because no `generation_end` was written.

The loading slowdown remains unresolved. No `/proc/self/io`, CPU, RSS, or GPU
utilization samples exist for this job. First-use kernel compilation may
contribute to the first 70-token prefill (layer 0 took about 46s versus 4.7s
on the long prefill), but the logs cannot distinguish it from CPU or storage
stalls. The 45-minute GPT-OSS load and 54-minute subsequent gap require the
opt-in load diagnostics from commit `4e07be3` on a fresh run. MXFP4 expert
kernel and inter-GPU transfer time also cannot be separated from the current
layer and attention timestamps. The layer-11-to-layer-12 timestamp gap is
only an upper-bound clue because it includes hooks and other dispatch work.
Even `attention_chunk_start` to `attention_chunk_end` includes telemetry
overhead and asynchronous CUDA dispatch; it is not pure attention-kernel time.

## Performance probe before another full feasibility

The new probe loads the same pinned GPT-OSS worker, then runs the original
`Reply briefly.` warmup twice with an 8-token ceiling, followed by the same
8,887-token diagnostic request with a 2-token ceiling. It does not load Qwen
or advance benchmark tasks. First warmup must finish within 600s, repeat
within 120s, and long prefill plus its first decode step within 900s. Each
request needs at least two output tokens, all 24 layers on the expected GPUs,
and at least 10 GiB sampled global free memory on each worker GPU. The probe
shortens only its own RPC deadlines; frozen evaluation timeouts stay intact.
It writes `performance_result.json` after each stage, so partial results
survive a wall-time failure. `performance_valid` permits consideration of a
new full feasibility run; it is not the full feasibility gate.

Use a clean Locust checkout containing the implementation commit, the same
pinned environments and L40S partition, and a **fresh** root. This command is
for an authorized future GPU run; no job is submitted by this report.

```bash
cd /home/evanpan/AgenticSDS-three-gpu-chunked-eager-v2
export THREE_GPU_ROOT=/shared_data0/evanpan/agent_support/three_gpu_chunked_eager_perf_probe_v1
export VENV_PATH=/home/evanpan/AgenticSDS/.venv
export AGENT_SUPPORT_GPTOSS_PYTHON=/shared_data0/evanpan/agent_support/venvs/gptoss20b/bin/python
export HF_HOME=/shared_data0/evanpan/agent_support/hf_cache
export AGENT_SUPPORT_GPTOSS_HF_HOME="$HF_HOME"
export AGENT_SUPPORT_M2_200_ARTIFACT_DIR=/shared_data0/evanpan/agent_support/gptoss_memory_v1/memory_scaling_v1/artifacts
export GPU_PARTITION="$(sacct -j 169675 -X -n -o Partition | awk 'NF {print $1; exit}')"
test -n "$GPU_PARTITION"
test ! -e "$THREE_GPU_ROOT"
"$VENV_PATH/bin/python" -m agent_support.three_gpu_feasibility preflight
mkdir -p "$THREE_GPU_ROOT/logs"
sbatch --partition="$GPU_PARTITION" --export=ALL \
  --output="$THREE_GPU_ROOT/logs/performance-%j.out" scripts/three_gpu_performance_probe.sbatch
```

If the exact archived 8,887-token messages are available, set
`THREE_GPU_MESSAGES_FILE` before submission. Review the model-load JSONL,
`performance_result.json`, and GPU memory JSONL. If the second warmup remains
slow after decode logging is removed, investigate MXFP4 kernel compilation
and transfer timing with a separate short CUDA profiler run before changing
attention or allocator settings.
