# Opt-in GPT-OSS checkpoint-load diagnostics

This runtime-only probe surrounds `AutoModelForCausalLM.from_pretrained` in
the GPT-OSS worker. Set `AGENT_SUPPORT_GPTOSS_MODEL_LOAD_DIAGNOSTICS=1` before
starting the worker. It creates a separate
`diagnostics/gpt_oss_model_load_<UTC>_<pid>.jsonl` file. Samples are appended
every 15 seconds; a `progress_stall` event appears every 120 seconds without
an observed increase in the existing `Loading weights` progress bar. Start,
sample, progress, stall, error, and end records survive a later worker timeout
or Slurm preemption to the extent the filesystem has flushed appended data.

Each sample contains `/proc/self/io` counters (`rchar`, `syscr`, `read_bytes`,
and the other kernel counters), CPU user/system time, CPU percentage of one
core over the sample interval, RSS, page faults, PyTorch's process CUDA
allocated/reserved bytes, and `nvidia-smi` memory and utilization for both
worker-visible GPU ordinals. Raw physical GPU rows and
their UUIDs are retained to catch a visibility mapping error. The GPU
utilization and `memory.used` values are device-wide, so unrelated processes
on a shared GPU can contribute. If `nvidia-smi` or `/proc` is unavailable,
the sample records a measurement error and model loading continues.

The [pinned Transformers 5.22.2 loader](https://github.com/huggingface/transformers/blob/v5.22.2/src/transformers/core_model_loading.py#L1625-L1666)
iterates checkpoint mappings with a `tqdm(..., desc="Loading weights")`
progress bar; it does not expose a public
callback for the duration of each tensor's read, conversion, and placement.
The probe records each visible progress update's count, loader elapsed time,
and interval since the previous update. When an update advances one weight,
`single_weight_interval_seconds` records that interval. It is rounded by the
progress bar and may include work outside that weight. One update can cover
several weights, so these intervals are **not** exact per-tensor durations.
No loader functions or tensor operations are patched.

## Reading a stall

Compare consecutive `sample` rows around a `progress_stall` event:

| Observation | What it suggests |
| --- | --- |
| `read_bytes` or major faults rise while CPU and GPU use stay low | Storage reads or mmap page faults may be blocking checkpoint materialization. |
| `rchar` rises but `read_bytes` stays low | Read syscalls are active, often from page cache; low physical-read counts alone do not prove a stall. |
| CPU percentage stays high with little I/O or GPU change | CPU-side tensor preparation or conversion is a candidate. |
| GPU utilization or memory rises while CPU and I/O are quiet | CUDA placement, transfers, or initialization are candidates. |
| All counters stay flat and progress stops | Inspect worker state and the shared filesystem; the sample alone cannot identify the wait. |

These are clues, not exclusive phase measurements. Async CUDA work, cached
mmap reads, and other processes' GPU activity limit attribution. The probe
does not change the pinned checkpoint, sharding, attention, decoding, prompts,
policies, tasks, or evaluation limits.

## Offline checks

```bash
python3 -m pytest -q tests/test_gpt_oss_load_diagnostics.py
bash -n scripts/three_gpu_chunked_eager_load_diagnostic.sbatch
python3 -m agent_support.three_gpu_feasibility preflight
```

## Fresh bounded GPU run, only after authorization

Use a clean Locust checkout containing the implementation commit and the same
pinned environments and partition as the v2 protocol. Set a fresh root,
leaving the active startup retry and all earlier feasibility output untouched.

```bash
cd /home/evanpan/AgenticSDS-three-gpu-chunked-eager-v2
export THREE_GPU_ROOT=/shared_data0/evanpan/agent_support/three_gpu_chunked_eager_load_diagnostic_v1
export VENV_PATH=/home/evanpan/AgenticSDS/.venv
export AGENT_SUPPORT_GPTOSS_PYTHON=/shared_data0/evanpan/agent_support/venvs/gptoss20b/bin/python
export HF_HOME=/shared_data0/evanpan/agent_support/hf_cache
export AGENT_SUPPORT_GPTOSS_HF_HOME="$HF_HOME"
export AGENT_SUPPORT_M2_200_ARTIFACT_DIR=/shared_data0/evanpan/agent_support/gptoss_memory_v1/memory_scaling_v1/artifacts
export GPU_PARTITION="$(sacct -j 169480 -X -n -o Partition | awk 'NF {print $1; exit}')"
test -n "$GPU_PARTITION"
test ! -e "$THREE_GPU_ROOT"
"$VENV_PATH/bin/python" -m agent_support.three_gpu_feasibility preflight
mkdir -p "$THREE_GPU_ROOT/logs"
export THREE_GPU_PHASE=feasibility
sbatch --partition="$GPU_PARTITION" --export=ALL \
  --output="$THREE_GPU_ROOT/logs/feasibility-%j.out" scripts/three_gpu_chunked_eager_load_diagnostic.sbatch
```

If the archived 8,887-token request is available, set `THREE_GPU_MESSAGES_FILE`
before submission. The existing feasibility gate still controls whether any
smoke or pilot can proceed.
