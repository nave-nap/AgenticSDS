# Three-GPU chunked eager startup retry

Job 169480 ended before generation. The copied worker log under
`reports/three_gpu_chunked_eager_v2/debug_logs/job_169480/` shows the first
weight-loading progress update appeared at 9 minutes 24 seconds, then loading
reached `19/411` at 10 minutes 30 seconds. It records no later progress before
the controller's 30-minute timeout. The Slurm stdout contains the same failed feasibility result and
identifies three allocated L40S devices. The feasibility result has
`request_source: null`; the separate CUDA telemetry reportedly contains only
`model_load_start`. No OOM or attention execution is established by this run.

The controller's pinned `timeout_seconds: 1800` was used for both the
initialization RPC and every generation RPC. It terminated the worker while
model loading was still incomplete.

The earlier 12/12 sharded run loaded all 411 weights in 41 seconds and took
90.6 seconds from `model_load_start` to `model_loaded`. An archived single-GPU
worker log loaded 411 weights in about 1.7 seconds. The new slowdown is in
`AutoModelForCausalLM.from_pretrained`, after tokenizer and config loading.
The full log shows stalls, not a steady 411-weight load at a lower rate. It
cannot distinguish shared-cache I/O from per-weight initialization, GPU
transfer, or a transient filesystem stall. No cache contents,
checkpoint revision, sharding, or model configuration is changed here.

The startup-only retry sets `AGENT_SUPPORT_GPTOSS_STARTUP_TIMEOUT_SECONDS=5400`
in `scripts/three_gpu_chunked_eager_startup.sbatch`. The frozen controller
configuration still limits each generation RPC to 1800 seconds. The worker
reports which deadline expired, and the feasibility result records worker
startup elapsed time even when initialization fails. The Slurm job remains
bounded at 2 hours 30 minutes. The longer timeout only prevents the known
30-minute startup termination; it does not resolve an underlying stall.
Run the read-only cache probe and, if possible, sample the loading worker's
CPU usage and `/proc/<pid>/io` on the compute node before using the retry.
If read bytes advance slowly, investigate the shared cache. If reads stop
while CPU or GPU work continues, investigate model initialization or transfer.

## Read-only cache probe on Locust

Run this outside a GPU job with the pinned `HF_HOME`. It checks the exact
snapshot, broken symlinks, and sample read time for each checkpoint file.
Warm OS cache can make this faster than a cold model load, so a fast probe
does not rule out intermittent shared-filesystem delay.

```bash
export HF_HOME=/shared_data0/evanpan/agent_support/hf_cache
python3 - <<'PY'
from pathlib import Path
from time import monotonic

revision = "6cee5e81ee83917806bbde320786a8fb61efebee"
snapshot = Path("/shared_data0/evanpan/agent_support/hf_cache/hub/models--openai--gpt-oss-20b/snapshots") / revision
if not snapshot.is_dir():
    raise SystemExit(f"Missing pinned snapshot: {snapshot}")
weights = sorted(snapshot.glob("*.safetensors"))
if not weights:
    raise SystemExit(f"No safetensors files in {snapshot}")
print("snapshot", snapshot, "weight_files", len(weights))
for path in weights:
    target = path.resolve(strict=True)
    start = monotonic()
    with path.open("rb") as stream:
        sample = stream.read(64 * 1024 * 1024)
    seconds = monotonic() - start
    print(path.name, "target", target, "size_bytes", target.stat().st_size,
          "sample_bytes", len(sample), "sample_seconds", round(seconds, 3))
PY
```

## Fresh bounded GPU feasibility, pending authorization

Publish the implementation commit to a clean Locust checkout. Set the same
pinned environment paths as the v2 protocol and use a new output root; do not
overwrite job 169480. The v2 feasibility gate still requires the 8,887-token
generation and correct two-GPU forward execution.

```bash
cd /home/evanpan/AgenticSDS-three-gpu-chunked-eager-v2
export THREE_GPU_ROOT=/shared_data0/evanpan/agent_support/three_gpu_chunked_eager_v2_startup_retry
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
  --output="$THREE_GPU_ROOT/logs/feasibility-%j.out" scripts/three_gpu_chunked_eager_startup.sbatch
```

For an archived exact request, set `THREE_GPU_MESSAGES_FILE` before `sbatch`.
The job must report `status: infrastructure_valid` before smoke or pilot.
