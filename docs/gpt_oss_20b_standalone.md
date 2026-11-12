# GPT-OSS-20B standalone TextCraft development comparison

This branch adds `openai/gpt-oss-20b` through the existing model registry. The runner still invokes `make_model(config)`; the GPT-OSS module is loaded by `component_modules`. The two never-support configs use the ordered 20-task development manifest, seed 42, history window 8, and limits of 30 actions, 30 decisions, and 30 model calls. Both disable all supports and support budgets. Qwen retains greedy generation with 100 new tokens; GPT-OSS uses its pinned Harmony template, medium reasoning, sampling, and up to 1024 new tokens. Compare actual input/output/reasoning tokens and inference time; the two models do not have equal token costs.

The GPT-OSS adapter records final-channel text as the only benchmark-visible output. It records the full generated completion, exact output token count, analysis-body token count, final-body token count, status for missing/malformed/truncated final messages, and CUDA peak allocated/reserved bytes. Missing final output becomes a parse error; no fallback action is invented. The pinned tokenizer template dynamically inserts today's date, so this adapter replaces that expression with the configured `prompt_date` and records a hash of the resulting template. This does not make sampling bitwise reproducible across preemption/restarts.

## Deployment assessment

[Hugging Face's MXFP4 documentation](https://huggingface.co/docs/transformers/v5.14.0/en/quantization/mxfp4) requires Accelerate, `kernels`, Triton 3.4+, and NVIDIA compute capability at least 7.5. [NVIDIA lists L40S as compute capability 8.9](https://developer.nvidia.com/cuda/gpus), and the [GPT-OSS Triton kernel card lists L40S](https://huggingface.co/kernels/kernels-community/gpt-oss-triton-kernels). The adapter uses Transformers MXFP4, not an unquantized PyTorch reference loader. It rejects a load if MXFP4 is absent, expert parameters are not packed `uint8`, or any parameter is off GPU. This is a documented compatibility path, but actual Locust load/generation and peak memory still require the smoke test. An older [OpenAI Transformers guide](https://developers.openai.com/cookbook/articles/gpt-oss/run-transformers) mentions Hopper-only MXFP4 support; the newer Hugging Face requirements and kernel card include L40S.

The published 20B MXFP4 weight footprint is about 16 GB, and [NVIDIA lists 48 GB for L40S](https://www.nvidia.com/en-in/data-center/l40s/). KV cache, temporary kernels, attention workspace, and fragmentation add to that footprint. The 1024-token limit is per call; prompt length grows with the episode. The adapter records actual peak allocated and reserved GPU memory. Treat 48 GB as a hard gate at smoke time, not as proof that Qwen and GPT-OSS can later coexist. Separate environments use different Transformers versions; same-process co-loading is untested. Same-GPU co-loading would also need a measured joint-memory test.

The [model revision](https://huggingface.co/api/models/openai/gpt-oss-20b) is `6cee5e81ee83917806bbde320786a8fb61efebee`. [Harmony's format guide](https://github.com/openai/harmony/blob/main/docs/format.md) documents `analysis` and `final` channels and the final `<|return|>` stop token. The pinned model's tokenizer template supplies the supported `reasoning_effort` argument. The proposed environment uses Python 3.12, [PyTorch 2.8.0 CUDA 12.8](https://pytorch.org/get-started/previous-versions/), [Transformers 5.22.2](https://pypi.org/project/transformers/5.22.2/), Accelerate 1.15.0, and `kernels` 0.17.1. Runtime versions, CUDA version, GPU identity, model and tokenizer revision, quantization, prompt date/hash, generation settings, framework commit, manifest/config hashes, and seed are stored in each run's `run.json`. Keep the generated `pip freeze` file alongside the run for a full environment record.

Plan for at least 35 GB of free persistent storage for the model cache, isolated environment, kernel cache, and download/compilation headroom. Check the actual home quota and shared storage before download; the cluster's home quota was previously reported as 50 GB. Point `AGENT_SUPPORT_DATA` to a writable persistent location with enough free space. Reuse the existing `HF_HOME` cache to avoid duplicate weights; do not clear or rewrite Qwen cache entries. The commands below do not download anything until the explicit preparation step.

## Push and create a separate Locust checkout

From the local isolated worktree after committing:

```bash
cd /private/tmp/AgenticSDS-gptoss20b
git status --short
git push -u origin gptoss20b_standalone
```

On Locust, after login:

```bash
cd "$HOME/AgenticSDS-framework"
git fetch origin gptoss20b_standalone
git worktree add "$HOME/AgenticSDS-gptoss20b" origin/gptoss20b_standalone
cd "$HOME/AgenticSDS-gptoss20b"
test -z "$(git status --porcelain)"
export AGENT_SUPPORT_DATA="$HOME/agent_support_data"
export HF_HOME="$AGENT_SUPPORT_DATA/hf_cache"
quota -s 2>/dev/null || true
df -h "$AGENT_SUPPORT_DATA" "$HF_HOME"
sinfo -N -h -p standby -o '%N %G %f' | grep -i l40s
scontrol show partition standby | grep -E 'MaxTime|AllowAccounts|AllowQos'
```

Use the account, partition, QoS, and typed GPU GRES actually shown for your allocation if they differ from `hassani`, `standby`, and `gpu:l40s:1` below.

## Prepare and validate the isolated environment

These commands leave the existing Qwen `.venv` untouched. They also preserve the prior Experiment 1 results and the held-out manifests.

```bash
cd "$HOME/AgenticSDS-gptoss20b"
export VENV_GPTOSS="$AGENT_SUPPORT_DATA/venvs/gptoss20b"
mkdir -p "$(dirname "$VENV_GPTOSS")" "$HF_HOME"
python3.12 -m venv "$VENV_GPTOSS"
"$VENV_GPTOSS/bin/python" -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 'torch==2.8.0'
"$VENV_GPTOSS/bin/python" -m pip install --no-cache-dir -r requirements-gptoss20b.txt
"$VENV_GPTOSS/bin/python" -m pip check
"$VENV_GPTOSS/bin/python" -m pytest -q
"$VENV_GPTOSS/bin/python" -m pip freeze --all > "$AGENT_SUPPORT_DATA/gptoss20b_pip_freeze.txt"
sha256sum "$AGENT_SUPPORT_DATA/gptoss20b_pip_freeze.txt"
```

Validate configs and all manifest partitions without loading weights:

```bash
"$VENV_GPTOSS/bin/python" - <<'PY'
from agent_support.common import read_json, validate_config, validate_manifest
from agent_support.components import load_components
from agent_support.run_store import audit_partitions
for path in ('configs/standalone_textcraft_dev_qwen_never.json',
             'configs/standalone_textcraft_dev_gptoss20b_never.json'):
    config = read_json(path)
    manifest = read_json(config['manifest'])
    load_components(config)
    validate_manifest(manifest)
    validate_config(config, manifest['benchmark'])
    audit_partitions(config, manifest)
    assert manifest['partition'] == 'development' and len(manifest['tasks']) == 20
    print(path, config['model_revision'])
PY
```

Only after the storage check, populate the pinned GPT-OSS snapshot in the shared cache. This downloads model weights and tokenizer files, and the first inference may also fetch Triton kernels. If compute nodes have outbound Hub access, the runner can populate this cache during the GPU smoke instead; explicit preparation makes failures easier to diagnose.

```bash
"$VENV_GPTOSS/bin/python" - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('openai/gpt-oss-20b', revision='6cee5e81ee83917806bbde320786a8fb61efebee')
PY
```

## Three-task GPU smoke

The script uses the first three development tasks when `AGENT_SUPPORT_LIMIT=3`. Set the wall time after checking `MaxTime`. Each job has one L40S and a separate output root. Submit Qwen first, inspect it, then submit GPT-OSS. This document does not submit either job.

```bash
cd "$HOME/AgenticSDS-gptoss20b"
export AGENT_SUPPORT_DATA="$HOME/agent_support_data"
export HF_HOME="$AGENT_SUPPORT_DATA/hf_cache"
export WALLTIME='04:00:00'
export AGENT_SUPPORT_LIMIT=3
export AGENT_SUPPORT_CONFIG="$PWD/configs/standalone_textcraft_dev_qwen_never.json"
export VENV_PATH="$HOME/AgenticSDS/.venv"
export AGENT_SUPPORT_OUTPUT_ROOT="$AGENT_SUPPORT_DATA/standalone_models/smoke_qwen"
mkdir -p "$AGENT_SUPPORT_OUTPUT_ROOT" "$AGENT_SUPPORT_DATA/standalone_models/logs"
QWEN_SMOKE_JOB="$(sbatch --parsable --export=ALL --account=hassani --partition=standby --qos=standby --gres=gpu:l40s:1 --cpus-per-task=4 --mem=32G --time="$WALLTIME" --output="$AGENT_SUPPORT_DATA/standalone_models/logs/%j.out" --open-mode=append scripts/locust_modular_job.sh)"
QWEN_SMOKE_JOB="${QWEN_SMOKE_JOB%%;*}"
squeue -j "$QWEN_SMOKE_JOB"
```

After Qwen completes, inspect `$AGENT_SUPPORT_DATA/standalone_models/smoke_qwen/textcraft_slurm_${QWEN_SMOKE_JOB}/run.json` and `summary.json`, then submit GPT-OSS:

```bash
export AGENT_SUPPORT_CONFIG="$PWD/configs/standalone_textcraft_dev_gptoss20b_never.json"
export VENV_PATH="$VENV_GPTOSS"
export AGENT_SUPPORT_OUTPUT_ROOT="$AGENT_SUPPORT_DATA/standalone_models/smoke_gptoss20b"
mkdir -p "$AGENT_SUPPORT_OUTPUT_ROOT"
GPT_SMOKE_JOB="$(sbatch --parsable --export=ALL --account=hassani --partition=standby --qos=standby --gres=gpu:l40s:1 --cpus-per-task=4 --mem=32G --time="$WALLTIME" --output="$AGENT_SUPPORT_DATA/standalone_models/logs/%j.out" --open-mode=append scripts/locust_modular_job.sh)"
GPT_SMOKE_JOB="${GPT_SMOKE_JOB%%;*}"
squeue -j "$GPT_SMOKE_JOB"
```

Inspect the completed smoke, including valid actions, generation statuses, token accounting, and peak memory. `run.json` must show `status=complete`, `git_dirty=false`, the pinned revision and `model_metadata.quantization=mxfp4_packed`; `summary.json` must show three tasks. This command prints a concise audit:

```bash
"$VENV_GPTOSS/bin/python" - "$AGENT_SUPPORT_DATA/standalone_models/smoke_gptoss20b/textcraft_slurm_${GPT_SMOKE_JOB}" <<'PY'
import json, pathlib, sys
root=pathlib.Path(sys.argv[1]); meta=json.loads((root/'run.json').read_text()); summary=json.loads((root/'summary.json').read_text())
assert meta['status']=='complete' and not meta['git_dirty'] and summary['tasks']==3
print('revision',meta['model_metadata']['model_revision'],'quantization',meta['model_metadata']['quantization'])
print('actions',summary['environment_actions'],'parser_errors',summary['parser_errors'],'generation_failures',summary['generation_failures'])
print('input/output/reasoning tokens',summary['input_tokens'],summary['output_tokens'],summary['reasoning_tokens'])
print('peak allocated/reserved GiB',round(summary['peak_gpu_memory_allocated_bytes']/2**30,2),round(summary['peak_gpu_memory_reserved_bytes']/2**30,2))
for path in sorted((root/'episodes').glob('*.json')):
    ep=json.loads(path.read_text())
    print(ep['task_id'],ep['status'],[(s['generation_status'],s['action'],s.get('error')) for s in ep['trace']])
PY
```

A complete run with all outputs malformed or no environment actions is **not** an acceptable smoke result. Diagnose template/kernel behavior before full evaluation. The per-step `raw_completion` may include reasoning; keep traces in the owner-only results directory.

## Full 20-task development runs and report

Proceed after the GPT-OSS smoke passes. Remove the limit and use fresh full-run roots. Submit Qwen and GPT-OSS sequentially; the dependency prevents both from occupying GPUs at once. If the Qwen job fails, the GPT-OSS job will not start.

```bash
cd "$HOME/AgenticSDS-gptoss20b"
unset AGENT_SUPPORT_LIMIT
export AGENT_SUPPORT_CONFIG="$PWD/configs/standalone_textcraft_dev_qwen_never.json"
export VENV_PATH="$HOME/AgenticSDS/.venv"
export AGENT_SUPPORT_OUTPUT_ROOT="$AGENT_SUPPORT_DATA/standalone_models/full_qwen"
mkdir -p "$AGENT_SUPPORT_OUTPUT_ROOT"
QWEN_FULL_JOB="$(sbatch --parsable --export=ALL --account=hassani --partition=standby --qos=standby --gres=gpu:l40s:1 --cpus-per-task=4 --mem=32G --time="$WALLTIME" --output="$AGENT_SUPPORT_DATA/standalone_models/logs/%j.out" --open-mode=append scripts/locust_modular_job.sh)"
QWEN_FULL_JOB="${QWEN_FULL_JOB%%;*}"
export AGENT_SUPPORT_CONFIG="$PWD/configs/standalone_textcraft_dev_gptoss20b_never.json"
export VENV_PATH="$VENV_GPTOSS"
export AGENT_SUPPORT_OUTPUT_ROOT="$AGENT_SUPPORT_DATA/standalone_models/full_gptoss20b"
mkdir -p "$AGENT_SUPPORT_OUTPUT_ROOT"
GPT_FULL_JOB="$(sbatch --parsable --export=ALL --dependency="afterok:$QWEN_FULL_JOB" --account=hassani --partition=standby --qos=standby --gres=gpu:l40s:1 --cpus-per-task=4 --mem=32G --time="$WALLTIME" --output="$AGENT_SUPPORT_DATA/standalone_models/logs/%j.out" --open-mode=append scripts/locust_modular_job.sh)"
GPT_FULL_JOB="${GPT_FULL_JOB%%;*}"
printf 'QWEN_FULL_JOB=%s\nGPT_FULL_JOB=%s\n' "$QWEN_FULL_JOB" "$GPT_FULL_JOB" > "$AGENT_SUPPORT_DATA/standalone_models/full_job_ids.env"
chmod 600 "$AGENT_SUPPORT_DATA/standalone_models/full_job_ids.env"
```

After both finish, report task success, paired outcomes, actions, failed actions, model calls, all token categories, parser/generation failures, inference time, GPU peak memory where available, and exact run provenance. Qwen's unchanged adapter does not measure peak CUDA memory or reasoning tokens; its peak is reported as unavailable and reasoning tokens as zero/not applicable. The report checks complete clean runs, exact task order, shared interaction settings, no supports, and the same framework commit. The JSON includes per-task metrics and full provenance; Markdown is a short summary.

```bash
source "$AGENT_SUPPORT_DATA/standalone_models/full_job_ids.env"
"$VENV_GPTOSS/bin/python" -m agent_support.standalone_model_report \
  --qwen "$AGENT_SUPPORT_DATA/standalone_models/full_qwen/textcraft_slurm_${QWEN_FULL_JOB}" \
  --gpt-oss "$AGENT_SUPPORT_DATA/standalone_models/full_gptoss20b/textcraft_slurm_${GPT_FULL_JOB}" \
  --expected-tasks 20 \
  --output "$AGENT_SUPPORT_DATA/standalone_models/report_20"
cat "$AGENT_SUPPORT_DATA/standalone_models/report_20/report.md"
```

A standalone success advantage would justify studying GPT-OSS as a candidate support, but it would not establish a favorable hybrid accuracy/compute tradeoff. That requires a later routing experiment.
