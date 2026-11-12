# Locust recovery after GPT-OSS MXFP4 smoke job 146021

The existing GPT-OSS environment has `kernels==0.17.1`; Transformers 5.22.2 requires 0.17.2 or newer within the 0.17 series. The corrected adapter checks the pinned dependency before loading weights and verifies the custom Triton FP4 tensors after loading. These commands preserve the existing weights, original dependency freeze, job 146021, and the Qwen reference at commit `3d9db8c`. They do not submit jobs until the explicit `sbatch` commands.

## Update the existing Locust checkout

Push the committed branch from the local worktree:

```bash
cd /private/tmp/AgenticSDS-gptoss20b
test -z "$(git status --porcelain)"
git push origin gptoss20b_standalone
```

On Locust, use the existing separate checkout. Stop if it contains local changes:

```bash
set -euo pipefail
cd "$HOME/AgenticSDS-gptoss20b"
test -z "$(git status --porcelain)"
git fetch origin gptoss20b_standalone
git switch --detach origin/gptoss20b_standalone
test -z "$(git status --porcelain)"
git rev-parse HEAD
export FRAMEWORK_COMMIT="$(git rev-parse HEAD)"
export AGENT_SUPPORT_DATA="/home/evanpan/agent_support_data"
export GPT_VENV="/shared_data0/evanpan/agent_support/venvs/gptoss20b"
export GPT_HF_HOME="/shared_data0/evanpan/agent_support/hf_cache"
test -x "$GPT_VENV/bin/python"
test -d "$GPT_HF_HOME/hub/models--openai--gpt-oss-20b"
```

## Repair only the incompatible dependency and record the environment

```bash
"$GPT_VENV/bin/python" -m pip install --no-cache-dir --no-deps --upgrade 'kernels==0.17.2'
"$GPT_VENV/bin/python" -m pip check
"$GPT_VENV/bin/python" - <<'PY'
from importlib.metadata import version
from agent_support.gpt_oss_model import check_mxfp4_dependencies
expected = {
    'torch': '2.8.0+cu128', 'transformers': '5.22.2',
    'accelerate': '1.15.0', 'triton': '3.4.0',
    'kernels': '0.17.2', 'textcraft': '0.0.3', 'pytest': '8.4.2',
}
for name, pinned in expected.items():
    actual = version(name)
    assert actual == pinned, (name, actual, pinned)
    print(name, actual)
check_mxfp4_dependencies('5.22.2')
PY
"$GPT_VENV/bin/python" -m pytest -q
mkdir -p "$AGENT_SUPPORT_DATA/gptoss20b_freezes"
freeze="$(mktemp "$AGENT_SUPPORT_DATA/gptoss20b_freezes/after_mxfp4_fix_XXXXXXXX")"
"$GPT_VENV/bin/python" -m pip freeze --all > "$freeze"
sha256sum "$freeze" | tee "$freeze.sha256"
printf 'New freeze: %s\n' "$freeze"
```

The new file and `.sha256` are separate from the original dependency freeze. Keep both with the experiment records.

## Submit a new three-task GPT-OSS smoke on one L40S

Use a new output root. Slurm's job ID gives the run an additional unique directory; the wrapper resumes only that job's directory after a preemption.

```bash
cd "$HOME/AgenticSDS-gptoss20b"
test "$(git rev-parse HEAD)" = "$FRAMEWORK_COMMIT"
export HF_HOME="$GPT_HF_HOME"
export VENV_PATH="$GPT_VENV"
export AGENT_SUPPORT_CONFIG="$PWD/configs/standalone_textcraft_dev_gptoss20b_never.json"
export AGENT_SUPPORT_LIMIT=3
export AGENT_SUPPORT_OUTPUT_ROOT="$AGENT_SUPPORT_DATA/standalone_models/smoke_gptoss20b_mxfp4_fix_$FRAMEWORK_COMMIT"
test ! -e "$AGENT_SUPPORT_OUTPUT_ROOT"
mkdir -p "$AGENT_SUPPORT_OUTPUT_ROOT" "$AGENT_SUPPORT_DATA/standalone_models/logs"
GPT_SMOKE_JOB="$(sbatch --parsable --export=ALL --account=hassani --partition=standby --qos=standby --gres=gpu:l40s:1 --cpus-per-task=4 --mem=32G --time=04:00:00 --output="$AGENT_SUPPORT_DATA/standalone_models/logs/%j.out" --open-mode=append scripts/locust_modular_job.sh)"
GPT_SMOKE_JOB="${GPT_SMOKE_JOB%%;*}"
printf 'GPT_SMOKE_JOB=%s\nGPT_SMOKE_ROOT=%s\n' "$GPT_SMOKE_JOB" "$AGENT_SUPPORT_OUTPUT_ROOT"
squeue -j "$GPT_SMOKE_JOB"
```

After the job completes, audit the actual representation and the generated actions. Run this in the same shell, or set `GPT_SMOKE_JOB` and `AGENT_SUPPORT_OUTPUT_ROOT` to the printed values:

```bash
"$GPT_VENV/bin/python" - "$AGENT_SUPPORT_OUTPUT_ROOT/textcraft_slurm_${GPT_SMOKE_JOB}" "$FRAMEWORK_COMMIT" <<'PY'
import json
from pathlib import Path
import sys

root, commit = Path(sys.argv[1]), sys.argv[2]
meta = json.loads((root / 'run.json').read_text())
summary = json.loads((root / 'summary.json').read_text())
model = meta['model_metadata']
assert meta['status'] == 'complete' and meta['git_dirty'] is False
assert meta['git_commit'] == commit and meta['selected_task_count'] == 3
assert meta['model_revision'] == '6cee5e81ee83917806bbde320786a8fb61efebee'
assert model['quantization'] == model['quantization_method'] + '_packed'
assert model['verified_expert_modules'] == 24
assert model['verified_packed_projections'] == 48
assert model['representation_type'].endswith('.Tensor')
assert model['weight_layout_type'] == model['scale_layout_type'] == 'StridedLayout'
assert model['weight_storage_dtype'] == model['scale_storage_dtype'] == 'torch.uint8'
assert model['backing_storage_device'] == 'cuda:0'
assert model['kernels_version'] == '0.17.2'
assert model['packed_weight_storage_bytes'] > 0 and model['scale_storage_bytes'] > 0
assert summary['tasks'] == 3 and summary['environment_actions'] > 0
assert summary['peak_gpu_memory_allocated_bytes'] > 0
episodes = [json.loads(p.read_text()) for p in sorted((root / 'episodes').glob('*.json'))]
assert len(episodes) == 3
for episode in episodes:
    assert episode['model_calls'] > 0
    for step in episode['trace']:
        assert step['generation_status'] in {
            'complete', 'missing_final', 'malformed_final', 'truncated_final',
            'truncated_before_final', 'malformed_harmony',
        }
        assert 'raw_completion' in step and 'action' in step
    print(episode['task_id'], episode['status'], episode['environment_actions'],
          [(step['generation_status'], step['action']) for step in episode['trace']])
print('quantization:', model['quantization'], model['representation_type'])
print('experts/projections:', model['verified_expert_modules'], model['verified_packed_projections'])
print('packed/scale bytes:', model['packed_weight_storage_bytes'], model['scale_storage_bytes'])
print('actions/failures:', summary['environment_actions'], summary['generation_failures'])
print('peak allocated/reserved GiB:',
      round(summary['peak_gpu_memory_allocated_bytes'] / 2**30, 2),
      round(summary['peak_gpu_memory_reserved_bytes'] / 2**30, 2))
PY
```

If the GPU smoke fails, inspect `run.json` (if present) and `$AGENT_SUPPORT_DATA/standalone_models/logs/${GPT_SMOKE_JOB}.out`. The exception now names the failing expert projection or scale condition. A completed run with no environment actions does not pass this smoke gate.

## Rerun the 20-task Qwen reference at the corrected commit

The previous Qwen run at `3d9db8c` cannot be paired with the corrected GPT-OSS run by the existing same-commit report check. After the GPT-OSS smoke passes, use the original Qwen environment and Qwen cache with a new output root. This leaves the old reference intact.

```bash
cd "$HOME/AgenticSDS-gptoss20b"
test "$(git rev-parse HEAD)" = "$FRAMEWORK_COMMIT"
export QWEN_VENV="$HOME/AgenticSDS/.venv"
export QWEN_HF_HOME="$AGENT_SUPPORT_DATA/hf_cache"
test -x "$QWEN_VENV/bin/python"
test -d "$QWEN_HF_HOME/hub/models--Qwen--Qwen2.5-7B-Instruct"
export HF_HOME="$QWEN_HF_HOME"
export VENV_PATH="$QWEN_VENV"
export AGENT_SUPPORT_CONFIG="$PWD/configs/standalone_textcraft_dev_qwen_never.json"
unset AGENT_SUPPORT_LIMIT
export AGENT_SUPPORT_OUTPUT_ROOT="$AGENT_SUPPORT_DATA/standalone_models/full_qwen_mxfp4_fix_$FRAMEWORK_COMMIT"
test ! -e "$AGENT_SUPPORT_OUTPUT_ROOT"
mkdir -p "$AGENT_SUPPORT_OUTPUT_ROOT" "$AGENT_SUPPORT_DATA/standalone_models/logs"
QWEN_FULL_JOB="$(sbatch --parsable --export=ALL --account=hassani --partition=standby --qos=standby --gres=gpu:l40s:1 --cpus-per-task=4 --mem=32G --time=04:00:00 --output="$AGENT_SUPPORT_DATA/standalone_models/logs/%j.out" --open-mode=append scripts/locust_modular_job.sh)"
QWEN_FULL_JOB="${QWEN_FULL_JOB%%;*}"
printf 'QWEN_FULL_JOB=%s\nQWEN_FULL_ROOT=%s\n' "$QWEN_FULL_JOB" "$AGENT_SUPPORT_OUTPUT_ROOT"
squeue -j "$QWEN_FULL_JOB"
```

For the later full GPT-OSS run, use the same `FRAMEWORK_COMMIT` and a separate new output root. Then pass both completed 20-task run directories to `python -m agent_support.standalone_model_report` as shown in the [standalone comparison guide](gpt_oss_20b_standalone.md#full-20-task-development-runs-and-report). The report's exact commit check remains unchanged.
