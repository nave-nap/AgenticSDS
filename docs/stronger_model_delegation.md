# Stronger-model delegation on TextCraft development tasks

The controller uses the existing TextCraft prompt, parser, environment step, and completed-episode checkpoint logic. `stronger_model_delegation` is a pre-generation support: the policy selects it before any Qwen generation, and the support requests GPT-OSS as the one action generator for that decision. The next decision returns to the policy. The initial hybrid configurations use only this support. GPT-OSS output is never added to Qwen's context as reasoning; Qwen sees the delegated action and its environment observation in ordinary history.

The two workers are persistent local subprocesses. Qwen uses its original interpreter and cache; GPT-OSS uses its separate interpreter and shared cache. Their JSON-line protocol runs over stdin/stdout, with worker diagnostics in per-run `workers/*.log`. Each worker loads once. A stable hash of the configured seed, task ID, and model type reseeds each worker before each incomplete episode. This makes policy draws and episode seeds independent of task scheduling. GPT-OSS sampling is not promised to be bitwise identical across process restarts.

The hybrid run charges each attempted GPT-OSS generation as one total model call and one support call. A Qwen fallback, if enabled and budgeted, is an additional total model call. Its result is recorded as a second `generation_attempts` entry on the same decision. Output tokens include reasoning tokens, so total tokens are input plus output. The `model_usage` and delegation counters appear in each episode and in `summary.json`. A selected support blocked by call or token budget is recorded separately from an attempted delegation. The pre-generation token reservation is recorded alongside measured tokens after generation; an actual overrun fails visibly.

## Local branch and Locust checkout

From the local isolated worktree after committing:

```bash
cd /private/tmp/AgenticSDS-delegation
test -z "$(git status --porcelain)"
git push -u origin stronger_model_delegation
```

On Locust, keep the existing standalone checkout and artifacts. Use a new worktree:

```bash
set -euo pipefail
cd "$HOME/AgenticSDS-gptoss20b"
git fetch origin stronger_model_delegation
git worktree add "$HOME/AgenticSDS-hybrid" origin/stronger_model_delegation
cd "$HOME/AgenticSDS-hybrid"
test -z "$(git status --porcelain)"
export HYBRID_COMMIT="$(git rev-parse HEAD)"
export AGENT_SUPPORT_DATA="$HOME/agent_support_data"
export AGENT_SUPPORT_QWEN_PYTHON="$HOME/AgenticSDS/.venv/bin/python"
export AGENT_SUPPORT_QWEN_HF_HOME="$AGENT_SUPPORT_DATA/hf_cache"
export AGENT_SUPPORT_GPTOSS_PYTHON="/shared_data0/$USER/agent_support/venvs/gptoss20b/bin/python"
export AGENT_SUPPORT_GPTOSS_HF_HOME="/shared_data0/$USER/agent_support/hf_cache"
test -x "$AGENT_SUPPORT_QWEN_PYTHON"
test -x "$AGENT_SUPPORT_GPTOSS_PYTHON"
test -d "$AGENT_SUPPORT_QWEN_HF_HOME/hub/models--Qwen--Qwen2.5-7B-Instruct"
test -d "$AGENT_SUPPORT_GPTOSS_HF_HOME/hub/models--openai--gpt-oss-20b"
mkdir -p "$AGENT_SUPPORT_DATA/hybrid_delegation/logs"
```

Keep the `TRITON_CACHE_DIR` and Python header environment used by successful GPT-OSS job 146347 in the submission shell. The subprocesses inherit those settings. These instructions do not modify either environment, cache, or model revision.

Validate configurations and CPU tests before GPU submission:

```bash
"$AGENT_SUPPORT_QWEN_PYTHON" - <<'PY'
from pathlib import Path
from agent_support.common import read_json, validate_config, validate_manifest
from agent_support.components import load_components
from agent_support.run_store import audit_partitions
for path in sorted(Path('configs').glob('hybrid_textcraft_dev_*.json')):
    config = read_json(path)
    load_components(config)
    manifest = read_json(config['manifest'])
    validate_manifest(manifest)
    validate_config(config, 'textcraft')
    audit_partitions(config, manifest)
    assert manifest['partition'] == 'development' and len(manifest['tasks']) == 20
    print(path, config['selection_policy'], config['max_support_calls'])
PY
"$AGENT_SUPPORT_QWEN_PYTHON" -m pytest -q
```

## Phase A: one-L40S worker co-residency smoke

The smoke loads both workers once, makes four alternating generation requests on the first development task, parses all four actions, and saves each worker's CUDA allocated/reserved memory plus `nvidia-smi` used memory. No environment action is executed. Run this before hybrid episodes:

```bash
cd "$HOME/AgenticSDS-hybrid"
export AGENT_SUPPORT_CONFIG="$PWD/configs/hybrid_textcraft_dev_smoke_always.json"
export AGENT_SUPPORT_OUTPUT_ROOT="$AGENT_SUPPORT_DATA/hybrid_delegation/worker_smoke_$HYBRID_COMMIT"
test ! -e "$AGENT_SUPPORT_OUTPUT_ROOT"
mkdir -p "$AGENT_SUPPORT_OUTPUT_ROOT"
WORKER_JOB="$(sbatch --parsable --export=ALL --account=hassani --partition=standby --qos=standby --gres=gpu:l40s:1 --cpus-per-task=8 --mem=64G --time=01:00:00 --output="$AGENT_SUPPORT_DATA/hybrid_delegation/logs/%j.out" --open-mode=append scripts/locust_hybrid_worker_smoke.sh)"
WORKER_JOB="${WORKER_JOB%%;*}"
printf 'WORKER_JOB=%s\n' "$WORKER_JOB"
squeue -j "$WORKER_JOB"
```

After completion, inspect the artifact and the worker logs:

```bash
"$AGENT_SUPPORT_QWEN_PYTHON" - "$AGENT_SUPPORT_OUTPUT_ROOT/worker_smoke_${WORKER_JOB}.json" <<'PY'
import json, sys
from pathlib import Path
data = json.loads(Path(sys.argv[1]).read_text())
workers = data['model_metadata']['workers']
assert workers['primary']['metadata']['model_revision'] == 'a09a35458c702b33eeacc393d103063234e8bc28'
strong = workers['delegated']['metadata']
assert strong['model_revision'] == '6cee5e81ee83917806bbde320786a8fb61efebee'
assert strong['quantization'] == 'mxfp4_packed' and strong['verified_expert_modules'] == 24
assert [x['model_type'] for x in data['samples']] == ['qwen','gpt_oss_20b'] * 2
assert all(x['parsed_action'] and x['generation'].get('generation_status','complete') == 'complete' for x in data['samples'])
assert len(set(data['worker_pids'].values())) == 2
for item in data['samples']:
    print(item['model_type'], item['parsed_action'], item['memory_after'], item['gpu_memory_mib'])
print('worker PIDs', data['worker_pids'], 'final GPU memory MiB', data['gpu_memory_mib'])
PY
```

If loading both workers on one GPU fails, inspect `workers/primary_worker.log`, `workers/delegated_worker.log`, and the Slurm log. The likely capacity alternative is two L40S devices on one node, with one persistent worker bound to each device; it needs explicit device affinity and a new smoke before using it. Do not silently offload or reload a model per decision.

## Phase B: three-task hybrid execution smoke

The always policy selects GPT-OSS until the three-delegation per-task cap. The budget then leaves later decisions to Qwen, so this smoke checks both routes without depending on random draws. Submit only after Phase A passes:

```bash
cd "$HOME/AgenticSDS-hybrid"
export AGENT_SUPPORT_CONFIG="$PWD/configs/hybrid_textcraft_dev_smoke_always.json"
export AGENT_SUPPORT_LIMIT=3
export VENV_PATH="$(dirname "$(dirname "$AGENT_SUPPORT_QWEN_PYTHON")")"
export HF_HOME="$AGENT_SUPPORT_QWEN_HF_HOME"
export AGENT_SUPPORT_OUTPUT_ROOT="$AGENT_SUPPORT_DATA/hybrid_delegation/episode_smoke_$HYBRID_COMMIT"
test ! -e "$AGENT_SUPPORT_OUTPUT_ROOT"
mkdir -p "$AGENT_SUPPORT_OUTPUT_ROOT"
HYBRID_SMOKE_JOB="$(sbatch --parsable --export=ALL --account=hassani --partition=standby --qos=standby --gres=gpu:l40s:1 --cpus-per-task=8 --mem=64G --time=04:00:00 --output="$AGENT_SUPPORT_DATA/hybrid_delegation/logs/%j.out" --open-mode=append scripts/locust_modular_job.sh)"
HYBRID_SMOKE_JOB="${HYBRID_SMOKE_JOB%%;*}"
printf 'HYBRID_SMOKE_JOB=%s\n' "$HYBRID_SMOKE_JOB"
squeue -j "$HYBRID_SMOKE_JOB"
```

Audit the finished run:

```bash
"$AGENT_SUPPORT_QWEN_PYTHON" - "$AGENT_SUPPORT_OUTPUT_ROOT/textcraft_slurm_${HYBRID_SMOKE_JOB}" "$HYBRID_COMMIT" <<'PY'
import json, sys
from pathlib import Path
root, commit = Path(sys.argv[1]), sys.argv[2]
meta = json.loads((root/'run.json').read_text())
summary = json.loads((root/'summary.json').read_text())
assert meta['status'] == 'complete' and meta['git_commit'] == commit and not meta['git_dirty']
assert summary['tasks'] == 3 and summary['delegation_attempts'] > 0
assert summary['model_usage']['qwen']['model_calls'] > 0
assert summary['model_usage']['gpt_oss_20b']['model_calls'] == summary['delegation_attempts']
assert summary['model_calls'] == sum(x['model_calls'] for x in summary['model_usage'].values())
assert summary['support_calls'] == summary['delegation_attempts']
assert summary['delegation_selections'] == summary['delegation_attempts'] + summary['blocked_delegations']
assert summary['environment_actions'] <= summary['model_decisions']
for path in sorted((root/'episodes').glob('*.json')):
    episode = json.loads(path.read_text())
    print(episode['task_id'], episode['success'], episode['model_usage'], episode['delegation_attempts'])
    for step in episode['trace']:
        assert len(step['generation_attempts']) in (1,2)
        print(step['decision'], step['support_policy']['before_decision']['reason'],
              step['selected_generation_model'], step['actual_generation_model'],
              step['delegation_blocked'], step['fallback_occurred'],
              step['action'], step.get('environment_failed_action'))
print('summary', summary)
PY
```

The wrapper resumes a requeued preempted job using its original `SLURM_JOB_ID`; completed episodes remain immutable. `run.json` records resume events and the checkpoint contains episode hashes. The normal Slurm requeue path handles resumption without another submission.

## Phase C: matched 20-task development runs

After Phase B passes, submit four fresh runs sequentially. Each job uses one L40S. The standalone paths remain in their original environments; hybrid runs use the Qwen controller interpreter and both workers. The `afterok` dependencies prevent overlap and stop later jobs if an earlier run fails.

```bash
cd "$HOME/AgenticSDS-hybrid"
unset AGENT_SUPPORT_LIMIT
export HF_HOME="$AGENT_SUPPORT_QWEN_HF_HOME"
export VENV_PATH="$(dirname "$(dirname "$AGENT_SUPPORT_QWEN_PYTHON")")"
export AGENT_SUPPORT_CONFIG="$PWD/configs/standalone_textcraft_dev_qwen_never.json"
export AGENT_SUPPORT_OUTPUT_ROOT="$AGENT_SUPPORT_DATA/hybrid_delegation/full_qwen_$HYBRID_COMMIT"
test ! -e "$AGENT_SUPPORT_OUTPUT_ROOT"
mkdir -p "$AGENT_SUPPORT_OUTPUT_ROOT"
QWEN_JOB="$(sbatch --parsable --export=ALL --account=hassani --partition=standby --qos=standby --gres=gpu:l40s:1 --cpus-per-task=8 --mem=64G --time=04:00:00 --output="$AGENT_SUPPORT_DATA/hybrid_delegation/logs/%j.out" --open-mode=append scripts/locust_modular_job.sh)"
QWEN_JOB="${QWEN_JOB%%;*}"

export HF_HOME="$AGENT_SUPPORT_GPTOSS_HF_HOME"
export VENV_PATH="$(dirname "$(dirname "$AGENT_SUPPORT_GPTOSS_PYTHON")")"
export AGENT_SUPPORT_CONFIG="$PWD/configs/standalone_textcraft_dev_gptoss20b_low_never.json"
export AGENT_SUPPORT_OUTPUT_ROOT="$AGENT_SUPPORT_DATA/hybrid_delegation/full_gptoss_$HYBRID_COMMIT"
test ! -e "$AGENT_SUPPORT_OUTPUT_ROOT"
mkdir -p "$AGENT_SUPPORT_OUTPUT_ROOT"
GPT_JOB="$(sbatch --parsable --export=ALL --dependency="afterok:$QWEN_JOB" --account=hassani --partition=standby --qos=standby --gres=gpu:l40s:1 --cpus-per-task=8 --mem=64G --time=04:00:00 --output="$AGENT_SUPPORT_DATA/hybrid_delegation/logs/%j.out" --open-mode=append scripts/locust_modular_job.sh)"
GPT_JOB="${GPT_JOB%%;*}"

export HF_HOME="$AGENT_SUPPORT_QWEN_HF_HOME"
export VENV_PATH="$(dirname "$(dirname "$AGENT_SUPPORT_QWEN_PYTHON")")"
export AGENT_SUPPORT_CONFIG="$PWD/configs/hybrid_textcraft_dev_random.json"
export AGENT_SUPPORT_OUTPUT_ROOT="$AGENT_SUPPORT_DATA/hybrid_delegation/full_random_$HYBRID_COMMIT"
test ! -e "$AGENT_SUPPORT_OUTPUT_ROOT"
mkdir -p "$AGENT_SUPPORT_OUTPUT_ROOT"
RANDOM_JOB="$(sbatch --parsable --export=ALL --dependency="afterok:$GPT_JOB" --account=hassani --partition=standby --qos=standby --gres=gpu:l40s:1 --cpus-per-task=8 --mem=64G --time=04:00:00 --output="$AGENT_SUPPORT_DATA/hybrid_delegation/logs/%j.out" --open-mode=append scripts/locust_modular_job.sh)"
RANDOM_JOB="${RANDOM_JOB%%;*}"

export AGENT_SUPPORT_CONFIG="$PWD/configs/hybrid_textcraft_dev_failure_triggered.json"
export AGENT_SUPPORT_OUTPUT_ROOT="$AGENT_SUPPORT_DATA/hybrid_delegation/full_failure_$HYBRID_COMMIT"
test ! -e "$AGENT_SUPPORT_OUTPUT_ROOT"
mkdir -p "$AGENT_SUPPORT_OUTPUT_ROOT"
FAILURE_JOB="$(sbatch --parsable --export=ALL --dependency="afterok:$RANDOM_JOB" --account=hassani --partition=standby --qos=standby --gres=gpu:l40s:1 --cpus-per-task=8 --mem=64G --time=04:00:00 --output="$AGENT_SUPPORT_DATA/hybrid_delegation/logs/%j.out" --open-mode=append scripts/locust_modular_job.sh)"
FAILURE_JOB="${FAILURE_JOB%%;*}"
printf 'QWEN_JOB=%s\nGPT_JOB=%s\nRANDOM_JOB=%s\nFAILURE_JOB=%s\n' "$QWEN_JOB" "$GPT_JOB" "$RANDOM_JOB" "$FAILURE_JOB"
```

When all four complete, build the paired report with integrity and provenance checks:

```bash
"$AGENT_SUPPORT_QWEN_PYTHON" -m agent_support.hybrid_report \
  --qwen "$AGENT_SUPPORT_DATA/hybrid_delegation/full_qwen_$HYBRID_COMMIT/textcraft_slurm_${QWEN_JOB}" \
  --gpt-oss-20b "$AGENT_SUPPORT_DATA/hybrid_delegation/full_gptoss_$HYBRID_COMMIT/textcraft_slurm_${GPT_JOB}" \
  --random "$AGENT_SUPPORT_DATA/hybrid_delegation/full_random_$HYBRID_COMMIT/textcraft_slurm_${RANDOM_JOB}" \
  --failure-triggered "$AGENT_SUPPORT_DATA/hybrid_delegation/full_failure_$HYBRID_COMMIT/textcraft_slurm_${FAILURE_JOB}" \
  --output "$AGENT_SUPPORT_DATA/hybrid_delegation/report_$HYBRID_COMMIT"
cat "$AGENT_SUPPORT_DATA/hybrid_delegation/report_$HYBRID_COMMIT/report.md"
```

The report records paired task outcomes, actual model calls by model, support selection/attempt/block counts, tokens, latency, failed actions, wall time, and per-task detail in `report.json`. It rejects different commits, task manifests, worker model configurations, or checkpoint hashes.
