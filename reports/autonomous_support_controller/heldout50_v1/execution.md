# Held-out50 deployment and execution protocol

## Locks and interpretation

`manifests/textcraft/heldout50_v1.json` is the first 50 eligible tasks in the
frozen seed-42 held-out manifest, skipping seed 51 before selection. Its
canonical digest is `614de2034ad3fa14063a9fb5d6013a6754212584020b3e994a7cdb65f5b84c88`.
The 100-task parent digest is `5c8dbc6caa2db221cc6ed4ce3208d9c53157af515563895a927957c3ac8349a0`.
The batch runner passes that original parent manifest to the ordinary runner,
so its M2 exclusion check uses the already frozen exclusion. The v3 config,
controller, prompts, capabilities, model revisions, and budgets are unchanged.
The legacy Qwen M2 run uses an older corpus (`d6e2e065…`/`6f5eabf7…`), so its
comparison is labeled secondary and partial.

`init` writes an exclusive `plan.json` in a new external output root. It locks
the batch size, code commit, Python/package versions, config digests, and any
reviewed historical sources, including a digest of their selected episode
contents. Each GPU attempt creates a unique `textcraft_*` run directory. An
interrupted attempt retains already written episode snapshots; the next
attempt selects only missing task indices. There is no mid-episode resume, so
an episode interrupted before its atomic write runs again. A batch file lock
prevents concurrent attempts, and duplicate completed task IDs cause an error.
`--max-tasks-per-attempt` can reduce the number attempted in one job without
changing the frozen batch mapping. Keep all output and old runs intact.

## Mac validation and transfer (user-run commands)

From `/Users/evanpan/AgenticSDS`:

```sh
python3 -m pytest -q tests
python3 -m agent_support.heldout50_evaluation preflight --arm autonomous_v3
python3 -m agent_support.heldout50_evaluation preflight --arm qwen_standalone
bash -n scripts/heldout50_eval.sbatch
git diff --check
git status --short
git add -- agent_support/runner.py agent_support/heldout50_evaluation.py \
  agent_support/heldout50_analysis.py manifests/textcraft/heldout50_v1.json \
  scripts/heldout50_eval.sbatch tests/test_heldout50_evaluation.py \
  tests/test_heldout50_analysis.py \
  reports/autonomous_support_controller/heldout50_v1/historical_audit.md \
  reports/autonomous_support_controller/heldout50_v1/historical_run_inventory.csv \
  reports/autonomous_support_controller/heldout50_v1/historical_task_results.csv \
  reports/autonomous_support_controller/heldout50_v1/baseline_coverage_matrix.csv \
  reports/autonomous_support_controller/heldout50_v1/execution.md
git diff --cached --check
git diff --cached --name-only
git commit -m "Add frozen TextCraft held-out50 batched evaluation"
git push -u origin autonomous-smoke-v3
```

Do not stage the unrelated `reports/complete_research_summary/` directory.
These commit and push commands are preparation only; they were not run.

## Locust discovery before initialization

In an existing Locust AgenticSDS checkout, use a **new** sibling worktree so
previous smoke and memory results stay untouched:

```sh
cd "$HOME/AgenticSDS"
git status --short
git fetch origin autonomous-smoke-v3
DEPLOY_REPO="$(dirname "$(git rev-parse --show-toplevel)")/AgenticSDS-heldout50-v1"
test ! -e "$DEPLOY_REPO"
git worktree add --detach "$DEPLOY_REPO" origin/autonomous-smoke-v3
cd "$DEPLOY_REPO"
git status --porcelain
git cat-file -e HEAD:manifests/textcraft/heldout50_v1.json
git rev-parse HEAD
find /shared_data0/evanpan/agent_support "$HOME/agent_support_data" \
  -maxdepth 6 -path '*/bin/python*' -print 2>/dev/null
sinfo -o '%P %G %m %l'
```

Search the historical workflow locations before deciding that a new Qwen
baseline is needed. This read-only discovery prints exact candidate paths:

```sh
python3 - <<'PY'
import json
from pathlib import Path
roots = [Path.home() / "agent_support_data/results",
         Path.home() / "agent_support_data/heldout_evaluation",
         Path.home() / "agent_support_data/standalone_models",
         Path("/shared_data0/evanpan/agent_support")]
for root in roots:
    if not root.exists():
        continue
    for path in root.rglob("run.json"):
        try:
            run = json.loads(path.read_text())
            config = json.loads((path.parent / "config.json").read_text())
        except (OSError, ValueError):
            continue
        if (run.get("benchmark") == "textcraft" and run.get("partition") == "heldout"
                and config.get("model_id") == "Qwen/Qwen2.5-7B-Instruct"):
            print(path.parent, run.get("status"), run.get("selected_task_count"),
                  run.get("manifest_sha256"), run.get("git_commit"),
                  config.get("condition"), config.get("selection_policy"))
PY
```

The archived legacy Qwen M2 source path was
`$HOME/agent_support_data/replacement_evaluation/m2_runs/textcraft_20261110T052311Z_16f5bf04`.
Check that it exists on Locust. A candidate held-out Qwen standalone directory
must pass the evaluator's complete-run, task, observation, Qwen revision,
decoding, action-limit, recipe, and package checks. Review its old commit's
task-agent prompt, parser, model generation, and TextCraft adapter against the
new commit before supplying `--reviewed-legacy-commit`. A source from a
different protocol remains contextual even if task IDs match.

## CPU preflight and plan freeze

Use the same Qwen and GPT-OSS environments and caches as the v3 smoke. Set
the following discovered paths; only the frozen artifact root is fixed by the
previous Locust runs:

```sh
export AGENT_SUPPORT_DATA=/shared_data0/evanpan/agent_support
export AGENT_SUPPORT_M2_200_ARTIFACT_DIR=/shared_data0/evanpan/agent_support/gptoss_memory_v1/memory_scaling_v1/artifacts
export VENV_PATH="<existing-Qwen-environment-directory>"
export AGENT_SUPPORT_GPTOSS_PYTHON="<existing-pinned-GPT-OSS-python>"
export HF_HOME="<existing-Qwen-model-cache-root>"
export AGENT_SUPPORT_GPTOSS_HF_HOME="<existing-GPT-OSS-model-cache-root>"
export EVAL_ROOT="$AGENT_SUPPORT_DATA/autonomous_support_evaluation/heldout50_v1"
test -x "$VENV_PATH/bin/python"
test -x "$AGENT_SUPPORT_GPTOSS_PYTHON"
test -r "$AGENT_SUPPORT_M2_200_ARTIFACT_DIR/corpus_200.json"
test -r "$AGENT_SUPPORT_M2_200_ARTIFACT_DIR/index_200.json"
"$VENV_PATH/bin/python" -m agent_support.heldout50_evaluation preflight --arm autonomous_v3
"$VENV_PATH/bin/python" -m agent_support.heldout50_evaluation preflight --arm qwen_standalone
```

After historical discovery, initialize **once**. If a reviewed compatible
Qwen standalone run exists, append `--historical-qwen-run "$QWEN_HIST_RUN"
--reviewed-legacy-commit "$QWEN_HIST_COMMIT"` to the `init` command. Otherwise
leave those options out and the runner will evaluate all 50 standalone tasks.
The archived legacy M2 run should be available at its original Locust path:

```sh
export LEGACY_M2_RUN="$HOME/agent_support_data/replacement_evaluation/m2_runs/textcraft_20261110T052311Z_16f5bf04"
test -r "$LEGACY_M2_RUN/run.json"
test ! -e "$EVAL_ROOT/plan.json"
"$VENV_PATH/bin/python" -m agent_support.heldout50_evaluation init \
  --root "$EVAL_ROOT" --batch-size 5 \
  --legacy-m2-run "$LEGACY_M2_RUN" \
  --reviewed-m2-commit aba4da79b5584c1a2a14991c9556967515597451
cat "$EVAL_ROOT/plan.json"
```

If the legacy M2 path has moved, discover its run directory by run ID
`20261110T052311Z_16f5bf04` and verify the source before `init`. The plan
must be frozen before any GPU task. `init` requires a clean committed checkout
and TextCraft 0.0.3. The Mac's CPU preflight does not test Locust's GPU.

## First batch, resume, and aggregation

Choose a GPU partition from `sinfo`; do not assume its name. `--time=02:00:00`
can be overridden on `sbatch` if the partition and measured task duration
require it. The first batch contains five tasks. Inspect its longest episode
and Slurm elapsed time before scheduling the rest. Only submit a batch when
`missing_indices` is nonempty:

```sh
export GPU_PARTITION="<verified-L40S-partition>"
mkdir -p "$AGENT_SUPPORT_DATA/autonomous_support_evaluation/logs"
"$VENV_PATH/bin/python" -m agent_support.heldout50_evaluation status \
  --root "$EVAL_ROOT" --arm autonomous_v3 --batch 0
sbatch --partition="$GPU_PARTITION" \
  --export=ALL,EVAL_ARM=autonomous_v3,EVAL_BATCH=0 \
  --output="$AGENT_SUPPORT_DATA/autonomous_support_evaluation/logs/heldout50-autonomous-0-%j.out" \
  scripts/heldout50_eval.sbatch
```

After the job, run `status` again. If it was preempted, submit **the same
batch number**; completed tasks are skipped. For a wall-time concern, add
`EVAL_MAX_TASKS_PER_ATTEMPT=2` to `--export=ALL,...` and repeat that batch
until `missing_indices` is empty. Continue batch numbers 1 through 9 in order.
Run the same status/submit pattern for `qwen_standalone`; its locked historical
coverage is skipped automatically. Submit its jobs only for batches that
report missing tasks. Do not run a new Qwen job if all 50 are covered.

When all required batches finish, write new, non-overwriting reports:

```sh
mkdir -p "$EVAL_ROOT/reports"
"$VENV_PATH/bin/python" -m agent_support.heldout50_evaluation aggregate \
  --root "$EVAL_ROOT" --arm autonomous_v3 --output "$EVAL_ROOT/reports/autonomous_v3.json"
"$VENV_PATH/bin/python" -m agent_support.heldout50_evaluation aggregate \
  --root "$EVAL_ROOT" --arm qwen_standalone --output "$EVAL_ROOT/reports/qwen_standalone.json"
"$VENV_PATH/bin/python" -m agent_support.heldout50_analysis legacy-m2 \
  --root "$EVAL_ROOT" --output "$EVAL_ROOT/reports/qwen_legacy_m2.json"
"$VENV_PATH/bin/python" -m agent_support.heldout50_analysis compare \
  --autonomous "$EVAL_ROOT/reports/autonomous_v3.json" \
  --baseline "$EVAL_ROOT/reports/qwen_standalone.json" \
  --output "$EVAL_ROOT/reports/primary_paired.json"
"$VENV_PATH/bin/python" -m agent_support.heldout50_analysis compare \
  --autonomous "$EVAL_ROOT/reports/autonomous_v3.json" \
  --baseline "$EVAL_ROOT/reports/qwen_legacy_m2.json" \
  --output "$EVAL_ROOT/reports/secondary_legacy_m2_paired.json"
```

The autonomous aggregate contains complete support sequences and per-decision
costs in `support_sequences.decision_rows`. Individual trajectories remain in
`$EVAL_ROOT/runs/autonomous_v3/batch_*/textcraft_*/episodes/*.json`.
The paired reports contain actual paired sample sizes, success disagreements,
Wilson intervals, paired task-bootstrap intervals, actions, failures, calls,
tokens, and inference time. The comparator requires a shared locked plan digest
and exact task-observation hashes. Compute differences include GPT-OSS controller
costs; more support activity or action changes alone is not evidence of an
intervention's causal benefit.

## GPU-time planning anchor

The one archived v3 smoke episode used 313.5 seconds of episode time; its
whole run took about 14 minutes including model startup. A literal
extrapolation gives about 4.4 GPU-hours for 50 episodes plus roughly 1.5 hours
for ten five-task startups. **This is not a measured held-out runtime**:
controller calls and task lengths can vary substantially. Use the first
five-task batch to estimate maximum per-task time, then choose the Slurm wall
request and `--max-tasks-per-attempt` for later jobs. The historical 20-task
Qwen standalone development run recorded 193 seconds of model inference in
total, but held-out difficulty and startup make that an uncertain baseline
cost estimate. If a compatible held-out Qwen run is found, no Qwen GPU work is
needed for the tasks it covers.
