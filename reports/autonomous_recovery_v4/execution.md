# Mac-to-Locust execution — user-run commands

No command below has been used to commit, push, or submit GPU work. Keep the running GPT-OSS control jobs and their output root untouched.

## Mac validation and branch transfer

The Mac worktree is `/Users/evanpan/AgenticSDS-autonomous-recovery-v4`, branching from committed `gptoss-controls-v1`. From that worktree:

```sh
cd /Users/evanpan/AgenticSDS-autonomous-recovery-v4
python3 -m pytest -q tests
python3 -m agent_support.recovery_v4_eval preflight --arm expanded_v3_v1
python3 -m agent_support.recovery_v4_eval preflight --arm autonomous_recovery_v4
bash -n scripts/recovery_v4_smoke.sbatch scripts/recovery_v4_batch.sbatch
git diff --check
git status --short
```

Mac preflight validates the frozen 30 tasks, disjointness, recipe order, and config. GPU dependencies and M2-200 artifacts may be unavailable locally; they must all pass on Locust before `init`. After review, stage only these v4 files, then commit and push **manually**:

```sh
git add -- agent_support/recovery_v4.py agent_support/recovery_v4_eval.py \
  agent_support/recovery_v4_analysis.py \
  configs/autonomous_recovery_v4/expanded_v3_v1.json \
  configs/autonomous_recovery_v4/autonomous_recovery_v4.json \
  manifests/textcraft/autonomous_recovery30_v1.json \
  scripts/recovery_v4_smoke.sbatch scripts/recovery_v4_batch.sbatch \
  tests/test_recovery_v4.py tests/test_recovery_v4_eval.py \
  tests/test_recovery_v4_analysis.py \
  reports/autonomous_recovery_v4/architecture.md \
  reports/autonomous_recovery_v4/execution.md \
  reports/autonomous_recovery_v4/cpu_preflight.md
git diff --cached --check
git commit -m "Add exploratory autonomous recovery v4"
git push -u origin autonomous-recovery-v4
```

On Locust, use a new worktree. Set `LOCUST_HOST` to the actual SSH hostname on the Mac:

```sh
ssh "$LOCUST_HOST"
cd "$HOME/AgenticSDS"
git fetch origin autonomous-recovery-v4
DEPLOY_REPO="$(dirname "$(git rev-parse --show-toplevel)")/AgenticSDS-autonomous-recovery-v4"
test ! -e "$DEPLOY_REPO"
git worktree add --detach "$DEPLOY_REPO" origin/autonomous-recovery-v4
cd "$DEPLOY_REPO"
git status --porcelain
sinfo -o '%P %G %m %l'
```

Assign the same pinned environments and read-only memory artifact directory used by v3, but choose a **new** output root:

```sh
export AGENT_SUPPORT_DATA=/shared_data0/evanpan/agent_support
export RECOVERY_ROOT="$AGENT_SUPPORT_DATA/autonomous_recovery_v4/exploratory30_v1"
export VENV_PATH="<existing-Qwen-environment-directory>"
export AGENT_SUPPORT_GPTOSS_PYTHON="<existing-pinned-GPT-OSS-python>"
export AGENT_SUPPORT_GPTOSS_HF_HOME="<existing-GPT-OSS-cache-root>"
export HF_HOME="<existing-Qwen-cache-root>"
export AGENT_SUPPORT_M2_200_ARTIFACT_DIR=/shared_data0/evanpan/agent_support/gptoss_memory_v1/memory_scaling_v1/artifacts
export GPU_PARTITION="<verified-L40S-partition>"
test -x "$VENV_PATH/bin/python"
test -x "$AGENT_SUPPORT_GPTOSS_PYTHON"
test -r "$AGENT_SUPPORT_M2_200_ARTIFACT_DIR/corpus_200.json"
test -r "$AGENT_SUPPORT_M2_200_ARTIFACT_DIR/index_200.json"
"$VENV_PATH/bin/python" -m agent_support.recovery_v4_eval preflight --arm expanded_v3_v1
"$VENV_PATH/bin/python" -m agent_support.recovery_v4_eval preflight --arm autonomous_recovery_v4
```

All `runtime_checks` must be true, and `memory.available` must be true with `overlap: 0`. Then run 2-task development smoke jobs, one arm at a time, and inspect traces, GPU metadata, model-call counts, support costs, wall time, and in-task checkpoints. Do not use held-out50 tasks for debugging:

```sh
mkdir -p "$RECOVERY_ROOT/logs"
sbatch --partition="$GPU_PARTITION" --export=ALL,RECOVERY_ARM=expanded_v3_v1 \
  --output="$RECOVERY_ROOT/logs/expanded-v3-smoke-%j.out" scripts/recovery_v4_smoke.sbatch
sbatch --partition="$GPU_PARTITION" --export=ALL,RECOVERY_ARM=autonomous_recovery_v4 \
  --output="$RECOVERY_ROOT/logs/recovery-v4-smoke-%j.out" scripts/recovery_v4_smoke.sbatch
```

Use `sacct -j <job-id> --format=JobID,Elapsed,MaxRSS,AllocTRES,State` and each run's worker log to assess startup and memory. The scripts request one L40S, 64 GiB host RAM, and 2.5 hours for three possible 40-minute tasks plus startup. Adjust Slurm time if measured smoke runtimes require it; do not change algorithm budgets after seeing exploratory outcomes.

After smoke review, initialize the non-overwriting plan. It locks the commit, Python/packages, configs, manifest, recipe order, memory paths and digests:

```sh
test ! -e "$RECOVERY_ROOT/plan.json"
"$VENV_PATH/bin/python" -m agent_support.recovery_v4_eval init --root "$RECOVERY_ROOT"
```

There are ten deterministic batches of three tasks per arm. Check status before each submission. The example is batch 0; repeat batches 1–9, and resubmit the same batch number after interruption. Completed task snapshots are never rerun. An interrupted task restarts from its initial observation; its partial checkpoint remains available for audit.

```sh
"$VENV_PATH/bin/python" -m agent_support.recovery_v4_eval status \
  --root "$RECOVERY_ROOT" --arm expanded_v3_v1 --batch 0
sbatch --partition="$GPU_PARTITION" --export=ALL,RECOVERY_ARM=expanded_v3_v1,RECOVERY_BATCH=0 \
  --output="$RECOVERY_ROOT/logs/expanded-v3-b00-%j.out" scripts/recovery_v4_batch.sbatch
"$VENV_PATH/bin/python" -m agent_support.recovery_v4_eval status \
  --root "$RECOVERY_ROOT" --arm autonomous_recovery_v4 --batch 0
sbatch --partition="$GPU_PARTITION" --export=ALL,RECOVERY_ARM=autonomous_recovery_v4,RECOVERY_BATCH=0 \
  --output="$RECOVERY_ROOT/logs/recovery-v4-b00-%j.out" scripts/recovery_v4_batch.sbatch
```

Set `RECOVERY_MAX_TASKS_PER_ATTEMPT=1` or `2` in `--export=ALL,...` if a shorter job is needed. Submit only when `missing_indices` is nonempty. A per-batch file lock prevents concurrent attempts.

After both arms cover all 30 tasks, write new reports without overwriting existing outputs:

```sh
mkdir -p "$RECOVERY_ROOT/reports"
"$VENV_PATH/bin/python" -m agent_support.recovery_v4_eval aggregate \
  --root "$RECOVERY_ROOT" --arm expanded_v3_v1 \
  --output "$RECOVERY_ROOT/reports/expanded_v3_v1.json"
"$VENV_PATH/bin/python" -m agent_support.recovery_v4_eval aggregate \
  --root "$RECOVERY_ROOT" --arm autonomous_recovery_v4 \
  --output "$RECOVERY_ROOT/reports/autonomous_recovery_v4.json"
"$VENV_PATH/bin/python" -m agent_support.recovery_v4_analysis \
  --root "$RECOVERY_ROOT" --output "$RECOVERY_ROOT/reports/paired_exploratory_v1.json"
```

The paired report separates success from computation and includes descriptive trajectory associations. It does not establish that a particular support or controller decision caused recovery. A fresh held-out set is required for final validation.
