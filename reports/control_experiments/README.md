# TextCraft GPT-OSS controls v1

These two experiments are exploratory follow-ups to the known held-out50 result. They use `heldout50_v1.json` (SHA-256 `614de2034ad3fa14063a9fb5d6013a6754212584020b3e994a7cdb65f5b84c88`), TextCraft 0.0.3, the pinned recipe order, 30 decisions, and 30 environment actions. The v3 implementation and existing evaluation outputs are unchanged.

- `gptoss_standalone_v1`: GPT-OSS alone proposes native TextCraft actions. Its final channel must contain one `Action: ...` line. A malformed or incomplete generation uses a decision but no environment action.
- `gptoss_review_only_v1`: Qwen uses its existing proposal prompt and parser. Invalid Qwen proposals take the v3 parse-error path without a reviewer call. One GPT-OSS call reviews each valid proposal and emits strict accept/revise JSON. Invalid reviewer output falls back to Qwen's proposal and is counted as a reviewer error. No support operation is available to either arm.

The GPT-OSS worker uses the exact v3 worker specification, model and tokenizer revision, low reasoning effort, decoding settings, per-task seed derivation, and Harmony final-channel parser. The reviewer sees the ordinary task, current observation, recent action/feedback history, Qwen proposal, and remaining budgets. It does not see support capabilities, support history, hidden state, or retrieved memory. The standalone agent sees the same task evidence without a Qwen proposal. The control prompts are new and model appropriate, so prompt wording is an unavoidable protocol difference from v3. Qwen's review-only proposals use the same `textcraft_messages` and `parse_textcraft` functions as v3.

Each trace stores input messages, full model generation objects (including GPT-OSS raw Harmony completion and token counts), proposed and executed actions, environment outcomes, and parse or generation errors. Per-task summaries include calls and tokens by model, inference time, failed actions, action revisions, and zero external support operations. If a backend raises before returning usage, its attempted call and `usage_unavailable` flag are saved; tokens and inference seconds cannot be recovered for that call.

## Mac CPU checks and user-run transfer

From `/Users/evanpan/AgenticSDS`:

```sh
python3 -m pytest -q tests
.venv/bin/python -m agent_support.control_experiments preflight --arm gptoss_standalone_v1
.venv/bin/python -m agent_support.control_experiments preflight --arm gptoss_review_only_v1
bash -n scripts/control_experiment_smoke.sbatch scripts/control_experiment_heldout50.sbatch
git diff --check
git status --short
```

The Mac CPU preflight verifies frozen config, manifest, recipe order and installed TextCraft. Its GPT-OSS executable/cache and review-only Qwen dependency checks are expected to be false on the Mac. They must be true in the Locust Qwen environment before plan initialization. The GPT-OSS worker itself verifies the pinned Transformers/MXFP4 backend, model revision, tokenizer revision, packed weights, and GPU at startup. The smoke jobs verify simultaneous model loading on an L40S.

After review, the user can stage only the control files, commit, and push. The unrelated untracked `reports/complete_research_summary/` must remain untouched. These commands are supplied for deployment; they have not been run:

```sh
git switch -c gptoss-controls-v1
git add -- agent_support/control_experiments.py agent_support/control_experiment_analysis.py \
  configs/control_experiments/gptoss_standalone_v1.json \
  configs/control_experiments/gptoss_review_only_v1.json \
  scripts/control_experiment_smoke.sbatch scripts/control_experiment_heldout50.sbatch \
  tests/test_control_experiments.py tests/test_control_experiment_analysis.py \
  reports/control_experiments/README.md reports/control_experiments/cpu_preflight.md
git diff --cached --check
git commit -m "Add GPT-OSS TextCraft control experiments"
git push -u origin gptoss-controls-v1
```

On Locust, use a new worktree from that pushed commit. Set `LOCUST_HOST` to the actual SSH hostname if connecting from the Mac:

```sh
ssh "$LOCUST_HOST"
cd "$HOME/AgenticSDS"
git fetch origin gptoss-controls-v1
DEPLOY_REPO="$(dirname "$(git rev-parse --show-toplevel)")/AgenticSDS-gptoss-controls-v1"
test ! -e "$DEPLOY_REPO"
git worktree add --detach "$DEPLOY_REPO" origin/gptoss-controls-v1
cd "$DEPLOY_REPO"
git status --porcelain
sinfo -o '%P %G %m %l'
```

Use the already pinned Qwen and GPT-OSS Python environments and caches from v3. The following commands are exact once the discovered environment paths and L40S partition are assigned:

```sh
export AGENT_SUPPORT_DATA=/shared_data0/evanpan/agent_support
export CONTROL_ROOT="$AGENT_SUPPORT_DATA/gptoss_controls_v1/heldout50_v1"
export VENV_PATH="<existing-Qwen-environment-directory>"
export AGENT_SUPPORT_GPTOSS_PYTHON="<existing-pinned-GPT-OSS-python>"
export AGENT_SUPPORT_GPTOSS_HF_HOME="<existing-GPT-OSS-model-cache-root>"
export HF_HOME="<existing-Qwen-model-cache-root>"
export GPU_PARTITION="<verified-L40S-partition>"
test -x "$VENV_PATH/bin/python"
test -x "$AGENT_SUPPORT_GPTOSS_PYTHON"
test -d "$AGENT_SUPPORT_GPTOSS_HF_HOME"
"$VENV_PATH/bin/python" -m agent_support.control_experiments preflight --arm gptoss_standalone_v1
"$VENV_PATH/bin/python" -m agent_support.control_experiments preflight --arm gptoss_review_only_v1
mkdir -p "$CONTROL_ROOT/logs"
```

Inspect both preflight JSON records; every `runtime_checks` value must be true. Run development smoke jobs before the held-out plan. They use only task index 0 from the frozen 20-task development manifest:

```sh
sbatch --partition="$GPU_PARTITION" --export=ALL,CONTROL_ARM=gptoss_standalone_v1 \
  --output="$CONTROL_ROOT/logs/standalone-smoke-%j.out" scripts/control_experiment_smoke.sbatch
sbatch --partition="$GPU_PARTITION" --export=ALL,CONTROL_ARM=gptoss_review_only_v1 \
  --output="$CONTROL_ROOT/logs/review-smoke-%j.out" scripts/control_experiment_smoke.sbatch
```

Inspect both `smoke/*/textcraft_*/episode.json` and `run.json` files. Check valid and failed environment actions, one review call per valid Qwen proposal, exact zero support operations, model calls, token totals, inference times, generation failures, and GPU metadata. Re-run a development smoke with another `CONTROL_DEV_INDEX` if the first task does not exercise a failed action or revision; no held-out task should be used for debugging. Verify Slurm memory use with `sacct -j <job-id> --format=JobID,Elapsed,MaxRSS,AllocTRES,State` and the worker log. The scripts request 64 GiB host memory and one L40S; the worker rejects another GPU type. Adjust Slurm wall time after the smoke measurement if necessary, without changing task budgets.

Initialize the new output root only after reviewing the smoke traces. This writes a non-overwriting plan locked to the clean commit, frozen task/config hashes, Python, and package versions:

```sh
test ! -e "$CONTROL_ROOT/plan.json"
"$VENV_PATH/bin/python" -m agent_support.control_experiments init --root "$CONTROL_ROOT"
```

Submit each arm in batches 0 through 9. Check status first; resubmit the **same** batch after preemption. Completed episode snapshots are skipped, and a batch file lock prevents concurrent attempts. `CONTROL_MAX_TASKS_PER_ATTEMPT=2` can shorten a job without changing batch membership. The example submits batch 0:

```sh
"$VENV_PATH/bin/python" -m agent_support.control_experiments status \
  --root "$CONTROL_ROOT" --arm gptoss_standalone_v1 --batch 0
sbatch --partition="$GPU_PARTITION" --export=ALL,CONTROL_ARM=gptoss_standalone_v1,CONTROL_BATCH=0 \
  --output="$CONTROL_ROOT/logs/standalone-b00-%j.out" scripts/control_experiment_heldout50.sbatch
"$VENV_PATH/bin/python" -m agent_support.control_experiments status \
  --root "$CONTROL_ROOT" --arm gptoss_review_only_v1 --batch 0
sbatch --partition="$GPU_PARTITION" --export=ALL,CONTROL_ARM=gptoss_review_only_v1,CONTROL_BATCH=0 \
  --output="$CONTROL_ROOT/logs/review-b00-%j.out" scripts/control_experiment_heldout50.sbatch
```

For each batch, submit only when `missing_indices` is nonempty. After all 50 tasks in each arm, write aggregates without overwriting prior outputs:

```sh
mkdir -p "$CONTROL_ROOT/reports"
"$VENV_PATH/bin/python" -m agent_support.control_experiments aggregate \
  --root "$CONTROL_ROOT" --arm gptoss_standalone_v1 \
  --output "$CONTROL_ROOT/reports/gptoss_standalone_v1.json"
"$VENV_PATH/bin/python" -m agent_support.control_experiments aggregate \
  --root "$CONTROL_ROOT" --arm gptoss_review_only_v1 \
  --output "$CONTROL_ROOT/reports/gptoss_review_only_v1.json"
```

## Versioned four-arm report

Use the existing validated Qwen and autonomous-v3 aggregates from the old held-out50 root. The report refuses incomplete arms or task-observation mismatches. It produces each arm's success rate and Wilson interval, all requested paired disagreements and paired task bootstrap intervals, actions, failures, model calls and tokens by model, inference time, controller errors, revisions, and external support operations. `--gpu-hour-price` is optional and applies only to an inference-time lower bound. Actual allocated GPU cost requires Slurm accounting for all jobs, including startup and idle time.

```sh
export OLD_ROOT="$AGENT_SUPPORT_DATA/autonomous_support_evaluation/heldout50_v1"
"$VENV_PATH/bin/python" -m agent_support.control_experiment_analysis \
  --qwen-standalone "$OLD_ROOT/reports/qwen_standalone.json" \
  --gptoss-standalone-v1 "$CONTROL_ROOT/reports/gptoss_standalone_v1.json" \
  --gptoss-review-only-v1 "$CONTROL_ROOT/reports/gptoss_review_only_v1.json" \
  --autonomous-v3 "$OLD_ROOT/reports/autonomous_v3.json" \
  --output "$CONTROL_ROOT/reports/four_arm_exploratory_v1.json"
```

The report includes all six pairwise disagreements and intervals. The three requested differences are autonomous v3 minus GPT-OSS standalone, autonomous v3 minus GPT-OSS review-only, and GPT-OSS review-only minus Qwen standalone. Compare accuracy alongside added calls, tokens, and GPU hours. The control arms isolate GPT-OSS direct solving and action review from the external support toolbox, but the observed differences alone do not identify a causal effect of support selection. A fresh held-out task set is needed for final validation. No control configuration should be tuned using these 50 outcomes.
