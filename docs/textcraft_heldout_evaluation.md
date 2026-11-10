# TextCraft 100-task held-out evaluation

## Audited task pool and limits

The original committed TextCraft manifests were generated from seeds 42–161 and shuffled with selection seed 42: 20 development tasks and 100 held-out tasks. The separate training manifest has 40 tasks from seeds 1000–1039. The currently staged manifests have 20, 100, and 40 distinct task IDs and distinct initial-observation hashes, with no identity or exact-content overlap across partitions. Inspected pilot seeds 124, 135, 138, and 156 are development tasks. The known eligible reserved set is therefore **100 tasks**, not an extrapolation to arbitrary TextCraft seed IDs. The audit cannot prove that no other task with similar recipe structure exists; it protects exact task identity and displayed-task content.

The exact held-out manifest hash is `5c8dbc6caa2db221cc6ed4ce3208d9c53157af515563895a927957c3ac8349a0`. It was an exclusion when the trajectory corpus was built. The workflow uses its original JSON without regenerating seeds or changing its partition or fields. A changed manifest would fail the M2 corpus exclusion check. The local transfer bundle currently contains no run that touches a held-out task. The Locust freeze command repeats this scan against the cluster's `results/` directory and refuses to freeze if one is found. It cannot audit private files outside the configured data root.
If Locust contains an unreported held-out run, stop rather than select the remaining tasks after inspecting its outcomes. A defensible alternative would predefine and validate a new seed candidate pool in a separate protocol, exclude every previously exposed identity and observation hash, and rebuild or explicitly extend the memory exclusion proof before evaluating. No such replacement pool is assumed here.

The existing runner could execute an arbitrary manifest, but its comparison required the development partition. This workflow extends the strict comparator to full held-out runs, and adds a three-condition report. It preserves the existing development pilot and memory artifacts. It refuses a second held-out run for the same condition under this lock, so infrastructure failures should be inspected before any deliberate retry protocol is adopted.
The runner has no resume mechanism. Estimate a 100-task wall time from the three development smoke logs before launching. If any condition is likely to exceed the partition's maximum wall time, stop before exposing held-out outcomes and implement a predeclared chunk-and-aggregate or checkpoint protocol first.

## Locked protocol

1. Keep the existing Python 3.12 virtual environment and CUDA-12.8-compatible PyTorch. Do not run the install script. Use one Git commit with no tracked local code edits; old untracked Slurm logs may remain.
2. Check the existing development manifest and frozen baseline on CPU. Run three **development** smoke jobs, one per condition, and inspect their logs and episode traces. These do not consume held-out outcomes.
3. On a CPU Slurm job, run `freeze-heldout`. It checks all 100 held-out task replays against TextCraft's installed recipe catalog without taking an action; verifies training/development exclusion, exact original manifest hash, trajectory-corpus exclusion and corpus/index hashes; resolves the already cached Qwen model commit without downloading; and records Python/package versions, code commit, shared agent settings, and the recipe-order snapshot. It writes new, owner-only `heldout_evaluation/manifest.json` and `lock.json` under `AGENT_SUPPORT_DATA`. Neither is overwritten.
4. Submit baseline, `m2_trajectory_v1`, and `reconsideration_only_v1` in sequence. All three use the same frozen 100-task manifest, Qwen model revision, seed 42, NF4, greedy decoding, prompts, history window 8, `max_actions=30`, `max_decisions=30`, and `max_new_tokens=100`. M2 uses the existing corpus/index and retrieval settings; the other conditions load no memory. The first model proposal and environment adapter stay unchanged.
5. After all three complete, run `compare-heldout` on CPU. It refuses incomplete or duplicate runs, changed snapshots, package/Python/code/model revisions, agent settings, or task lists. It writes `heldout_evaluation/analysis.json` without overwriting a different report.

The report contains each condition's successes and task-level outcomes, three paired helped/hurt comparisons, model calls, actual tokens, inference and retrieval time, parser errors, and action counts. It adds 95% Wilson intervals for each success rate and a fixed-seed 10,000-replicate **paired task bootstrap** interval for each success-rate difference. The intervals describe variation across tasks resembling this reserved set; this seed-defined set is not a random sample of all possible TextCraft tasks. The primary research contrast is M2 versus reconsideration. Baseline comparisons are secondary. This design matches M2 and reconsideration's potential second-call schedule after valid proposals, not prompt length or token count.

## Locust commands

Use the existing working environment. The data root below matches the original transfer layout; substitute your actual `AGENT_SUPPORT_DATA` if it was moved to `/shared_data0/`. Run `git status --short` first and reconcile any local edit, including an older local `locust_job.sh` fix, before updating. The published script already uses `SLURM_SUBMIT_DIR`. Do not discard unrelated local changes.

```bash
ssh evanpan@locust-login.seas.upenn.edu
cd "$HOME/AgenticSDS"
git status --short
git pull --ff-only
export AGENT_SUPPORT_DATA="$HOME/agent_support_data"
export HF_HOME="$AGENT_SUPPORT_DATA/hf_cache"
export VENV_PATH="$PWD/.venv"
export PYTHONHASHSEED=0
mkdir -p "$AGENT_SUPPORT_DATA/logs"
chmod 700 "$AGENT_SUPPORT_DATA/logs"
sinfo -o '%P %G %l'
"$VENV_PATH/bin/python" -m agent_support.cluster_workflow preflight --benchmark textcraft
```

Choose partition names available to your account. Check the reported time limit and adjust the wall times below if needed. A four-CPU, 32 GB, one-GPU batch job holds its GPU only while running. The smoke jobs use the already explored development tasks; inspect all three outputs before freezing.
The older generic `preflight` may report that the frozen Colab baseline (Python 3.13) is incompatible with Locust (Python 3.12). That is expected and is why all three held-out conditions run anew in the same locked Locust environment. It must still find the transferred manifests and training run.

```bash
GPU_PARTITION='YOUR_GPU_PARTITION'
CPU_PARTITION='YOUR_CPU_PARTITION'
S1=$(sbatch --parsable --partition="$GPU_PARTITION" --gpus=1 --cpus-per-task=4 --mem=32G --time=01:00:00 --output="$AGENT_SUPPORT_DATA/logs/smoke-baseline-%j.out" scripts/locust_job.sh smoke --benchmark textcraft --condition baseline)
S2=$(sbatch --parsable --dependency="afterok:$S1" --partition="$GPU_PARTITION" --gpus=1 --cpus-per-task=4 --mem=32G --time=01:00:00 --output="$AGENT_SUPPORT_DATA/logs/smoke-m2-%j.out" scripts/locust_job.sh smoke --benchmark textcraft --condition m2_trajectory_v1)
S3=$(sbatch --parsable --dependency="afterok:$S2" --partition="$GPU_PARTITION" --gpus=1 --cpus-per-task=4 --mem=32G --time=01:00:00 --output="$AGENT_SUPPORT_DATA/logs/smoke-reconsideration-%j.out" scripts/locust_job.sh smoke --benchmark textcraft --condition reconsideration_only_v1)
squeue -u "$USER"
```

After the three smoke jobs complete successfully, inspect `$AGENT_SUPPORT_DATA/logs/smoke-*.out` and their three-task `results/` directories. Then freeze the full held-out protocol on a CPU node. This job uses no GPU and does not reinstall packages or rebuild memory.

```bash
FREEZE_JOB=$(sbatch --parsable --partition="$CPU_PARTITION" --cpus-per-task=2 --mem=8G --time=01:00:00 --output="$AGENT_SUPPORT_DATA/logs/freeze-heldout-%j.out" scripts/locust_job.sh freeze-heldout --benchmark textcraft)
squeue -j "$FREEZE_JOB"
```

Check that the freeze log reports `eligible_count: 100`, `replay_checked_task_count: 100`, no prior held-out runs, and a model revision. Inspect the saved lock and manifest before submitting evaluation. Then submit the three full jobs; each dependency waits for the previous job to succeed, and each GPU is released when its job exits. A 16-hour request is a conservative upper bound; use a shorter limit if the smoke throughput supports it, and do not exceed the partition limit.

```bash
BASE_JOB=$(sbatch --parsable --partition="$GPU_PARTITION" --gpus=1 --cpus-per-task=4 --mem=32G --time=16:00:00 --output="$AGENT_SUPPORT_DATA/logs/heldout-baseline-%j.out" scripts/locust_job.sh evaluate-heldout --benchmark textcraft --condition baseline)
M2_JOB=$(sbatch --parsable --dependency="afterok:$BASE_JOB" --partition="$GPU_PARTITION" --gpus=1 --cpus-per-task=4 --mem=32G --time=16:00:00 --output="$AGENT_SUPPORT_DATA/logs/heldout-m2-%j.out" scripts/locust_job.sh evaluate-heldout --benchmark textcraft --condition m2_trajectory_v1)
D_JOB=$(sbatch --parsable --dependency="afterok:$M2_JOB" --partition="$GPU_PARTITION" --gpus=1 --cpus-per-task=4 --mem=32G --time=16:00:00 --output="$AGENT_SUPPORT_DATA/logs/heldout-reconsideration-%j.out" scripts/locust_job.sh evaluate-heldout --benchmark textcraft --condition reconsideration_only_v1)
squeue -u "$USER"
```

After all three runs finish, use the existing Python environment on the login node for the lightweight report (or submit it to a CPU partition if login-node policy requires):

```bash
"$VENV_PATH/bin/python" -m agent_support.cluster_workflow compare-heldout --benchmark textcraft
```

The command automatically finds the unique full run for each condition and saves the report under `$AGENT_SUPPORT_DATA/heldout_evaluation/analysis.json`. Download that directory and the three result directories to your Mac after verifying them. If a job fails, preserve its partial run and log; the workflow will not silently retry or replace it.

## Interpretation

The manifest fixes task composition and identity across conditions, while the lock fixes the code, model snapshot, package versions, recipes, seed, and memory hashes. Greedy decoding and a fixed seed reduce sampling variation; they do not guarantee bitwise identical GPU inference. TextCraft task generation historically varied with unordered collections, so this runner replays each manifest's displayed task instead of regenerating from seed. The adapter starts with an empty inventory and fixed goal, but this protocol does not claim that every environment transition or CUDA operation is provably deterministic. Repeating the same 100 tasks without a predefined new source of randomness mostly repeats the same evidence. If stochastic transitions are demonstrated, predefine multiple paired replicates and analyze them with task as the unit of clustering; do not count task-by-repeat outcomes as independent new tasks. Do not adjust prompts, retrieval, support policies, budgets, or task selection after seeing held-out outcomes.
