# ADaPT dynamic replanning for TextCraft

This development condition adapts the published [ADaPT paper](https://aclanthology.org/2024.findings-naacl.264/) and the authors' [TextCraft implementation](https://github.com/archiki/ADaPT/blob/ecdc4ab0030b4be9be122622d8ea78f8c59c44c4/run_textcraft.py) at revision `ecdc4ab0030b4be9be122622d8ea78f8c59c44c4`. The upstream planner and executor demonstrations are retained in `agent_support/adapt_prompts.py` under the [upstream MIT license](adapt_upstream_LICENSE). This is an adaptation of the mechanism to AgenticSDS and Qwen2.5-7B-Instruct; no published ADaPT result has been reproduced here.

The historical 100-task held-out lock, runs, and frozen `aba4da7` checkout are outside this development worktree. ADaPT is rejected for held-out comparisons.

## Mechanism and source mapping

| Authors' `run_textcraft.py` | This condition |
| --- | --- |
| `plan_and_run` attempts a task with `textcraft_run` before planning | `ADaPTController._execute` calls `_executor_attempt` before `_plan` |
| `plan_prompt` and `atomic_exec_prompt` with TextCraft demonstrations | `adapt_prompts.py`, wrapped by `QwenPlanner` and `QwenExecutor` |
| `plan_to_args` and AND/OR execution order | `parse_plan`, `PlanNode`, and `_run_plan` |
| Recursive `plan_and_run`, `max_depth=4` | `_execute`, configurable `adapt_max_depth=4` |
| Same `env` object across subtasks; inventory after success | One `TextCraftEpisode` for the tree; inventory feedback is propagated |
| `max_runs=40` with a `range(1, max_runs)` loop | `adapt_max_executor_steps=39` per direct attempt |
| The executor's `think: Task Completed!` self-report | The claim is logged; root success requires environment reward, and subtask success requires a matching inventory count |

The controller is separate from the per-turn `FixedConditionPolicy`: ADaPT controls multiple executor and planner calls and can recursively call itself after failure. Its run metadata uses `support_policy_id=adapt_controller_v1`; it does not invoke retrieval or M2. Planner and executor are separate interfaces sharing the local Qwen wrapper for now.

An example tree from the CPU tests:

```text
craft 1 plank                          direct attempt: failed
└─ plan: Step 1 AND Step 2
   ├─ fetch 1 wood                    get 1 wood; inventory verifies [wood] (1)
   └─ craft 1 plank using 1 wood      craft action; TextCraft gives reward 1
```

The trace stores the original task, direct outcome and failure reason, raw and parsed plans, depths, each model output and token count, environment actions and observations, subtask outcomes, parser errors, and final environment-verified reward. `summary.json` includes planner and executor calls, decomposition and plan-parse counts, false completion claims, total tokens, action counts, and inference time. `model_calls` includes both planner and executor; `support_calls` is the planner count, not retrieval. Retrieval time is zero.

## Deviations and limits

- The authors call GPT completion/chat endpoints. This version uses the existing NF4 Qwen2.5-7B-Instruct chat wrapper with greedy decoding (`do_sample=False`). Planner and executor have separate token ceilings, 800 and 150. The wrapper takes the first generated nonempty line for an executor action because it has no newline stop sequence. A short cue requests the exact displayed TextCraft recipe and `get N ITEM` syntax.
- The ADaPT executor accepts the upstream TextCraft syntax `craft ITEM using N INGREDIENT` without an output count. TextCraft uses the recipe's actual output quantity. The shared baseline action parser is unchanged. A smoke trace review found this compatibility gap; none of its three episodes used the omitted-count syntax, so the fix does not explain their observed outcomes.
- The authors' final TextCraft executor uses patience 8 and accepts a self-reported completion heuristic. This version keeps patience 8 but verifies inventory for intermediate goals and environment reward for the original goal. It counts verification `inventory` commands against the environment-action budget. A false claim triggers decomposition if depth and budgets allow. A completed plan without root reward remains a failed episode.
- Upstream parsing falls back to treating malformed plans as a sequential AND list. Here malformed or ambiguous Step/Execution Order text is retained in the trace and fails that decomposition; no invented plan replaces it. The parser accepts parenthesized nested AND/OR groups, rejects duplicate/unknown Step references, and has a 12-step size guard.
- Global environment-action, model-call, and planner-call limits and an ancestor-task loop guard prevent unbounded work. Upstream has per-attempt `max_runs` and `max_depth` but no equivalent global ceilings. The controller uses one persistent TextCraft environment; it propagates observed inventory even after a failed OR branch. It does not rewind a failed branch's side effects.
- The existing TextCraft adapter replays a frozen displayed task and recipe catalog from a manifest. ADaPT uses that same adapter. The development comparison uses a **new shared 120-action, 120-model-call budget**, not the historical 30-action baseline budget. It must be analyzed as a new development experiment, not compared to the old pilot or held-out outcomes.
- A cached Qwen snapshot revision is frozen into both development configs before running. `prepare` does not download model weights. The source branch must be committed and clean, and the development manifest must match its tracked snapshot. Greedy decoding does not imply bitwise reproducibility across GPUs, PyTorch versions, or process restarts.

## CPU verification and local preparation

Run from the ADaPT branch checkout:

```bash
python3 -m pytest -q
```

The two configs are `configs/textcraft_adapt_dynamic_replanning.json` and `configs/textcraft_adapt_control_baseline.json`. The control retains the existing baseline prompt and Qwen wrapper; it has the same model, seed, history window, action and model-call ceiling, and executor token ceiling as ADaPT. The ADaPT planner has its own larger token ceiling, so this is not token matched. Prepare writes immutable revision-pinned copies and a protocol hash under `$AGENT_SUPPORT_DATA/adapt_development/`:

```bash
export AGENT_SUPPORT_DATA="$HOME/agent_support_data"
export HF_HOME="$AGENT_SUPPORT_DATA/hf_cache"
export VENV_PATH="$PWD/.venv"
export PYTHONHASHSEED=0
"$VENV_PATH/bin/python" -m agent_support.adapt_development prepare --data-root "$AGENT_SUPPORT_DATA"
"$VENV_PATH/bin/python" -m agent_support.adapt_development preflight --data-root "$AGENT_SUPPORT_DATA"
```

These are CPU-only checks. If `prepare` reports no cached Qwen snapshot, populate the existing Hugging Face cache using the cluster's approved transfer method; do not reinstall PyTorch or replace the working Python 3.12 virtual environment. If a development artifact differs, inspect it and use a new data directory for a new protocol rather than overwriting it.

## Locust smoke test

After the ADaPT branch has been pushed by its owner, use a **separate clone or worktree** on Locust so the frozen `aba4da7` checkout remains intact. On the login node, connect through the Penn VPN if off campus, then:

```bash
git clone --branch adapt_dynamic_replanning https://github.com/nave-nap/AgenticSDS.git "$HOME/AgenticSDS-ADaPT"
cd "$HOME/AgenticSDS-ADaPT"
git status --short
export AGENT_SUPPORT_DATA="$HOME/agent_support_data"
export HF_HOME="$AGENT_SUPPORT_DATA/hf_cache"
export VENV_PATH="$HOME/AgenticSDS/.venv"
export PYTHONHASHSEED=0
"$VENV_PATH/bin/python" -m agent_support.adapt_development prepare --data-root "$AGENT_SUPPORT_DATA"
"$VENV_PATH/bin/python" -m agent_support.adapt_development preflight --data-root "$AGENT_SUPPORT_DATA"
sinfo -o '%P %G %l'
```

Use the existing Python 3.12 virtual environment and CUDA-12.8-compatible PyTorch. The `SLURM_SUBMIT_DIR` behavior in `scripts/locust_adapt_dev.sh` makes the job use the ADaPT checkout, even though Slurm spools the script elsewhere. The following is a **command to submit**, not a job already run:

```bash
sbatch --account=hassani --partition=standby --qos=standby --gres=gpu:1 \
  --cpus-per-task=4 --mem=32G --time=01:00:00 --no-requeue \
  scripts/locust_adapt_dev.sh smoke adapt_dynamic_replanning_v1
```

It runs the first three frozen development tasks and saves a new run under `agent_support_data/adapt_development/results/`. Inspect its `run.json`, `summary.json`, episode JSON files, and `slurm-JOBID.out`. The development wrapper creates a unique run directory and never overwrites an episode. `--no-requeue` prevents the preemptible standby partition from restarting into a partial run. A preempted job remains an incomplete artifact; submit a **new** smoke job after inspecting it. The specialized held-out M2 continuation tool cannot safely resume this different recursive controller without a separate audited recovery design. No GPU job was launched while developing this condition.

If the smoke trace and wall time look reasonable, run the baseline control and ADaPT on the same **full 20-task development manifest**. Use separate jobs, preferably a nonpreemptible partition if your allocation provides one. On standby, keep `--no-requeue` and treat an interrupted job as incomplete:

```bash
sbatch --account=hassani --partition=standby --qos=standby --gres=gpu:1 \
  --cpus-per-task=4 --mem=32G --time=04:00:00 --no-requeue \
  scripts/locust_adapt_dev.sh full baseline
```

After the baseline job completes and its full 20-task `run.json` reports `status: complete`, submit ADaPT:

```bash
sbatch --account=hassani --partition=standby --qos=standby --gres=gpu:1 \
  --cpus-per-task=4 --mem=32G --time=04:00:00 --no-requeue \
  scripts/locust_adapt_dev.sh full adapt_dynamic_replanning_v1
```

On the login node, after both are complete, compare the **explicit** run directories (not the three-task smoke run):

```bash
"$VENV_PATH/bin/python" -m agent_support.cluster_workflow compare \
  --benchmark textcraft \
  --left-run "$AGENT_SUPPORT_DATA/adapt_development/results/EXACT_BASELINE_RUN_DIR" \
  --right-run "$AGENT_SUPPORT_DATA/adapt_development/results/EXACT_ADAPT_RUN_DIR"
```

The paired report checks the full identical manifest, shared settings, package versions, recipe snapshot, and code commit. It reports helped/hurt tasks, success counts, model calls, tokens, inference time, parser errors, and action counts. Keep the original episode traces for the recursion tree and planning-specific metrics. This 20-task development comparison is exploratory; it is not a new held-out result.
