# TextCraft autonomous support evaluation protocol

## Question and comparison boundary

Does a controller that chooses and composes support operations improve task
success at a given compute cost? The primary unit is a **task** in the frozen
20-task TextCraft development manifest (canonical SHA-256
`4ec57f36416ffe7f97b004b67d107e94750fa20b889d95f580690f4dfca4900e`).
All new primary arms use the same Qwen2.5-7B-Instruct task proposer, revision
`a09a35458c702b33eeacc393d103063234e8bc28`, NF4, greedy decoding,
seed 42, eight prior interactions, 100 proposal tokens, and 30 environment
actions/decisions. The TextCraft recipe-order snapshot and current runner are
shared. The controller/support model is GPT-OSS-20B revision
`6cee5e81ee83917806bbde320786a8fb61efebee`, packed MXFP4, low reasoning
effort, 1,024 output-token ceiling, seed 42. The M2-200 corpus and index are
fixed by the content digests in the evaluation configs.

| Primary arm | Selector and support | Intended role |
| --- | --- | --- |
| `qwen_standalone` | No support | Same task-agent control |
| `qwen_reconsideration` | Existing Qwen matched second-call condition | Extra task-agent reasoning without retrieval |
| `qwen_m2_200` | Existing always-on M2 trajectory retrieval and Qwen reconsideration, frozen 200 trajectories | Fixed memory pipeline |
| `random_support` | Preseeded probability 0.5; at most one uniformly selected enabled capability | Simple support selector |
| `autonomous` | Prompted GPT-OSS controller with sequential supports | Research condition |
| `fixed_reasoning_2` | Two GPT-OSS reasoning calls per valid proposal until its 30-additional-call cap, fixed request objective | Approximate extra-call control without adaptive support selection |

The fixed reasoning control shares the autonomous arm's 30-additional-call
ceiling but follows a fixed two-call schedule until that ceiling is reached.
It does not guarantee matching realized calls, tokens, inference seconds, or
decision counts after trajectories diverge. Compare those actual costs
explicitly. The random controller uses no
LLM to select support; this lower-cost selector helps expose the cost of
controller reasoning. No support order is hardcoded in the autonomous policy.

The completed GPT-OSS standalone (`slurm_152866`), matched reconsideration
(`slurm_153283`), and M2-11 (`slurm_153974`) development runs are preserved.
All three used the same frozen manifest and GPT-OSS task agent and each solved
14/20. They are **historical descriptive references**, because the new primary
arms use Qwen as task agent, the new memory arms use M2-200 rather than M2-11,
and the code/runtime differ. Do not pool or label their differences as an
isolated controller effect. No existing completed run is rerun merely to
populate this report; the contemporaneous Qwen controls are necessary for a
matched primary cohort.

## Frozen stages and gates

1. Commit this protocol, configs, and code. The GPU runner refuses a dirty
   checkout and records the commit, selected manifest, config, package
   versions, recipe snapshot, memory digests, budgets, and seeds.
2. On CPU, run `preflight` for each arm. It validates the 20-task manifest,
   memory exclusions, 200-trajectory count, frozen index structure, and
   configured digests without generating new memory.
3. Run **one-task smoke** for each new arm using the first frozen task. Check
   model startup, parser behavior, support accounting, trace completeness,
   retrieval provenance, final action validation, and resource use. A failed
   smoke is an infrastructure failure; preserve its artifacts.
4. Run **five-task pilot** for each arm after its successful smoke. Inspect
   support sequences, no-support choices, repeated/empty retrievals, model
   generation failures, budget stops, token and wall costs, and task-agent
   trajectories. The pilot is for operational validation, not prompt or
   hyperparameter selection. Any policy change creates a new version and
   requires a new smoke/pilot.
5. Write a reviewed gate for each arm after examining its pilot trace. A full
   run requires that gate and identical code commit, config, package versions,
   Python version, recipe snapshot, and frozen manifest.
6. Run the **full 20 tasks** only after all relevant gates pass. Do not
   automatically submit jobs, retry failures, or overwrite completed runs.
   The current runner checkpoints completed episodes atomically but cannot
   resume an interrupted support loop within an episode.

Use `configs/autonomous_evaluation/*.json`. `agent_support.autonomous_evaluation`
resolves the M2 path for the legacy runner and stores the resolved config in
each run. The autonomous and random configs pin the 200-memory digests.
Frozen artifacts are currently at
`~/agent_support_data/m2_200/{corpus_200,index_200}.json`.

## Metrics and prespecified analysis

The primary descriptive endpoint is task success out of 20. For each arm,
report task successes/failures, Wilson interval, environment actions, failed
actions, all model calls, input/output tokens, inference seconds, parser and
generation failures. Separate proposal, controller, support, and embedding
costs where logs permit. Report controller decision calls/tokens/wall time,
support execution wall time, retrieval time, support operation types and
counts, complete ordered sequences at each decision, sequence lengths,
operation combinations, no-support frequency, composed decisions, action
replacements, and budget terminations.
The legacy Qwen reconsideration/M2 runner records support-model inference and
retrieval time but not full per-operation wall time; that field is `null` for
those arms rather than an invented zero.

Compute paired task helped/hurt sets and success-rate differences for the
autonomous arm against all five controls. Report exact two-sided McNemar
discordance p-values, Holm-adjusted values across those five contrasts, and
fixed-seed paired-task bootstrap intervals. With 20 previously explored
development tasks, these are exploratory uncertainty summaries, not a
confirmatory generalization claim. Report realized model calls, tokens, and
inference time beside every success comparison; no equal-compute conclusion
follows from equal caps alone. Compare success against cost across arms and
inspect task-level tradeoffs.

The sequence analysis reports procedure frequencies overall and split by
successful/failed tasks, capability use, repeated operations, no-support
decisions, and task-level support-count versus success summaries. These are
**associations**: hard tasks may trigger more support, support can alter
subsequent states, and the controller observes feedback that controls do not
counterfactually share. Per-call causal claims require randomized intervention
or reproducible counterfactual replay from identical states. This runner does
not provide verified counterfactual replay. Do not infer that an individual
retrieval, action replacement, or accepted action caused success.

## Resource and command plan

The new GPU job script is `scripts/autonomous_support_eval.sbatch`. Defaults:
one GPU, six CPUs, 64 GB RAM, two-hour wall time. The command may override
time after checking the available partition and pilot throughput. It uses
the existing Qwen virtual environment plus one persistent GPT-OSS worker for
the three GPT-OSS support arms. Keep runs and Slurm logs outside the repository
under `$AGENT_SUPPORT_DATA/autonomous_support_evaluation/`.

```bash
cd "$HOME/AgenticSDS"
export AGENT_SUPPORT_DATA="$HOME/agent_support_data"
export VENV_PATH="$PWD/.venv"
export AGENT_SUPPORT_GPTOSS_PYTHON="$HOME/gptoss_env/bin/python"  # set to the pinned installed environment
export AGENT_SUPPORT_GPTOSS_HF_HOME="$AGENT_SUPPORT_DATA/hf_cache"
export PYTHONHASHSEED=0
mkdir -p "$AGENT_SUPPORT_DATA/autonomous_support_evaluation/logs"
for ARM in qwen_standalone qwen_reconsideration qwen_m2_200 random_support autonomous fixed_reasoning_2; do
  "$VENV_PATH/bin/python" -m agent_support.autonomous_evaluation preflight --arm "$ARM"
done
```

Review the preflight output, cluster partition limits, and both Python
environments. The following are **submission commands for review**, not jobs
submitted by this work:

```bash
GPU_PARTITION='YOUR_GPU_PARTITION'
ARM='autonomous'  # repeat per arm; keep each arm's paths separate
EVAL_ARM="$ARM" EVAL_STAGE=smoke sbatch --partition="$GPU_PARTITION" --time=01:00:00 \
  --output="$AGENT_SUPPORT_DATA/autonomous_support_evaluation/logs/${ARM}-smoke-%j.out" \
  scripts/autonomous_support_eval.sbatch
```

After reading the smoke log and its episode JSON, set its exact run path.
`run` prints that path. A pilot uses the first five tasks and checks the
smoke's config and commit:

```bash
SMOKE_RUN='/absolute/path/to/one-task/run'
EVAL_ARM="$ARM" EVAL_STAGE=pilot EVAL_SMOKE_RUN="$SMOKE_RUN" \
  sbatch --partition="$GPU_PARTITION" --time=04:00:00 \
  --output="$AGENT_SUPPORT_DATA/autonomous_support_evaluation/logs/${ARM}-pilot-%j.out" \
  scripts/autonomous_support_eval.sbatch
PILOT_RUN='/absolute/path/to/five-task/run'
"$VENV_PATH/bin/python" -m agent_support.support_sequence_analysis \
  --run "$PILOT_RUN" --output "$AGENT_SUPPORT_DATA/autonomous_support_evaluation/${ARM}-pilot-sequences.json"
```

Inspect the pilot sequence report and raw episode traces. Then explicitly
write its gate and review it:

```bash
GATE="$AGENT_SUPPORT_DATA/autonomous_support_evaluation/gates/${ARM}.json"
"$VENV_PATH/bin/python" -m agent_support.autonomous_evaluation gate \
  --arm "$ARM" --smoke-run "$SMOKE_RUN" --pilot-run "$PILOT_RUN" \
  --output "$GATE" --acknowledge-traces-reviewed
cat "$GATE"
EVAL_ARM="$ARM" EVAL_STAGE=full EVAL_GATE_FILE="$GATE" \
  sbatch --partition="$GPU_PARTITION" --time=16:00:00 \
  --output="$AGENT_SUPPORT_DATA/autonomous_support_evaluation/logs/${ARM}-full-%j.out" \
  scripts/autonomous_support_eval.sbatch
```

Only use the full-run command after all six pilot reviews and gate files
exist. Choose a wall time supported by the partition and pilot throughput.
For analysis, supply each completed run directory explicitly:

```bash
"$VENV_PATH/bin/python" -m agent_support.support_sequence_analysis \
  --run "$AUTONOMOUS_RUN" --output "$AGENT_SUPPORT_DATA/autonomous_support_evaluation/autonomous-sequences.json"
"$VENV_PATH/bin/python" -m agent_support.autonomous_evaluation_compare \
  --run "qwen_standalone=$STANDALONE_RUN" \
  --run "qwen_reconsideration=$RECONSIDERATION_RUN" \
  --run "qwen_m2_200=$M2_RUN" --run "random_support=$RANDOM_RUN" \
  --run "autonomous=$AUTONOMOUS_RUN" --run "fixed_reasoning_2=$FIXED_RUN" \
  --output "$AGENT_SUPPORT_DATA/autonomous_support_evaluation/full-comparison.json"
```

The comparison command rejects incomplete, truncated, mismatched-task,
mismatched-agent, mismatched-code/runtime, or mismatched-memory runs. Optional
`--historical LABEL=PATH` arguments add preserved GPT-OSS run summaries in a
separate descriptive section. They never enter the matched paired contrasts.
