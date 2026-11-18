# Autonomous TextCraft smoke reliability v2

This is a one-task diagnostic configuration. It uses the frozen development
manifest and the same task agent, controller model, capabilities, and budgets
as `autonomous.json`. Only the controller ID changes to
`prompted_gpt_oss_v2`. The original job 157532 and its outputs are not
modified. Do not compare v2 results with v1 without recording the code commit
and config digest.

## Audit of job 157532

The downloaded, read-only copy at
`/Users/evanpan/Downloads/autonomous_157532/textcraft_20261118T161025Z_b50935d6/`
contains the episode, run/config/summary snapshots, and worker log. The
episode confirms 30 Qwen proposals, 30 GPT-OSS controller calls, 12 executed
state inspections, 27 failed environment actions, and no action replacements.
The original run used commit `e43b51b418254b85993df4e9bfac411b629c911a`,
the frozen first development task `textcraft:seed:51`, and the pinned M2-200
digests. The worker initialized packed GPT-OSS on an NVIDIA L40S; the job
completed without a worker crash. No retrieval or additional reasoning
operation was invoked, so their runtime behavior was not tested.

| Controller-call outcome | Calls | Input tokens | Output tokens | Inference seconds |
| --- | ---: | ---: | ---: | ---: |
| `missing_final` | 6 | 8,381 | 591 | 25.592 |
| Other malformed JSON/schema | 2 | 2,568 | 334 | 12.722 |
| Invalid support arguments | 6 | 7,470 | 480 | 18.728 |
| Executed support returned `empty` | 11 | 14,837 | 1,316 | 51.359 |
| Executed support returned `ok` | 1 | 1,069 | 149 | 5.579 |
| Operation-budget block | 1 | 1,264 | 72 | 2.896 |
| Execute decision | 3 | 4,045 | 269 | 10.588 |

These categories partition all 30 controller calls. The controller used
42,845 of 120,000 additional tokens and all 30 additional model calls by
decision 18. Decisions 19–30 received no controller inference because the
model-call budget was exhausted. The 11 empty support requests alone took
51.359 seconds of controller inference; the operations themselves used no
model calls and reported zero support inference time.
The older sequence analysis labels 23 decisions as having "no support"
because no operation executed; only one was an explicit controller choice to
execute immediately. The others include parse failures, invalid requests,
and budget fallback. V2 analysis adds `controller_chose_no_support_decisions`
without changing the older descriptive count.

The code confirms that `parse_harmony_completion` returns an empty `text`
whenever the generated token sequence lacks exactly one complete `final`
channel followed by `<|return|>`. Generated tokens can still be nonzero.
For `missing_final`, the generated sequence is shorter than its limit and
contains no final marker. The six affected generations used 44–191 output
tokens, versus a 1,024-token limit: length truncation is excluded for these
events. The old controller logged only the parsed `text`, so its trace cannot
distinguish reasoning-only text, a return without a final channel, or another
short malformed sequence. The worker log contains only initialization output,
not those generations. V2 records the full generated
completion, prompt tail, last generated token ID, stop/limit flags, and
reasoning/final token counts. It does not promote reasoning-channel text to
an executable decision or retry without a measured cause.
The [pinned GPT-OSS chat template](https://huggingface.co/openai/gpt-oss-20b/blob/6cee5e81ee83917806bbde320786a8fb61efebee/chat_template.jinja)
ends the generation prompt at the assistant header, without pre-filling a
final channel. The next run still needs to verify the locally rendered prompt
tail and generated token sequence in the pinned environment.

The two other malformed outputs were a request missing its required `reason`
field and JSON with an extra closing brace. All six invalid requests named
the available `inspect_state_v1` operation: five combined several scope
names into one invalid scope value; one omitted required `limit`. This is a
prompt/schema presentation problem, not a capability-registration failure.
V2 presents machine-readable single-value enums, required fields, and bounds
for all four capabilities. The executor continues to reject invalid arguments.

All 19 attempted support requests targeted `inspect_state_v1`. Of the 12
executed requests, 11 asked for inventory only and returned `empty` because
the task agent never executed `inventory`. The one `all` request returned
`ok` with recipes and craft feedback but explicitly had
`inventory_observed:false`; it did not satisfy its inventory objective. At
decision 8, four empty inventory inspections occurred in one support loop.
The controller prompt already included recent action/feedback history and
the four registered capability names. The old context exposed support results
within that loop but did not retain requests between environment decisions.
The request objectives repeatedly ask to discover current inventory, which
`inspect_state_v1` cannot query. This supports a capability-semantics
misunderstanding; the trace does not establish that retrieval or extra
reasoning would have improved the action.
Repeating after an empty result was partly model behavior; the absence of a
prior-request ledger and the
description's ambiguous comma-separated scope list were implementation
weaknesses. V2 supplies a bounded ledger, explains the unavailable inventory
field, and blocks another valid inventory-only request after an empty one
until an `inventory` environment action has been observed, even when the
request changes `limit` or `query`. It also blocks identical empty requests
with unchanged visible evidence. The guard adds no hidden state access or
mandatory support sequence.

The existing action-revision path is `execute.action` or a valid
`reason_over_artifacts_v1` candidate, followed by the benchmark parser and
environment execution. The trace confirms that GPT-OSS never emitted a
different action; its one explicit action at decision 18 was identical to
Qwen's proposal. Qwen proposed `craft 8 cyan dye ...` 13 times while the
visible recipe listed `craft 2 cyan dye ...`, and each such action failed.
That mismatch is evidence of poor action selection, although changing the
quantity alone would not establish that ingredients were available. CPU tests
verify that a controller-selected `craft 2 ...` action executes through the
validation path. The benchmark parser checks command syntax, while the
environment determines whether the action succeeds. The existing episode-level
`parser_errors` counts invalid task actions; v2 additionally reports
`controller_parse_errors`, `controller_generation_failures`, and
`invalid_support_requests`. Model-call and token budgets remain unchanged.

## Mac review and transfer commands

Run from `/Users/evanpan/AgenticSDS` after reviewing the diff. The untracked
`reports/complete_research_summary/` is unrelated and must remain unstaged.

```sh
python3 -m pytest -q tests
bash -n scripts/autonomous_support_eval.sbatch
git diff --check
git switch -c autonomous-smoke-v2
git add -- agent_support/autonomous_evaluation.py \
  agent_support/autonomous_evaluation_compare.py agent_support/autonomous_runner.py \
  agent_support/autonomous_operations.py agent_support/autonomous_support.py \
  agent_support/common.py \
  agent_support/gpt_oss_model.py agent_support/learned_support_gate.py \
  agent_support/llm_support_controller.py agent_support/runner.py \
  agent_support/support_sequence_analysis.py \
  configs/autonomous_evaluation/autonomous_v2.json \
  scripts/autonomous_support_eval.sbatch tests/test_autonomous_evaluation.py \
  tests/test_llm_support_controller.py \
  reports/autonomous_support_controller/smoke_reliability_v2.md
git diff --cached --check
git diff --cached --name-only
git commit -m "Diagnose and bound autonomous TextCraft smoke v2"
git push -u origin autonomous-smoke-v2
```

## Locust deployment and CPU preflight

Run these in an existing Locust AgenticSDS checkout. Use the working Qwen
and GPT-OSS environments from job 157532; their locations are not known here.
The new worktree keeps the old checkout and M2 outputs untouched.

```sh
EXISTING_REPO="$(git rev-parse --show-toplevel)"
git fetch origin autonomous-smoke-v2
DEPLOY_REPO="$(dirname "$EXISTING_REPO")/AgenticSDS-autonomous-smoke-v2"
test ! -e "$DEPLOY_REPO"
git worktree add --detach "$DEPLOY_REPO" origin/autonomous-smoke-v2
cd "$DEPLOY_REPO"
git status --porcelain
git rev-parse HEAD
find /shared_data0/evanpan/agent_support -maxdepth 6 -path '*/bin/python*' -print
env | sort | grep -E '^(AGENT_SUPPORT|HF_HOME|VENV_PATH|CONDA_PREFIX)=' || true
sacct -j 157532 -X -o Partition%30,ReqTRES%60,State
```

After identifying the executable and cache paths from the successful job,
set them and verify the frozen artifacts. The artifact root and output root
below are supplied by the original run; the environment and cache paths
must be discovered on Locust.

```sh
export AGENT_SUPPORT_DATA=/shared_data0/evanpan/agent_support
export AGENT_SUPPORT_M2_200_ARTIFACT_DIR=/shared_data0/evanpan/agent_support/gptoss_memory_v1/memory_scaling_v1/artifacts
export VENV_PATH="<existing-Qwen-environment-directory>"
export AGENT_SUPPORT_GPTOSS_PYTHON="<existing-pinned-GPT-OSS-python>"
export HF_HOME="<existing-Qwen-model-cache-root>"
export AGENT_SUPPORT_GPTOSS_HF_HOME="<existing-GPT-OSS-model-cache-root>"
test -x "$VENV_PATH/bin/python"
test -x "$AGENT_SUPPORT_GPTOSS_PYTHON"
test -r "$AGENT_SUPPORT_M2_200_ARTIFACT_DIR/corpus_200.json"
test -r "$AGENT_SUPPORT_M2_200_ARTIFACT_DIR/index_200.json"
"$VENV_PATH/bin/python" -m agent_support.autonomous_evaluation preflight --arm autonomous_v2
"$VENV_PATH/bin/python" -m pytest -q tests
"$AGENT_SUPPORT_GPTOSS_PYTHON" -c 'from agent_support.gpt_oss_model import check_mxfp4_dependencies; print(check_mxfp4_dependencies("5.22.2"))'
sinfo -o '%P %G %m %l'
```

## One-task GPU smoke command (review, then submit manually)

Choose a partition confirmed to provide the GPU used for job 157532. The
script requests one GPU, six CPUs, 64 GB system RAM, and two hours. It
samples GPU memory during the smoke. This command has **not** been submitted.

```sh
export GPU_PARTITION="<verified-partition>"
mkdir -p "$AGENT_SUPPORT_DATA/autonomous_support_evaluation/logs"
EVAL_ARM=autonomous_v2 EVAL_STAGE=smoke \
  sbatch --partition="$GPU_PARTITION" \
  --output="$AGENT_SUPPORT_DATA/autonomous_support_evaluation/logs/autonomous-v2-smoke-%j.out" \
  scripts/autonomous_support_eval.sbatch
```

The job prints its exact `run_dir`, under
`$AGENT_SUPPORT_DATA/autonomous_support_evaluation/runs/smoke/autonomous_v2/`.
Inspect `summary.json`, `run.json`, and the one file in `episodes/`; the
complete per-decision events are at `trace[*].support_trace.events`.

```sh
export RUN_DIR="<run_dir printed in the Slurm log>"
"$VENV_PATH/bin/python" - "$RUN_DIR" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
episode = json.loads(next((root / "episodes").glob("*.json")).read_text())
summary = json.loads((root / "summary.json").read_text())
print({key: summary.get(key) for key in ("controller_model_calls", "controller_parse_errors",
      "controller_generation_failures", "invalid_support_requests", "support_calls",
      "actions_changed_by_support", "failed_actions")})
for decision in episode["trace"]:
    support = decision["support_trace"]
    print("decision", decision["decision"], "proposal", decision["proposed_action"],
          "action", decision["action"], "stop", support["termination_reason"])
    for event in support["events"]:
        metadata = event.get("controller_metadata", {})
        print("  ", event.get("request", {}).get("operation"),
              event.get("request", {}).get("arguments"),
              event.get("result", {}).get("status"),
              metadata.get("generation_status"), metadata.get("parse_error"),
              "raw_tail", repr((metadata.get("raw_completion") or "")[-250:]),
              "hit_limit", metadata.get("hit_token_limit"),
              "stopped_by_return", metadata.get("stopped_by_final_return"))
PY
```

Review whether missing-final completions are reasoning-only, hit the output
limit, or ended on a return token; whether empty inventory requests cease;
whether valid capabilities beyond state inspection are chosen when useful;
whether action revisions are validated and actually executed; and how much
budget remains after each decision. One smoke task cannot establish a
performance improvement or causal benefit from any support operation.
