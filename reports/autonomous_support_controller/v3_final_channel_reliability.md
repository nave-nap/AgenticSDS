# V3 GPT-OSS final-channel reliability smoke

## Evidence from the saved v2 run

The read-only source is
`/Users/evanpan/Downloads/autonomous_v2/autonomous_v2/textcraft_20261118T173331Z_b42dae0d/`.
The episode JSON SHA-256 is
`d7b90ed0d7272306fcb1a78c367d9c68d254d505fe12c9d2e6fd11284a5144b6`.
It is a successful one-task TextCraft run from commit
`140a88c7e5c8f3e7ed7c1cccef04e6f6827c741c`, with 25 controller calls,
three episodic-memory retrievals, four action changes, and 17 controller
parse errors. The original run and its outputs are unchanged.

The [pinned tokenizer configuration](https://huggingface.co/openai/gpt-oss-20b/blob/6cee5e81ee83917806bbde320786a8fb61efebee/tokenizer_config.json)
defines `<|constrain|>` as special token 200003 and `<|return|>` as 200002.
The [pinned chat template](https://huggingface.co/openai/gpt-oss-20b/blob/6cee5e81ee83917806bbde320786a8fb61efebee/chat_template.jinja)
ends the inference prompt at `<|start|>assistant` and renders ordinary final
messages without `<|constrain|>`. The recorded prompt tails agree. Thus the
variant headers appear in the generated completion, not in the prompt or
because decoding replaced a canonical header. Prompting for JSON may have
influenced the model's choice of header; the trace cannot establish that
mechanism. All 16 `missing_final` completions ended in `<|return|>` and none
hit the generation-token limit. The old parser required adjacency of
`<|channel|>final<|message|>` and therefore rejected headers containing
`final <|constrain|>json`, `JSON`, or `schema` before `<|message|>`.

The read-only command

```sh
python3 -m agent_support.reparse_controller_completions \
  --run /Users/evanpan/Downloads/autonomous_v2/autonomous_v2/textcraft_20261118T173331Z_b42dae0d
```

classified every saved controller completion:

| Offline classification | Count | Decisions |
| --- | ---: | --- |
| Already accepted by v2 | 8 | 1, 2 first call, 9, 11 three calls, 15, 22 |
| Framed final variant; schema-valid decision | 12 | 2 second call, 3, 5, 7, 10, 12, 13, 17–21 |
| Valid JSON; invalid decision schema | 3 | 6 invented operation, 8 and 14 missing `objective` |
| Malformed final header; unsafe to recover | 2 | 4 and 16 |

The v3 decoder accepts only a fully bounded assistant final message with
the observed format labels `json`, `JSON`, or `schema`, preceded by a complete
analysis message and ended by the final-return token. It checks the pinned
special-token boundaries as well as decoded text. It does not scan reasoning
text for JSON, accept unknown format labels, repair missing fields, or turn an
unlisted operation into a valid request. The old decoder remains the default
unless the new config opts into `harmony_final_v3`.

At decisions 20 and 21, the saved model text proposed
`craft 8 cyan stained glass using 8 glass, 1 cyan dye`; the old parser rejected
those outputs. At decision 22 the same action was accepted and the task
succeeded. Offline recovery describes potential decisions only. It does not
mean the rejected actions were executed, nor does it predict a new v3
trajectory under stochastic generation.

The v2 Qwen output at decision 2 was
`Action: get 8 glass, 1 cyan dye`, which its action parser rejected. The
support loop nevertheless spent two controller calls and one memory retrieval
before falling back to no action. V3 records `invalid_task_proposal` and the
ordinary task parse error without entering the support loop. This prevents
support expenditure when there is no validated task-agent action to fall back
to; v1 and v2 behavior is preserved.

## Version boundary and validation

`configs/autonomous_evaluation/autonomous_v3.json` differs from v2 only in
the condition (`autonomous_support_v3`), controller ID
(`prompted_gpt_oss_v3`), and explicit completion parser
(`harmony_final_v3`). The frozen development manifest, Qwen/GPT-OSS model
revisions, memory checksums, support capabilities, action limits, and all
support budgets are unchanged. V3 is restricted to the one-task smoke stage.
The run snapshot records the v3 condition and the worker metadata records
the parser ID. No historical results are rewritten.

Local CPU tests use exact saved completion text for decisions 2, 3, 4, 6,
8, 14, 16, 20, 21, and 22. They cover canonical and constrained headers,
schema-invalid requests, malformed boundaries, safe fallback, model-call
accounting, and the invalid-Qwen-proposal path. The synthetic CPU tokenizer
preserves special-token boundaries. A tokenizer-only check with the actual
pinned GPT-OSS tokenizer is required on Locust before the GPU smoke.

## Mac: review and transfer

Run from `/Users/evanpan/AgenticSDS`. The current branch is already named
`autonomous-smoke-v3`, but its HEAD and `origin/autonomous-smoke-v3` still
point to the v2 commit `140a88c`. All v3 registration and parser changes are
currently uncommitted. A Locust worktree made from that remote branch will
therefore import v2 code and reject `--arm autonomous_v3`. The unrelated
untracked `reports/complete_research_summary/` is excluded from the commit.

```sh
python3 -m pytest -q tests
bash -n scripts/autonomous_support_eval.sbatch
git diff --check
git add -- agent_support/autonomous_evaluation.py agent_support/autonomous_operations.py \
  agent_support/autonomous_runner.py agent_support/autonomous_support.py \
  agent_support/common.py agent_support/gpt_oss_model.py \
  agent_support/llm_support_controller.py agent_support/reparse_controller_completions.py \
  agent_support/runner.py configs/autonomous_evaluation/autonomous_v3.json \
  scripts/autonomous_support_eval.sbatch tests/test_autonomous_evaluation.py \
  tests/test_llm_support_controller.py tests/test_gpt_oss_final_v3.py \
  tests/fixtures/v2_controller_final_channels.json \
  reports/autonomous_support_controller/v3_final_channel_reliability.md
git diff --cached --check
git diff --cached --name-only
git commit -m "Add versioned GPT-OSS final-channel parser for autonomous smoke"
git push -u origin autonomous-smoke-v3
git cat-file -e HEAD:configs/autonomous_evaluation/autonomous_v3.json
git show HEAD:agent_support/autonomous_evaluation.py | rg 'autonomous_v3'
```

These commit and push commands are for review; this preparation did not run
them.

## Locust: clean deployment and preflight

From an existing Locust AgenticSDS checkout, discover the paths of the Qwen
and GPT-OSS environments used for v2. Deploy a separate clean worktree.

```sh
EXISTING_REPO="$(git rev-parse --show-toplevel)"
git fetch origin autonomous-smoke-v3
DEPLOY_REPO="$(dirname "$EXISTING_REPO")/AgenticSDS-autonomous-smoke-v3-registered"
test ! -e "$DEPLOY_REPO"
git worktree add --detach "$DEPLOY_REPO" origin/autonomous-smoke-v3
cd "$DEPLOY_REPO"
git status --porcelain
git rev-parse HEAD
git cat-file -e HEAD:configs/autonomous_evaluation/autonomous_v3.json
git show HEAD:agent_support/autonomous_evaluation.py | rg 'autonomous_v3'
find /shared_data0/evanpan/agent_support -maxdepth 6 -path '*/bin/python*' -print
env | sort | grep -E '^(AGENT_SUPPORT|HF_HOME|VENV_PATH|CONDA_PREFIX)=' || true
sinfo -o '%P %G %m %l'
```

Set the discovered environment and cache paths. The artifact and output roots
are the same as in the completed v2 run:

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
"$VENV_PATH/bin/python" -m agent_support.autonomous_evaluation preflight --arm autonomous_v3
"$AGENT_SUPPORT_GPTOSS_PYTHON" -c 'from agent_support.gpt_oss_model import check_mxfp4_dependencies; print(check_mxfp4_dependencies("5.22.2"))'
```

Check the actual cached tokenizer and the v3 parser on saved completion
structures without loading the 20B model or using a GPU:

```sh
"$AGENT_SUPPORT_GPTOSS_PYTHON" - <<'PY'
import json
from pathlib import Path
from transformers import AutoTokenizer
from agent_support.gpt_oss_model import parse_harmony_completion_v3
config = json.loads(Path("configs/autonomous_evaluation/autonomous_v3.json").read_text())
spec = config["autonomous_support"]["model"]["model"]
tokenizer = AutoTokenizer.from_pretrained(spec["model_id"], revision=spec["model_revision"],
                                          local_files_only=True)
cases = json.loads(Path("tests/fixtures/v2_controller_final_channels.json").read_text())["cases"]
expected = {2: "complete", 3: "complete", 4: "malformed_final_header",
            6: "complete", 14: "complete", 20: "complete", 21: "complete", 22: "complete"}
for case in cases:
    decision = case["decision"]
    if decision not in expected:
        continue
    ids = tokenizer.encode(case["raw_completion"], add_special_tokens=False)
    parsed = parse_harmony_completion_v3(tokenizer, ids, len(ids) + 1)
    print(decision, parsed["generation_status"], parsed.get("final_channel_format"))
    assert parsed["generation_status"] == expected[decision]
PY
```

If the pinned-tokenizer check fails, investigate before a GPU submission.
After selecting a verified GPU partition, submit exactly one task manually:

```sh
export GPU_PARTITION="<verified-partition>"
mkdir -p "$AGENT_SUPPORT_DATA/autonomous_support_evaluation/logs"
EVAL_ARM=autonomous_v3 EVAL_STAGE=smoke \
  sbatch --partition="$GPU_PARTITION" \
  --output="$AGENT_SUPPORT_DATA/autonomous_support_evaluation/logs/autonomous-v3-smoke-%j.out" \
  scripts/autonomous_support_eval.sbatch
```

The job prints its exact `run_dir`, under
`$AGENT_SUPPORT_DATA/autonomous_support_evaluation/runs/smoke/autonomous_v3/`.
Inspect `run.json` for `condition=autonomous_support_v3` and
`controller_model_metadata.completion_parser=harmony_final_v3`. Inspect
`summary.json` and `episodes/*.json` for controller parse errors, recovered
final-channel formats, action revisions, malformed-header fallbacks, and
`invalid_task_proposal` events. The next smoke tests the new protocol; the
saved v2 success is not a prediction of v3 task success.
