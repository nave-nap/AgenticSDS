# Autonomous recovery v4 integration repair

## Evidence and changes

The three Locust samples supplied on 2026-11-19 are copied, with their raw Harmony completions unchanged, into `tests/fixtures/recovery_v4_failed_controller_fixtures.json`. The trimmed fixture records the SHA-256 of the original supplied file. No original smoke trace or evaluation output was edited.

| Sample | Final extraction | Exact rejection | Repaired handling |
| --- | --- | --- | --- |
| Seed 51, decision 3, `inspect_state_v1` | Complete canonical final, valid JSON | Old v4 parser required `reason`, `state_update`, `progress_assessment`, `progress_evidence`, and `support_assessment` even though this request supplied its operation, valid arguments, uncertainty, and expected information. Its prior environment outcome also had an unassessed `last_progress`, so the old pending-assessment rule would reject this request even after the field mismatch was fixed | Accept omitted explanatory and state fields, allow a pending progress assessment to remain unassessed, validate support arguments before state mutation, execute `inspect_state_v1` |
| Seed 126, decision 3, `craft 1 brown dye using 1 cocoa beans` | Complete constrained JSON final, valid JSON | Old v4 parser required `reason`; it also expected `progress_evidence` to be a string, while the output used `[]` for no evidence. Its `last_progress` was pending while the generated `progress_assessment` was null | Accept absent reason and normalize empty evidence list; permit an omitted assessment, retain native action validation, and send the craft command to TextCraft |
| Seed 51, decision 1 | Incomplete protocol: `<|constrain|>reflection` | `malformed_final_header`; the text is not an allowed constrained JSON final | Continue to reject; record the generation status and use the explicit Qwen fallback |

The archived v2 Harmony fixture also shows constrained JSON final headers. The GPT-OSS parser now accepts a complete constrained JSON final with or without a preceding analysis message and with or without a separator space. It still requires exactly one final return token, exact framing, an allowed constraint label, and a body without structural tokens. Unknown support names, malformed support arguments, nonnative actions, unknown state fields, and invalid assessments remain errors. A malformed support request is classified as a controller validation error before a state update can persist.

`autonomous_recovery_v4_repair_v1` identifies this integration in preflight and episode results. The v4 configuration and the expanded v3 control are unchanged. Use a new output root for the repaired development smoke.

## Offline verification

From the Mac repair branch, `python3 -m pytest -q tests` passed with **181 tests**. The regressions replay all three real generations, cover canonical and constrained headers, strict failures, request dispatch for state, history, and memory supports, execute dispatch, multi-support chaining, state and assessment persistence, and fallback accounting. The Mac preflight reports all five protocol checks true; GPU runtime checks require Locust.

## Mac to Locust, user-run commands

The following commands are for manual review and deployment. They have not been run by the repair process. Run from the current Mac checkout:

```sh
cd /Users/evanpan/AgenticSDS
git branch --show-current  # autonomous-recovery-v4-repair
python3 -m pytest -q tests
python3 -m agent_support.recovery_v4_eval preflight --arm autonomous_recovery_v4
bash -n scripts/recovery_v4_smoke.sbatch
git diff --check
git add -- agent_support/gpt_oss_model.py agent_support/recovery_v4.py \
  agent_support/recovery_v4_eval.py tests/test_gpt_oss_final_v3.py \
  tests/test_recovery_v4.py tests/test_recovery_v4_eval.py \
  tests/fixtures/recovery_v4_failed_controller_fixtures.json \
  reports/autonomous_recovery_v4/repair_v1.md
git diff --cached --check
git diff --cached --stat
git commit -m "Repair autonomous recovery v4 controller integration"
git push -u origin autonomous-recovery-v4-repair
```

On Locust, choose the actual SSH host and the same pinned environments used by the original smoke. This deploys a fresh worktree and a separate output root:

```sh
ssh "$LOCUST_HOST"
cd "$HOME/AgenticSDS"
git fetch origin autonomous-recovery-v4-repair
DEPLOY_REPO="$(dirname "$(git rev-parse --show-toplevel)")/AgenticSDS-autonomous-recovery-v4-repair"
test ! -e "$DEPLOY_REPO"
git worktree add --detach "$DEPLOY_REPO" origin/autonomous-recovery-v4-repair
cd "$DEPLOY_REPO"
git status --porcelain  # must be empty
export AGENT_SUPPORT_DATA=/shared_data0/evanpan/agent_support
export RECOVERY_ROOT="$AGENT_SUPPORT_DATA/autonomous_recovery_v4/repair_v1_smoke"
export VENV_PATH="<existing-Qwen-environment-directory>"
export AGENT_SUPPORT_GPTOSS_PYTHON="<existing-pinned-GPT-OSS-python>"
export AGENT_SUPPORT_GPTOSS_HF_HOME="<existing-GPT-OSS-cache-root>"
export HF_HOME="<existing-Qwen-cache-root>"
export AGENT_SUPPORT_M2_200_ARTIFACT_DIR=/shared_data0/evanpan/agent_support/gptoss_memory_v1/memory_scaling_v1/artifacts
export GPU_PARTITION="<verified-L40S-partition>"
"$VENV_PATH/bin/python" -m pytest -q tests
"$VENV_PATH/bin/python" -m agent_support.recovery_v4_eval preflight --arm autonomous_recovery_v4
test ! -e "$RECOVERY_ROOT"
mkdir -p "$RECOVERY_ROOT/logs"
sbatch --partition="$GPU_PARTITION" --export=ALL,RECOVERY_ARM=autonomous_recovery_v4 \
  --output="$RECOVERY_ROOT/logs/recovery-v4-repair-%j.out" scripts/recovery_v4_smoke.sbatch
```

Inspect the two development episode JSON traces under `$RECOVERY_ROOT/smoke/autonomous_recovery_v4/`. Confirm at least one selected support has a `request`, a `result`, and an incremented `support_calls`; verify execute decisions reach `action` and `environment_actions`. Compare `controller_fallback_reason`, `controller_fallback_detail`, `controller_errors`, `controller_fallbacks`, support findings, and recovery state snapshots. A real-model smoke is still needed to measure how often GPT-OSS supplies valid arguments and how often it chooses supports after this integration repair. Do not use held-out50 results for tuning.
