# Autonomous recovery v4 protocol and observability update

The repaired v4 development smoke succeeded on both tasks. The reported seed 51 trace used episodic memory and history inspection before choosing an action consistent with retrieved information. The remaining failures were two `malformed_final_header` generations, two invalid `kind=request, operation=get` decisions, and one JSON object with an extra closing brace. Prior smoke outputs remain unchanged.

This update clarifies the existing controller protocol: native `get`, `craft`, and `inventory` commands belong in `execute.action`; `request.operation` must name a registered support. It asks for progress judgments from predicted versus observed feedback and for strategy or plan updates when evidence changes the approach. Those fields remain optional, and support selection remains autonomous. Parsing and fallback behavior are unchanged: `operation=get`, malformed JSON, and malformed Harmony finals still fail explicitly.

For observability, `support_assessments` counts recorded useful **and** not-useful judgments in v4 episode results and v4 aggregate totals. `replanning_events` counts revisions of an existing nonempty plan; an initial plan and an identical repeated plan do not count. The new episode and preflight marker is `autonomous_recovery_v4_protocol_v2`. The frozen manifest, budgets, v4 configuration, expanded v3 arm, and earlier results are unchanged.

## CPU verification

`python3 -m pytest -q tests`: **189 passed**. Added regressions for `request.operation=get`, extra-brace JSON, two malformed final fallbacks, optional state fields, recorded true and false support assessments, progress assessments, initial versus revised plans, and v4-only aggregate assessment counts. The malformed-output tests use the reported structures; exact new smoke completions were not available on this Mac. `git diff --check` and `bash -n` pass. Mac protocol preflight checks pass; GPU runtime checks remain for Locust.

## Fresh development smoke, user-run commands

Review and transfer the small update manually from the Mac:

```sh
cd /Users/evanpan/AgenticSDS
python3 -m pytest -q tests
python3 -m agent_support.recovery_v4_eval preflight --arm autonomous_recovery_v4
bash -n scripts/recovery_v4_smoke.sbatch
git diff --check
git add -- agent_support/recovery_v4.py agent_support/recovery_v4_eval.py \
  tests/test_recovery_v4.py tests/test_recovery_v4_eval.py \
  reports/autonomous_recovery_v4/protocol_v2.md
git diff --cached --check
git diff --cached --stat
git commit -m "Clarify recovery v4 controller protocol and accounting"
git push origin autonomous-recovery-v4-repair
```

On Locust, use the same pinned Qwen, GPT-OSS, and memory paths as the successful repaired smoke. Set `LOCUST_HOST` to the actual SSH hostname on the Mac:

```sh
ssh "$LOCUST_HOST"
cd "$HOME/AgenticSDS"
git fetch origin autonomous-recovery-v4-repair
DEPLOY_REPO="$(dirname "$(git rev-parse --show-toplevel)")/AgenticSDS-autonomous-recovery-v4-protocol-v2"
test ! -e "$DEPLOY_REPO"
git worktree add --detach "$DEPLOY_REPO" origin/autonomous-recovery-v4-repair
cd "$DEPLOY_REPO"
git status --porcelain  # must be empty
export AGENT_SUPPORT_DATA=/shared_data0/evanpan/agent_support
export RECOVERY_ROOT="$AGENT_SUPPORT_DATA/autonomous_recovery_v4/protocol_v2_smoke"
export VENV_PATH="<same pinned Qwen environment directory>"
export AGENT_SUPPORT_GPTOSS_PYTHON="<same pinned GPT-OSS Python>"
export AGENT_SUPPORT_GPTOSS_HF_HOME="<same pinned GPT-OSS cache>"
export HF_HOME="<same pinned Qwen cache>"
export AGENT_SUPPORT_M2_200_ARTIFACT_DIR=/shared_data0/evanpan/agent_support/gptoss_memory_v1/memory_scaling_v1/artifacts
export GPU_PARTITION="<same verified L40S partition>"
"$VENV_PATH/bin/python" -m agent_support.recovery_v4_eval preflight --arm autonomous_recovery_v4
test ! -e "$RECOVERY_ROOT"
mkdir -p "$RECOVERY_ROOT/logs"
sbatch --partition="$GPU_PARTITION" \
  --export=ALL,RECOVERY_ARM=autonomous_recovery_v4,RECOVERY_SMOKE_COUNT=2 \
  --output="$RECOVERY_ROOT/logs/protocol-v2-smoke-%j.out" scripts/recovery_v4_smoke.sbatch
```

Before a 30-task run, inspect the two new development episode files under `$RECOVERY_ROOT/smoke/autonomous_recovery_v4/*/episodes/`. Check `controller_fallback_detail`, `controller_errors`, `controller_fallbacks`, `support_calls`, `support_assessments`, `support_useful_judgments`, `progress_assessments`, `strategy_changes`, `replanning_events`, and the corresponding controller events and state snapshots. Do not use the two seeds to tune effectiveness or consult held-out50 results.
