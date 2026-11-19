# Autonomous Recovery v4 — exploratory protocol v1

## Objective and scope

V4 is an autonomously deciding, **within-task adaptive** agent. GPT-OSS chooses whether to execute, revise, use a support, reflect, or stop. Persistent controller state evolves as it sees environment feedback and support results. No model weights update across tasks. The evaluation is exploratory, and the previously observed held-out50 outcomes do not constitute confirmatory evidence for this algorithm.

The new code and manifests live on branch `autonomous-recovery-v4` in the sibling Mac worktree. The running GPT-OSS control experiments, frozen v3 runner, prior manifests, plans, and memory artifacts remain untouched. Outputs use a new Locust root.

## Architecture audit and reuse

- Qwen proposals reuse `QwenModel`, `textcraft_messages`, and `parse_textcraft`.
- GPT-OSS uses the pinned v3 worker, tokenizer revision, low reasoning effort, Harmony final-channel parser, decoding settings, and per-task seed derivation.
- TextCraft uses the existing `TextCraftEpisode`, frozen recipe order, and manifest replay. The environment alone determines task success.
- V3 already provides `SupportContext`, `SupportExecutor`, `SupportBudget`, `run_support_loop`, the `prompted_gpt_oss_v3` controller, and four external supports. Expanded-budget v3 invokes these exact components, including its invalid-Qwen behavior and ordinary capability catalog.
- Existing supports: `retrieve_episodic_memory_v1` is a real embedding/index query; `inspect_state_v1` and `inspect_history_v1` transform **visible** task/history data; `reason_over_artifacts_v1` is a separate model inference. Internal GPT-OSS reflection in v4 is **not** counted as an external support.
- V4 adds two visible-evidence supports: `analyze_failures_v1` summarizes failed/repeated actions, and `verify_prerequisites_v1` compares recipe ingredients with a **fresh, observed** inventory response. With no fresh inventory response, quantities remain unknown. `inspect_candidate_v1` already exists and is enabled for v4. Planning uses controller state or the existing `reason_over_artifacts_v1` with `output_type=plan`; it is not a privileged plan oracle.

The v4 support catalog exposes descriptions, argument schemas, return schemas, expected information, and estimated model-call/output cost. The controller dynamically supplies request arguments, an uncertainty, and expected information. Every support invocation and next-step usefulness judgment is logged. No support can step TextCraft or inspect hidden simulator state. Inventory remains an ordinary environment action.

## Controller state and protocol

`RecoveryState` contains an objective, subgoal, working strategy, plan, hypotheses, uncertainties, attempted approaches, an expected outcome, the last observed action outcome, and bounded support findings. Model-authored claims stay in interpretation fields; raw observations and full support results remain in the trace. State lists are capped at 8 entries and text fields at 400 characters. Support-result excerpts in the live state are capped; the complete result stays in the per-decision trace.

Every GPT-OSS final-channel response is one strict JSON object with `kind`, `reason`, `state_update`, `progress_assessment`, `progress_evidence`, and `support_assessment`. The four kinds are:

- `execute`: use or revise the candidate and predict observable feedback.
- `request`: choose an allowlisted support with validated arguments, uncertainty, and expected information.
- `reflect`: update state and deliberate again, consuming a controller call but no external support.
- `stop`: declare `no_viable_strategy` or `budget_exhausted`; never declare success.

After an environment action, the runner saves the predicted outcome, actual feedback, failed-action flag, repetition count, observation-change flag, and environment success. On the next controller call, GPT-OSS must assess progress from that evidence. After a support result, it must judge whether the result was useful and resolved the stated uncertainty. These signals inform model decisions; **there are no failure-count or timeout-based support triggers**. The runner only enforces budgets and action validity.

Malformed controller output, backend errors, or a bounded deliberation-loop stop are recorded. Where a valid Qwen proposal remains, the runner can fall back to it; explicit model stop and global wall/inference ceilings terminate the task. Internal loops are bounded by depth and per-decision operation limits. An incomplete episode checkpoint preserves its trace for audit, while resume restarts that task because TextCraft simulator state is not serialized.

## Frozen exploratory task set and budgets

`autonomous_recovery30_v1.json` contains the existing 20 development tasks in original order, followed by deterministic TextCraft seeds 200–209. Their task IDs and exact initial observations are recorded. The manifest digest is `150c7bbd973936e0f3ef8fe1266060e88de0181f0068ce98bc7bce0989a2d562`. All 30 are disjoint by ID and observation hash from the original 100 held-out tasks, including heldout50. Preflight verifies the pinned M2-200 corpus has no task-ID or observation-hash overlap with these 30 tasks. Both arms read the same frozen corpus; no result is written back to it. This extends the development partition; it is not a fresh independent held-out benchmark.

| Ceiling | Both arms |
|---|---:|
| Environment actions | 150 |
| Qwen task decisions | 220 |
| External support operations per decision / task | 7 / 100 |
| Additional controller, support, and embedding calls combined | 500 |
| Total Qwen + additional calls | 720 |
| Internal controller depth per task decision | 10 |
| Additional GPT-OSS input + output tokens | 3,000,000 |
| Per-task wall time | 2,400 s (40 min) |
| Per-task model/support inference time | 2,100 s (35 min) |

The 500 additional-call ceiling accommodates up to 220 controller calls plus support chains and embedding retrievals; the 720 total-call ceiling adds 220 Qwen proposals. These are provisional development budgets. There is no hidden smaller per-task action limit. Time is checked at inference and action boundaries; a single in-flight model call can exceed a ceiling before it returns. The GPT-OSS worker retains its pinned 1,800-second backend timeout, so Slurm still provides the outer job limit.

Expanded v3 changes the frozen v3 config's resource ceilings for this experiment and runs the existing controller and support loop. V4 uses the same Qwen/GPT-OSS models, memory, task set, and ceilings, but changes controller state/protocol and enables three additional visible-evidence supports. Any performance difference therefore combines algorithm and toolbox changes; it is **not** an isolated causal estimate of support selection.

## Evaluation records and interpretation

Each decision checkpoint contains Qwen output, GPT-OSS decisions, state versions/changes, support requests/results/costs, progress assessments, selected environment action, and feedback. Completed task snapshots additionally include actions, decisions, failed actions, support counts, Qwen/GPT-OSS calls and tokens, embedding retrieval calls/time, wall time, exact termination reason, strategy changes, and replanning events. Aggregation reports cumulative success at action ceilings 30, 60, 90, 120, and 150. Paired analysis reports success disagreement, bootstrap uncertainty, Wilson intervals, computation differences, support sequences, repeated failures, useful-support judgments followed by action changes, and successful tasks requiring more than 30 actions. These are descriptive associations.

## Limitations and open questions

- The existing M2-200 memory is a fixed prior corpus. It is checked for overlap, but its quality and task distribution may still favor some tasks.
- V4 has a richer support toolbox than expanded v3. A later factorial or replay study is needed to isolate controller-state value from new support value.
- State updates and progress/usefulness assessments are model judgments; logs permit audit but do not make them ground truth.
- Per-task time ceilings are checked between calls. A stalled in-flight Qwen call or GPT-OSS worker call can overrun until its backend/Slurm limit.
- Resume is at task boundaries; interrupted partial episodes are rerun from the initial observation. Their checkpoints remain for diagnosis.
- The 30-task development evaluation can guide research but cannot validate generalization. A fresh held-out set is required before a final claim.
- Future research can ask whether recovered strategies should become a learned policy. This version does not train or update model weights.
