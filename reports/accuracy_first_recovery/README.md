# Accuracy-first adaptive recovery: audit, baseline, and evaluation plan

## Objective and evidence boundary

The primary outcome is environment-verified TextCraft task success. Model calls,
support calls, tokens, inference time, and wall time are recorded separately and
never enter the policy score. The existing 150 environment-action ceiling, 220
decision ceiling, per-decision and per-task support limits, model-call and token
limits, wall/inference limits, strict controller parser, and fatal CUDA worker
handling remain in force. No GPU evaluation was run for this change.

This audit uses the checked-in v4 source and two archived episode JSON files in
the original workspace under `reports/flex89_failure_artifacts/.../episodes/`.
Those artifacts are untracked in the original checkout and were not copied or
modified here. These two episodes are diagnostic examples, not a
representative estimate of v4 success.

## Phase 1: current v4 policy audit

| Question | Implemented behavior | Important limit |
| --- | --- | --- |
| Support selection | Qwen proposes an action. GPT-OSS receives the visible task, recent history, recovery state, support catalog, and remaining budget, then emits strict JSON `request`, `execute`, `reflect`, or `stop`. A `request` names one allowlisted operation and arguments. | The prompt asks for autonomous choice, but no coded rule requires a support after failure, uncertainty, or an unverified craft. |
| State adaptation | `RecoveryState` persists strategy, plan, hypotheses, uncertainties, attempted approaches, expected outcome, last action feedback, and recent support findings. The model can revise them and assess progress/support usefulness. | These fields are model judgments. The runner does not turn failed-action counts, negative support outcomes, or stale plans into a mandatory strategy change. |
| Support interaction | Retrieval searches frozen M2-200 examples; state/history/failure inspections expose only visible evidence; prerequisite verification compares a displayed recipe with a fresh observed inventory; candidate inspection checks format, recipe display, and prior use; artifact reasoning uses GPT-OSS and can propose an action. A support's proposed action becomes the current candidate for later supports and the controller. | These operations cannot query hidden simulator state. `inventory` is an environment action. A prerequisite result with no fresh inventory leaves quantities unknown. |
| Repeated support use | The v4 loop can call several supports, including the same operation with different arguments, before an action. Results return to the controller each time. | Maximum 7 external operations and 10 controller turns per decision; 100 external operations per task, plus the other finite limits. |
| Final action | The last valid `execute.action` wins, or `null` uses the current candidate, which may have been revised by support. The runner validates TextCraft syntax and alone calls `episode.step`. Certain controller errors fall back to Qwen's proposal; explicit stop and fatal CUDA/unavailable worker states terminate. | No separate candidate-ranking or prerequisite gate exists before execution. |
| Stagnation/recovery | The model sees failed/repeated action counts in the next state, may request `analyze_failures_v1`, revise strategy/plan, reflect, or stop. The analysis script counts repeated failures after a run. | No deterministic stagnation trigger, failed-strategy exclusion, or competing-strategy search is implemented. The analysis is retrospective. |

The older `agent_support/support_policy.py` is a per-turn policy for a different
runner and defaults to a fixed condition schedule. It is not v4's selector.
V4's multi-support loop, reflection, memory, prerequisite checks, and state
logging already exist. Automatic failure-triggered support selection,
multi-candidate comparison, checked alternative selection, and adaptive
deliberation based on unresolved evidence were proposals before this branch.

### Archived trajectory examples

- Seed 151 succeeded in 7 actions, with 3 failed actions, 1 support call, and 2
  controller parse fallbacks. The first craft failed; a prerequisite check
  confirmed the displayed recipe but reported unknown ingredient quantities
  because no inventory action had been observed. The same craft failed again
  after obtaining glowstone; obtaining redstone then made it succeed. This
  illustrates a support result that was valid but insufficient to verify the
  immediate craft. It is not a case of unrecovered task failure.
- Seed 89 reached the 150-action limit with 140 failed actions and no success.
  The same invalid `craft 6 black wool using 1 black dye` action failed 76
  times; `get 6 black wool` failed 13 times and `get 4 white wool` 11 times.
  Eight support calls occurred: seven `inspect_state_v1` and one
  `verify_prerequisites_v1`, all returning `empty`. The controller fell back on
  all 150 turns: 5 parse errors followed by 145 backend errors. Repeated
  actions and empty support choices are visible, but the backend failure makes
  this primarily an infrastructure-contaminated episode. It cannot establish
  the success rate of a functioning v4 policy. The archived CUDA diagnostic
  and fatal-error work must be resolved before causal policy comparisons.

## Phase 2: proposed policy

The policy operates after a strictly parsed controller `execute` proposal. It
may replace that proposed execution with one allowlisted support request. The
existing loop validates the request, charges its usage, returns its result to
the controller, and repeats. The model can still make its own requests or
reflect. The policy never steps the environment or reads hidden state.

1. **State-aware selection.** Inspect the current action, visible feedback
   since the last successful `get` or `craft`, last progress judgment, prior
   support results, and available operations. A repeated failure, a failed
   last action, or two recent failures triggers failure analysis and a query
   to the frozen episodic memory. Craft proposals get prerequisite checks.
   Candidate inspection checks syntax, displayed recipe, and prior use.
2. **Multi-candidate deliberation.** On stagnation, request two existing
   `reason_over_artifacts_v1` candidate-action analyses: one for prerequisite
   repair and one for a meaningfully different route. Candidate actions and
   failure evidence are supplied to subsequent controller turns. The policy
   inspects generated candidates before using them.
3. **Action arbitration.** If the controller still proposes an action that
   failed since the last visible inventory-changing success, prefer a distinct
   checked candidate. A candidate must parse as a native action and be marked
   admissible by `inspect_candidate_v1`. An alternative craft must also match a
   displayed recipe and have sufficient ingredients in a fresh observed
   inventory. With no checked candidate and no fresh inventory action,
   use `inventory` to gather evidence before repeating the failed craft.
   Environment feedback remains the only success test.
4. **Adaptive depth and safeguards.** Extra cycles occur only while evidence
   remains unresolved and a relevant operation has not already run with the
   same arguments in this decision. Existing depth, support, model-call, token,
   time, and environment-action limits remain hard stops. A policy intervention
   is logged next to the original model decision.

The selector does not claim that textual failure prefixes capture all forms
of stagnation or that two generated candidates are truly different. These are
testable baseline heuristics. A successful `get` or `craft` clears the active
failure signal, so a once-failed craft can be retried after its ingredients
change. The controller's own support requests and explicit stop remain valid.

## Phase 3: minimal implementation and interfaces

- `agent_support/recovery_accuracy_policy.py` defines the replaceable
  `RecoveryPolicy` interface: `guidance(context, state)`,
  `select(context, state, available) -> PolicyDirective | None`, and
  `choose_action(context, proposed) -> action`. `StateAwarePolicy` uses failure
  analysis, memory, and verification without generating two candidates;
  `AccuracyFirstPolicy` adds the two candidate support calls. Both reuse the
  existing support implementations.
- `agent_support/recovery_v4.py` accepts an optional `policy` in
  `adaptive_loop` and `run_recovery_episode`. `None` preserves original v4.
  The strict JSON parser runs before any policy selection. The existing
  executor, accounting, support trace, fallback, and CUDA fatal-error paths
  handle all selected requests. The result records `support_policy_id`.
- No frozen config, manifest, memory artifact, dependency, model weight,
  controller parser, or worker implementation changed. No training system or
  extra planning model was added. A later learned selector can implement the
  same interface and use the same evidence and dispatch path.

The development entry point is an explicit episode call with
`policy=StateAwarePolicy()` or `policy=AccuracyFirstPolicy()`. The frozen v4
evaluation CLI recognizes only its original two arms. Before GPU experiments,
add a new opt-in three-arm evaluator or runtime policy mapping with distinct
run directories and policy IDs; do not repurpose frozen arm names or mutate
their saved protocol/config digests. Preserve the current clean-commit,
preflight, memory-disjointness, and resume checks in that evaluator.

## Phase 4: controlled comparison

| Arm | Runtime policy | Same components |
| --- | --- | --- |
| Original v4 | `None` | Qwen/GPT-OSS weights, revisions, settings, frozen M2-200 memory, support catalog, environment, and ceilings |
| State-aware | `StateAwarePolicy()` | Same; active-failure analysis, memory retrieval, and candidate/prerequisite verification |
| Accuracy-first | `AccuracyFirstPolicy()` | Same as state-aware, plus two competing candidate analyses on stagnation |

Use the same 30 development instances from the frozen recovery manifest, in
the same order and with identical task seeds, initial observations, recipe
order, runtime image, model backend, and limits. Preflight each run for exact
model/config/artifact hashes and disjoint memory. Separate output roots and
policy IDs prevent accidental mixing. One successful infrastructure-stability
smoke per arm precedes a full paired run. Do not tune the selector on a held-out
set; after development, freeze the policy and evaluate on a new held-out set.

Primary score is the fraction of tasks with `episode.success == true` before
the action limit. Report paired task-level wins/losses, success at action
checkpoints, a paired bootstrap confidence interval, and exact discordant-pair
counts. The exploratory development gate is at least two additional successes
for accuracy-first against **each** comparator, with no infrastructure-invalid
episodes. A later held-out accuracy claim requires a positive paired success
difference against both comparators with a pre-specified 95% paired bootstrap
interval excluding zero. The small development sample and uncertainty
intervals must accompany every estimate. Report support operations by type,
controller/support/Qwen/embedding calls and tokens, inference and wall time,
failed actions, and deliberation depth as secondary measurements only.

Classify CUDA OOM/fatal worker states, unavailable worker, backend errors,
parse errors, and environment errors separately from valid task failures and
resource-limit outcomes. Preserve the existing terminal CUDA behavior; never
count a fatal worker episode as an ordinary unsolved task in the primary
accuracy estimate. If an arm has infrastructure-contaminated episodes, report
the count and stop the paired policy comparison until the runtime is stable.
Do not silently retry only one arm or impute its outcome. Re-run matched
instances under a documented, identical protocol if an infrastructure repair
is required.

## Offline verification completed

`tests/test_recovery_accuracy_policy.py` uses fake model outputs and no GPU to
check two candidate support calls, support accounting, checked alternative
selection, prerequisite evidence for alternative crafts, frozen-memory
retrieval, full episode-runner integration, rejection of invalid policy
requests, baseline state-aware behavior, no-failure verification, and the
reset of failure evidence after a successful collection. The existing v4 tests
continue to cover strict parsing, support chaining, fallback, and CUDA fatal
errors. The full offline suite passed: **231 tests**. `git diff --check` passed.
