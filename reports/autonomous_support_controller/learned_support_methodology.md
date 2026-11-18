# Learning adaptive support allocation: first defensible baseline

## Evidence currently available

No completed `autonomous_support_v1` GPU run was found in this checkout, the
available `agent_support_data` tree, or the local temporary run bundles as of
2026-11-18. The autonomous examples are CPU mocks and cannot measure support
effectiveness. The preserved 20-task GPT-OSS TextCraft standalone,
reconsideration, and M2-11 runs provide complete action/feedback trajectories,
but their supports were selected by other policies and they do not contain
counterfactual outcomes for alternative choices at the same state. Their
14/20 success counts must not be used as labels that a particular support
call helped.

The frozen M2-200 corpus contains 200 successful training trajectories and is
separate from the 20 development and 100 held-out task identities and exact
initial-observation hashes. Of the 40 tasks in the staged original TextCraft
training manifest, 11 appear in that memory corpus; the remaining 29 are
eligible candidate tasks for an initial disjoint intervention dataset. This
is a small training pool. Any larger training manifest must be frozen and
checked against corpus, development, and held-out identities and hashes
before collection. Do not train on development or held-out trajectories.

| Signal in complete traces | What it establishes | Use for learning |
| --- | --- | --- |
| Final task success / terminal TextCraft reward | Goal reached within the action limit | Primary outcome for paired intervention branches; sparse and delayed |
| `failed_action` and visible “Could not”/“Wrong item” feedback | Immediate command rejection | Diagnostic or auxiliary outcome, not proof of task progress |
| Repeated rejected action | A local failure loop | Diagnostic; reduction may be useful, but can coexist with off-goal behavior |
| Accepted `get` or `craft` action | Environment changed inventory or produced an item | Activity, not necessarily progress toward the goal |
| Inventory or recipe dependency change | Potential subgoal movement | Inventory is not always visible to the agent; a validated, goal-aware progress metric does not yet exist |
| Support-induced action replacement | Controller changed a candidate | Activity only; replacement may help or harm |
| Support result `ok` or memory similarity | An operation returned an artifact | Availability/relevance diagnostic, not intervention value |
| Later actions and final success after support | Downstream outcome of one realized path | Confounded without a same-state alternative branch |
| Model calls, tokens, retrieval and wall time | Resource consumption | Negative utility term and cost-efficiency outcome |

Support operations do not step TextCraft. They can change the candidate action
or information in the controller context; the environment state changes only
when the runner executes a final action. The benchmark reward is zero until
the goal item is crafted, so it provides no native dense progress reward.
Counting accepted actions, retrieved memories, plan length, or replacements
as “progress” would reward activity rather than task completion.

## What reproducible branching supports

`TextCraftEpisode.reset` reconstructs the task from the manifest's exact
displayed recipes and goal, starts with empty inventory, and checks the
recipe catalog. TextCraft 0.0.3 `step` uses goal, inventory, and a frozen
crafting tree; it does not sample randomness during `get`, `craft`, or
`inventory`. The replay utility checks each prefix action against recorded
input observation, output observation, reward, done, success, and failure.
With `PYTHONHASHSEED=0` in the installed TextCraft environment, all executed
actions in three preserved runs replayed exactly: standalone **298/298**,
reconsideration **280/280**, and M2-11 **297/297**. A seed-130 three-action
prefix was independently reconstructed twice; `craft 4 oak planks using 1 oak
logs` and `inventory` gave distinct, repeatable next observations. These
checks support environment-level branching on this installation. They do not
prove bitwise reproducibility for another package version or for GPU model
generations. Re-run the replay audit on every new run before creating labels.

For a training decision state, collect a paired intervention as follows:

1. Save the **visible** task prompt, current observation, agent action/feedback
   history, frozen task-agent proposal, prior support results, and remaining
   budgets. Store task ID and fingerprint only as provenance, outside the
   policy feature state. Record the exact executed environment-action prefix.
2. Independently reconstruct the prefix for every branch. Abort the pair if
   any recorded observation/outcome differs. Compare an auditor-only digest of
   the reconstructed goal and inventory across branches before intervention;
   never feed this hidden digest or inventory into the controller.
3. Branch A is **execute now** with the frozen proposed action and no further
   support at this decision. Branch B permits the prompting-only controller
   to continue seeking support under the same capability allowlist and
   budgets. Validate every final action. For later studies, add individual
   capabilities or candidate procedures as separate, predeclared branches.
4. Execute each branch's final action, then run the **same follow-up policy**
   to termination or the action limit. Log the whole trajectory, final success,
   failed actions, calls, tokens, support sequence, and stop reason. The
   intervention is the decision to continue support from this state; later
   state divergence is part of its total downstream effect.
5. Use multiple predeclared rollout seeds and randomized branch order. Keep
   task-agent proposal fixed at the branching state. Use separate RNG streams
   for proposal, support, and follow-up generation; common random numbers may
   reduce variance, but different prompt/call sequences cannot be assumed to
   consume identical GPT-OSS samples. Record every seed and model revision.
   Qwen's greedy proposals reduce one source of variation but do not make
   GPT-OSS controller output deterministic.
6. Limit branching states per task with a fixed sampling rule that includes
   no-failure and failure states, early and late decisions, and states after
   prior support results. Do not sample only states where the controller
   happened to ask for help. Weight tasks equally when fitting.

This is costly: each pair needs at least two support decisions/rollouts and
possibly many GPU calls. Start with a CPU replay audit and a small GPU
feasibility pilot after approval, then budget data collection based on
observed branch cost. The current implementation verifies replay and exposes
the paired-data training interface; it does **not** run GPU branch rollouts or
claim that a dataset's `replay_verified` assertion is independently proven.
An intervention dataset must be generated by an audited collector and retain
its raw branch traces.

The paired target is the difference in eventual task success minus a
predeclared call and token penalty. It estimates the **total value of
continuing support from a sampled state under a fixed follow-up policy**.
Immediate command acceptance and repeated-failure reduction are secondary
diagnostics. If branch rollouts or RNG control cannot be made reliable, use
randomized online assignment with known probabilities and report policy-level
effects; do not label observational traces as causal intervention effects.

## Candidate algorithms

| Approach | Variable-length support and open requests | Data / compute | Main limitation now |
| --- | --- | --- | --- |
| Cost-sensitive support-value predictor | Scores execute versus continuing; LLM constructs each request; rescore after every result | Paired state rollouts; small CPU fit | Learns stopping/allocation only, not operation choice |
| Contextual bandit over support choices | Choose execute or a small catalog of request templates with known propensities | Randomized exploration and many labeled choices | Open-ended arguments make the action set large; current deterministic logs lack propensities |
| Sequential value estimates / fitted Q iteration | Can value another support step versus execute using updated support state | Full transitions and delayed outcomes; substantially more rollouts | Off-policy bias, sparse terminal reward, unstable estimates at low sample size |
| Hierarchical RL / options | Learns multi-step procedures as reusable options, potentially variable length | Large training set, expensive GPU rollouts, option discovery | Premature for 29 immediately eligible disjoint tasks and no autonomous outcomes |
| LLM controller with learned usefulness estimate | Gate or annotate prompted decisions; preserves open-ended construction | Same paired labels plus existing controller | LLM can still choose poor capability/arguments after gate opens |

The first baseline is the last row implemented as a **cost-sensitive ridge
gate**. It is simpler and more sample efficient than RL. At each support-loop
step it extracts 13 features from visible state only: decision/budget position,
action type, visible failure/repetition, displayed recipe match, and prior
support-result status. It predicts the net value of letting the existing
prompted controller make another decision. If the estimate is below a frozen
threshold, it explicitly executes the current validated candidate; otherwise
the existing GPT-OSS controller chooses `execute` or one safe support request.
The gate is applied again after each support result, so it can learn when to
stop a variable-length procedure. It does **not** yet learn which operation
to request or how to write its arguments. That is the next algorithmic step if
the gate demonstrates value.

The CPU trainer uses regularized weighted least squares with equal total
weight per training task. A training pair needs identical state digest,
verified replay, the same follow-up policy and rollout seed for both branches,
complete outcomes, and explicit call/token costs. It rejects records that
overlap the memory corpus, development, or held-out tasks. The default target
penalties are 0.01 success units per extra model call and 0.001 per extra
1,000 tokens; regularization is 1.0. These are predeclared baseline choices,
not tuned on held-out data. The artifact records dataset, manifest, memory,
model revisions, feature schema, weights, and label counts. The runner pins
its digest and snapshots it with each learned-policy run. Each decision trace
records the predicted net effect and whether the gate delegated to GPT-OSS.

## Evaluation and release gates

1. Obtain a complete prompted-controller run and audit its traces/replay.
   The current checkout has none. Preserve its exact config, artifact digests,
   code commit, task manifest, seeds, runtime, and budget. Do not treat the
   earlier CPU mock traces as outcomes.
2. Freeze a disjoint training-task manifest and intervention sampling rule.
   Exclude all M2-200 source task IDs/fingerprints and the 20 development/100
   held-out identities/fingerprints. Collect paired training interventions
   with the identical-state replay protocol above. Keep raw branch traces.
3. Fit the gate on **training** pairs only. Use development tasks to check
   feature shift, no-support rate, procedure lengths, success and cost, and
   to select among predeclared thresholds if necessary. Do not refit on the
   20 development outcomes. Keep the 100 held-out tasks untouched until the
   policy and threshold are frozen.
4. Run prompted and learned-gated controllers on the same frozen manifest,
   task agent, GPT-OSS revision, memory index, capabilities, budgets, action
   limits, code commit, and runtime. The learned gate changes only the
   controller condition and incurs its CPU scoring cost. For stochastic
   GPT-OSS generation, predeclare multiple paired seeds if resources allow;
   keep task as the analysis unit and cluster replicates by task.
5. Report paired task helped/hurt sets, success difference, calls/tokens/time,
   controller decisions avoided, no-support frequency, support sequences,
   budget stops, parser/generation errors, and task-level trace examples.
   The strict comparator rejects mismatched run settings. A success increase
   with much higher compute is not evidence of better allocation. A gate that
   mostly says “support” has not learned a useful selective policy.
6. Only after a frozen development analysis, perform one held-out evaluation
   with no further fitting or threshold changes. State the repeated use of
   the 20-task development set and do not interpret correlations between
   chosen support calls and outcomes as individual causal effects.

The current code contains the learner, replay verifier, strict comparator,
config template, and CPU tests. **No real gate artifact has been fitted**
because the required paired intervention records do not yet exist. The
template's zero digest must be replaced with the fitted artifact digest before
any GPU evaluation. No GPU job or model training was launched here.
