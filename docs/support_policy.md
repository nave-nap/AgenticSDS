# Per-turn support policy

The support policy decides **which existing support mechanisms run at each turn**. It does not implement retrieval, prompts, or model generation. Those mechanisms remain in `agent_support/runner.py` and the memory modules. This separates the research question of *when to provide support* from the behavior of each support mechanism.

`agent_support/support_policy.py` defines a `TurnContext` snapshot and a `SupportPolicy.select(context)` interface. The runner calls the selector twice per decision:

1. `before_proposal`: after the environment observation and base prompt are available, before the model's first action proposal.
2. `after_proposal`: after parsing that proposal, before executing an environment action.

The context includes the benchmark, condition, task, current observation, available actions, action/feedback history, remaining action and decision budgets, support calls so far, and (in the second phase) the parsed proposal and raw model output. A policy returns an ordered tuple of support IDs. The runner validates phase and support IDs, then executes the selected mechanisms in that order. If more than one post-proposal support is selected, each sees the current action produced by the preceding support. Each call is logged in `support_events`; the existing `support` field remains as a backward-compatible view of the most recent event. Per-turn `support_policy` logs the policy ID and both selections. Run metadata records `support_policy_id`.

The initial `fixed_by_condition_v1` policy reproduces the existing experiment schedule:

| Condition | Before proposal | After valid proposal |
| --- | --- | --- |
| `baseline` | None | None |
| `expel_retrieval_only_v1` | Retrieve task-level examples once per episode, reuse on later turns | None |
| `m2_trajectory_v1` | None | Retrieve action-conditioned trajectory examples and reconsider if hits exist |
| `m2_always` | None | Use the legacy action memory and reconsider if hits exist |

No learned gating or dynamic choice is claimed for this initial policy. It exists to hold the current behavior steady while later policies are tested.

To try a different schedule with the **same support mechanisms**, implement `select` in a new class, assign a new `policy_id`, register it in `POLICY_REGISTRY`, and set `"support_policy": "your_policy_id"` in a new condition config. Keep the old config and run IDs frozen. The runner constructs a fresh policy for each episode, so a stateful selector does not leak information across tasks. Add a test with a fake model and environment that verifies support selection, model-call cost, and trace logging before running a GPU pilot.

Adding a *new mechanism* still requires an executor path in the runner and a new support ID. For controlled B–C comparisons, `compare_trajectory` requires the same policy ID when both new runs record one, as well as the same corpus, index, top-k, and token budget. A change in policy should be labeled as a separate experiment rather than silently folded into the current B–C contrast.
