# Run format

Schema-v2 development runs add `checkpoint.json`, immutable episode
publication, component and artifact provenance, structured policy decisions,
budget usage, and verified restart history. See the
[modular framework guide](modular_framework.md). The schema-v1 format below
remains readable by the same analysis loader.

Each invocation creates `results/<benchmark>_<UTC timestamp>_<random suffix>/` with:

```text
config.json             Exact experiment configuration
manifest.json           Exact selected tasks, including a smoke-test limit
alfworld_config.yaml    ALFWorld-only source config snapshot
textcraft_recipe_order.json  TextCraft-only frozen recipe file order and data hash
run.json                Status, provenance, versions, hashes, selected count
summary.json            Updated after every completed episode
episodes/0000_*.json    One full trajectory per task
```

`run.json` is marked `running`, `complete`, or `failed`. It records the original and selected manifest hashes, Git commit, package and Python versions, and the SHA-256 of the ALFWorld source YAML or TextCraft recipe order snapshot when applicable. An interrupted run retains already completed episode files; `summary.json` reflects those episodes, and incomplete runs cannot be compared as finished results.

Every episode records the benchmark and task ID; success and final status; environment-action and model-decision counts; parser and failed-action counts; input and output token totals; inference and elapsed time; initial observation; and a `trace` array. Each trace entry stores the decision number, model output, parsed action, input observation, output observation, token counts, and benchmark outcome. ALFWorld entries also store the action list visible at that decision. Parse errors have no environment-action number.

The runner accepts `baseline`, the frozen legacy `m2_always`,
`expel_retrieval_only_v1`, and `m2_trajectory_v1`. Supported runs keep the same
benchmark adapters, base prompts, parsers, and action budget. Their episode
and summary files additionally record support calls, extra model calls and
tokens, retrieval time, support parse errors, and decisions whose action
changed after support. `run.json` records the content hashes of the loaded
memory corpus and index and the `support_policy_id` used to select support.

Each supported decision has a `support_policy` object with the policy ID and
the selected support IDs before and after the first model proposal. Each
executed support call is recorded in the ordered `support_events` list with
its retrieved memory IDs, similarity scores, displayed content, retrieval
time, and any revision output. The `support` field remains a compatibility
view of the latest event. The first proposal remains in `proposed_action`
and `proposal_raw_model_output` when post-proposal support runs. This makes
multiple support calls on one turn auditable without changing the baseline
trace format.
