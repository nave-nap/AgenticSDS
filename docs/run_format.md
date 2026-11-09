# Baseline run format

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

The runner accepts only `condition: baseline` for now. A future support implementation should add support-specific fields and its own compatibility contract, while retaining the same benchmark adapters, task manifests, base prompts, parsers, action budget, and result fields. Add retrieval IDs, similarity scores, support calls, and support tokens when M2 is implemented.
