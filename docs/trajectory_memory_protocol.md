# Successful-trajectory memory protocol

## Scope and source

This experiment adapts the **retrieval-only** part of [ExpeL](https://arxiv.org/html/2308.10144v3), using the [authors' implementation](https://github.com/LeapLabTHU/ExpeL) as a design reference. ExpeL retrieves past successful trajectories as in-context examples. The authors use the `all-mpnet-base-v2` sentence embedding model; the [published agent configuration](https://github.com/LeapLabTHU/ExpeL/blob/main/configs/agent/expel.yaml) uses two ALFWorld examples. Our implementation is an adaptation to this project's Qwen agent and action-only prompts. It does **not** reproduce ExpeL's self-reflection retries, insight extraction, manual examples, original agent model, or FAISS index. With this small corpus, cosine ranking over stored vectors is enough. Label it `expel_retrieval_only_v1`, not full ExpeL.

## Conditions

| Label | Config condition | Behavior |
| --- | --- | --- |
| A | `baseline` | Frozen no-support pilot; one model call per decision. |
| B | `expel_retrieval_only_v1` | Retrieve by task at reset. Add the same successful training examples to each action prompt. One model call per decision. |
| C | `m2_trajectory_v1` | Propose an action, retrieve from the same examples using task, observation, and proposal, then ask for one revision when examples fit the budget. Retain the proposal if no example fits or the revision cannot be parsed. |

The old `m2_always` transition-memory condition and its results remain separate. B and C use the **same corpus and index files**, MPNet model, top-k=2, 1,800 Qwen-token example budget, base Qwen model, environment rules, and development tasks. Only the retrieval query and placement of the support call differ. C's second model call is part of its cost and is logged.

The per-turn choice of support is held in the replaceable [`fixed_by_condition_v1` policy](support_policy.md). This first policy reproduces the current condition behavior; later policies can change when support is invoked while reusing the same retrieval and revision mechanisms. The run records policy identity and each turn's selected support calls.

## Corpus and leakage controls

1. Use the already completed, full **baseline training** runs from Drive. No new training run is needed. The 40-task training runs yielded three successful ALFWorld episodes and 11 successful TextCraft episodes.
2. Provide both development and held-out manifests for each benchmark. Corpus construction rejects overlap by task ID and task content hash. ALFWorld sources must be from the train split. Evaluation rechecks the excluded manifest hash, selected task IDs, and content hashes.
3. Store each successful episode's ordered action/feedback trace with its run ID and task fingerprint. The complete source trace stays in the JSON corpus. The prompt renderer skips actions the environment rejected. In the existing successful TextCraft episodes, 122 of 186 environment actions were rejected; showing them as examples would teach invalid commands.
4. Embed task descriptions and accepted action contexts with `sentence-transformers/all-mpnet-base-v2`. B ranks past episodes by task similarity. C ranks the same episodes by a weighted task match (40%) and their best matching accepted action context (60%), using its current observation and proposed action. These fixed weights are an implementation choice to test, not a claim from ExpeL. Save the embedding model and revision, corpus SHA-256, memory IDs, and both vector sets in one index. Evaluation rejects a mismatched corpus or model revision.
5. Keep old artifacts under `agent_support/memory/` and new ones under `agent_support/trajectory_memory/` on Drive. Existing run directories are immutable; each evaluation creates a unique run ID.

The corpus has few ALFWorld successes and may give B or C no useful example for some tasks. The token budget can also leave a decision without examples. Both cases are logged. Neither mechanism is assumed to improve success until the paired runs are reviewed.

## Run sequence

Use [the new Colab launcher](../notebooks/trajectory_memory_launcher.ipynb). It has six numbered sections:

1. Set repository and Drive paths, then mount Drive and install dependencies. Use a published commit SHA in `REVISION` for a frozen pilot.
2. Set `BUILD_TRAJECTORY_MEMORY=True` and build the shared corpora and indices from existing completed training runs. This step can run on CPU. Turn the flag off after success.
3. Check the saved baseline's development manifests, Python, shared package versions, and ALFWorld YAML. If these differ, resolve the mismatch or run a new paired baseline under the new environment before drawing a causal comparison.
4. On a GPU, set `RUN_TRAJECTORY_SMOKE=True` for three development tasks per condition and benchmark. Review `run.json`, `summary.json`, and a few episode traces. Turn the flag off after success.
5. On a GPU, set `RUN_TRAJECTORY_PILOT=True` for the full 20-task B and C pilots. Turn it off after success.
6. Set `RUN_TRAJECTORY_COMPARE=True` to write A–B, A–C, and B–C paired JSON reports under `agent_support/comparisons/` on Drive.

Colab usage limits only delay sections 4 and 5; they do not require rebuilding memory or rerunning training. The same commands can later run on a compatible cluster GPU if Colab remains unavailable.

## Analysis contract

`python -m agent_support.analysis LEFT_RUN RIGHT_RUN --trajectory-comparison --output comparison.json` accepts distinct A/B/C conditions only when both runs cover the identical full development manifest. Shared model settings and benchmark package versions must match; the support-only `sentence-transformers` dependency can be absent from the old baseline metadata. B–C additionally requires identical corpus and index hashes, retrieval policy, top-k, and token budget. The report gives paired helped/hurt/same outcomes, success-rate difference, model calls, tokens, retrieval events/hits, and per-task deltas. A smoke run cannot pass this comparison.
