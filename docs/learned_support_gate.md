# First learned support baseline

The `learned_gate_v1` controller is a CPU ridge score placed in front of the
existing prompted GPT-OSS controller. It chooses **execute now** or **let the
prompted controller make another support decision** at every step of the
support loop. The prompted controller still chooses the operation and its
arguments. The gate is rescored after every support result. No support
operation can step the environment; the runner validates and executes one
final action.

The method and evidence limits are in
`reports/autonomous_support_controller/learned_support_methodology.md`.
There is currently no real paired intervention dataset or fitted gate
artifact. The config template is deliberately not runnable until its artifact
digest is replaced. CPU tests use synthetic pairs and mock model responses;
they are not evidence of a success improvement.

## Required paired-data format

`python -m agent_support.learned_support_training` accepts a JSON object with:

```json
{
  "schema_version": 1,
  "partition": "training",
  "pairing_protocol": "replayed_identical_textcraft_state_v1",
  "manifest_sha256": "<canonical digest of training manifest>",
  "memory_corpus_sha256": "<canonical digest of frozen M2-200 corpus>",
  "task_agent_model_revision": "<pinned Qwen revision>",
  "controller_model_revision": "<pinned GPT-OSS revision>",
  "records": [
    {
      "pair_id": "<unique state/seed pair ID>",
      "task_id": "textcraft:seed:<seed>",
      "task_fingerprint": "<initial-observation SHA-256>",
      "decision_state": {"benchmark": "textcraft", "task_prompt": "<visible prompt>",
                         "observation": "<visible observation>", "history": [],
                         "decision": 1, "original_action": "get 1 oak logs",
                         "current_action": "get 1 oak logs", "results": [],
                         "remaining_budget": {"operations_this_decision": 4,
                                              "operations_task": 12,
                                              "additional_model_calls": 30}},
      "state_sha256": "<canonical digest of decision_state>",
      "replay_verified": true,
      "rollout_seed": 42,
      "followup_policy_id": "<frozen follow-up policy ID>",
      "branches": {
        "no_support": {"complete": true, "task_success": false,
                       "total_model_calls": 10, "total_tokens": 9000,
                       "rollout_seed": 42,
                       "followup_policy_id": "<same ID>"},
        "prompted_support": {"complete": true, "task_success": true,
                             "total_model_calls": 13, "total_tokens": 12000,
                             "rollout_seed": 42,
                             "followup_policy_id": "<same ID>"}
      }
    }
  ]
}
```

The sample shows required fields; a collector should retain additional raw
prefix, model, controller, support, and rollout traces alongside each pair.
The trainer checks split and memory overlap, state digests, both branches,
complete outcomes, and shared seeds/policy IDs. `replay_verified` is a
collector assertion, not independent proof: run the replay verifier on raw
prefixes and retain its evidence before admitting data. Dataset task IDs and
exact initial-observation hashes must be disjoint from development, held-out,
and frozen memory tasks. Multiple pairs per training task are allowed and are
weighted to give each task equal total weight.
At least ten distinct eligible training tasks are required for the first gate.

## Artifact paths and commands

Memory locations are config settings, not paths compiled into the controller.
For Locust, set `autonomous_support.memory.corpus_path` and `index_path` in a
**copy** of the config to:

```
/shared_data0/evanpan/agent_support/gptoss_memory_v1/memory_scaling_v1/artifacts/corpus_200.json
/shared_data0/evanpan/agent_support/gptoss_memory_v1/memory_scaling_v1/artifacts/index_200.json
```

Keep the pinned corpus/index digests in the template. The corresponding
provenance file is
`/shared_data0/evanpan/agent_support/gptoss_memory_v1/memory_scaling_v1/artifacts/index_200.provenance.json`.
For local development, keep the `~/agent_support_data/m2_200/` paths if copies
are present, or use the mock memory in CPU tests. Never regenerate or overwrite
the frozen artifacts.

Audit a complete run's environment transitions in an environment with
TextCraft 0.0.3 installed:

```sh
PYTHONHASHSEED=0 .venv/bin/python -m agent_support.intervention_replay --run <run-directory>
```

After an audited paired collector has produced a dataset, fit the CPU gate:

```sh
python3 -m agent_support.learned_support_training \
  --paired-data <paired-training-data.json> \
  --training-manifest data/locust_transfer/manifests/textcraft/training.json \
  --development-manifest manifests/textcraft/development.json \
  --heldout-manifest manifests/textcraft/heldout.json \
  --memory-corpus <frozen-corpus_200.json> \
  --output <new-gate-artifact.json>
```

Before a GPU evaluation, set `autonomous_support.learned_gate.artifact_path`,
`artifact_sha256`, and a frozen `minimum_effect` in a copy of
`configs/autonomous_learning/learned_gate.template.json`. Use the same memory
paths/digests, task manifest, model revisions, budgets, action limits, seed,
and runtime as the prompted-controller run. The runner snapshots the learned
artifact and records its digest. Compare complete frozen development runs:

```sh
python3 -m agent_support.learned_support_compare \
  --prompted-run <prompted-run-directory> \
  --learned-run <learned-run-directory> \
  --output <new-comparison-report.json>
```

The comparator checks the full 20-task manifest, config agreement except for
the gate, runtime/code and memory provenance, artifact snapshot, and task-level
outcomes. It reports model calls, tokens, support sequences, and paired helped
and hurt tasks. Commit the code before either run so a shared commit actually
identifies the implementation.
