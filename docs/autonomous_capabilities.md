# Initial autonomous support capabilities

`configs/textcraft_prompted_gptoss_capabilities.json` enables four independent
operations on the existing `autonomous_support_v1` runner. The controller sees
their descriptions, argument schemas, LLM and embedding-call costs, and token
ceilings. It selects one operation per decision and sees each structured result
before deciding again. The production policy contains no required order.

| Capability | Request | Returned information |
| --- | --- | --- |
| `retrieve_episodic_memory_v1` | Controller-written `query`, `scope` (`task`, `steps`, `both`), `top_k` | Bounded successful training trajectories, similarity components, memory and source IDs, corpus/index digests, embedding and retrieval time |
| `inspect_history_v1` | `scope` (`recent`, `failures`, `repeated`), `limit` | Indexed action/feedback pairs, failure markers derived from visible text, occurrence counts |
| `reason_over_artifacts_v1` | Controller-written `objective`, `output_type` (`note`, `plan`, `candidate_action`) | Strictly parsed model artifact, model calls, tokens, inference time; a candidate is validated before use |
| `inspect_state_v1` | `scope` (`inventory`, `recipes`, `craft_feedback`, `all`), `limit`, optional substring `query` | Only the displayed goal and recipes, current observation, previously observed inventory reply, and crafting feedback |

`inspect_state_v1` never calls the TextCraft `inventory` command. If the agent
has not taken that environment action, its inventory field is unavailable.
The history operation uses the benchmark's visible failure-feedback pattern.
It does not pass the environment's `failed_action` flag to the controller or
invent a diagnosis beyond the observed failure marker.
Reasoning can return a plan as information, but it cannot step the environment.

## Frozen memory adapter

The memory capability loads the supplied 200-trajectory TextCraft corpus and
schema-v2 MPNet index from the paths and SHA-256 content digests pinned in the
new config. It uses the existing
`TrajectoryMemory` constructor to check corpus digest, model revision, vector
order, dimensions, and accepted-step indexing. The runner also checks the
evaluation manifest exclusion digest, task IDs, and task fingerprints before
starting a run. It does not rebuild either artifact or retrain MPNet.

The controller writes its own bounded retrieval query. The adapter embeds that
query once and ranks the frozen task vectors, accepted-step vectors, or their
existing 0.4/0.6 combination. It renders only accepted training steps through
the existing trajectory renderer. A configured result-token ceiling and `top_k`
limit bound the artifact. Results include source run/task IDs and similarity
components; they are examples, not proof that an action will work now.

The supplied paths were checked locally: the corpus contains 200 trajectories,
the index contains 200 matching IDs and 1,187 accepted-step vectors, and the
index's corpus digest matches. This is an artifact integrity check, not a
semantic retrieval or benchmark evaluation. The CPU integration test replaces
MPNet's embedding call with a deterministic vector while using the actual
frozen artifacts. A full run requires the pinned MPNet model and GPT-OSS/Qwen
runtime environments.

## Costs, failures, and traces

Retrieval reserves and charges one additional model call for the MPNet query
embedding, with elapsed retrieval time in metadata. No LLM input/output tokens
are attributed to that embedding. Its time appears in support inference and
retrieval totals. History and state inspection report zero model calls. The
reasoning operation reuses the controller's persistent GPT-OSS worker and
reports one additional model call with tokens and inference time. Invalid
arguments stop the support loop safely. Missing visible
information or an empty retrieval returns `empty`; model or retrieval errors
return `error` and the runner falls back to the validated task proposal.

`reports/autonomous_support_controller/capability_traces_mock.json` provides
compact **CPU mock** traces for immediate execution, composition of all four
operations, and a second operation chosen after inspecting a memory result.
The sample controller decisions are scripted and are not GPT-OSS observations.
Run `python3 -m pytest -q tests` for CPU tests. No full benchmark or GPU
experiment has been launched.
