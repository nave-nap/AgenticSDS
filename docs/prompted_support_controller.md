# Prompted GPT-OSS support controller v1

This is an initial, prompting-based research baseline on the isolated
`autonomous_support_v1` runner path. It does not train or implement a learned
policy. `agent_support/llm_support_controller.py` builds a decision prompt from
the task objective, current observation, last `history_window` action/feedback
pairs, original and current candidate actions, previous support results, the
enabled capability catalog, and remaining support and environment budgets.
Manifest task IDs and hashes are retained in run logs but omitted from model
prompts so the controller does not receive a seed-specific task identifier.
The controller chooses one strictly parsed JSON `request` or `execute` per
step. It receives each operation's structured result before choosing again.
The prompt explicitly treats support as costly and permits immediate action;
it contains no failure-to-capability mapping or prescribed support order.
The expanded TextCraft capability set and frozen-memory wiring are described
in `docs/autonomous_capabilities.md`.

## Available capabilities

- `inspect_history_v1`: returns bounded recent action/feedback pairs from the
  current episode. It does not classify failures or select a next operation.
- `inspect_candidate_v1`: reports action syntax, current ALFWorld
  admissibility or TextCraft displayed-recipe match, and prior repetition.
  These checks do not certify task progress or inventory preconditions.
- `reason_over_artifacts_v1`: asks the same GPT-OSS worker for a bounded note
  or candidate action using the current visible state and prior support
  results. Its output is strictly parsed; any candidate is validated by the
  benchmark action parser before it can be executed.

The controller receives descriptions, argument schemas, model-call counts,
and output-token ceilings for only the
capabilities enabled by its config. A request for an unlisted operation,
unexpected JSON fields, wrong argument types, code fences, or prose fails
validation. No model text becomes a shell command, Python program, import,
or environment step. The runner alone executes a final validated action.

## Model and cost path

`configs/textcraft_prompted_gptoss_controller.json` pins the Qwen task model
and GPT-OSS-20B controller revision. The controller runs in one persistent
subprocess using the validated packed-MXFP4 adapter reused from the
`delegation_token_reservation_fix` branch. This keeps its Transformers 5.22.2
stack separate from the checkout's Qwen Transformers 4.57.6 stack. The same
GPT-OSS worker serves every controller decision and model-backed support
operation in a run; it is not reloaded on each turn. The worker verifies its
model identity, revision, MXFP4 representation, and pinned backend stack
before inference. The run snapshots controller model metadata and worker
stderr in its own directory.

Set `AGENT_SUPPORT_GPTOSS_PYTHON` to the GPT-OSS environment's Python
executable and `AGENT_SUPPORT_GPTOSS_HF_HOME` to its model cache. The parent
process still needs the existing Qwen GPU environment for the task agent.
The runner reserves exact chat-template input tokens plus configured maximum
output tokens before each controller or model-backed support call, then
charges reported input/output tokens and inference time separately by role.
Reasoning tokens are included once in GPT-OSS output tokens. Operation and
controller wall times, environment-step time, and total episode time are
separately logged. If a reservation cannot fit, the loop stops and a
validated candidate action proceeds.

## Output and recovery

An `execute` decision may name one benchmark-formatted action or use `null`
to keep the current candidate. A `request` names one enabled capability with
validated arguments and an objective. The controller's raw generation,
prompt digest, parse status, cost, selected request, result, and final
environment outcome appear in each decision's `support_trace`. Malformed or
incomplete GPT-OSS output is counted as a controller call and becomes an
explicit `malformed_controller_output` stop; the original validated task
proposal is used. A failed support operation likewise falls back to the
original proposal. If the task proposal is invalid, the existing benchmark
parse-error behavior applies, with no environment action.

`reports/autonomous_support_controller/example_traces_mock.json` contains
four **CPU mock** traces using the tracked TextCraft seed-130 beehive task:
immediate execution, a history-inspection → model-reasoning → execution
sequence, a result-dependent choice, and malformed-output recovery. They illustrate the data flow and
logging; they are not observed GPT-OSS decisions or benchmark results.

Run CPU tests with `python3 -m pytest -q tests`. The small integration tests
inject a counted fake GPT-OSS backend, exercise the real prompt and parser,
and verify that one backend instance is shared across two tasks. A worker
protocol test uses a fake model without CUDA. No GPU benchmark was launched.

## Remaining limitations before GPU evaluation

- The GPT-OSS backend and worker protocol are wired and CPU tested, but this
  checkout has not run a GPU smoke to confirm its pinned runtime on the
  current machine.
- The initial capability set has no training-trajectory retrieval or
  deterministic inventory verifier. Those should be added as separate,
  versioned capabilities after leakage and state semantics are audited.
- `reason_over_artifacts_v1` uses the same model as the controller, so its
  recommendation can repeat the controller's mistaken premise. It adds a
  separately logged inference pass, not independent evidence.
- The main checkout persists completed episodes atomically but lacks
  mid-episode resume and the later branch's schema-v2 checkpoint verifier.
  A preempted support loop cannot be resumed in place.
- Backend failures after an inference call may lack a token receipt. The
  controller charges one attempted call and the reserved input estimate,
  then falls back; exact output-token accounting for such a failed worker
  call requires a worker-level receipt or recovery protocol.
- The prompted policy has not been evaluated for selection quality. The
  generated mock traces establish interface behavior, not autonomous
  performance or benefit over fixed support controls.
