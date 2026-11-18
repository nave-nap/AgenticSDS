# Sequential support framework

`autonomous_support_v1` is an isolated runner condition. It uses the existing
TextCraft or ALFWorld task prompt, parser, benchmark adapter, manifest checks,
and per-episode JSON output. Historical conditions still use their existing
runner and support-policy paths. The first prompted controller and three
registered capabilities are described in [prompted_support_controller.md](prompted_support_controller.md).

## Contracts

`agent_support/autonomous_support.py` defines:

- `SupportController.reservation(context)` and `decide(context)`. A decision is
  exactly one `SupportRequest` or an explicit `execute`. The response includes
  controller reasoning cost.
- `SupportContext`, which contains the current task, observation, action
  history, original and current candidate, previous structured results, and
  remaining budget. Each subsequent decision receives the latest result.
- `SupportOperation.validate_arguments`, `reservation`, and `execute`, installed
  through an explicit `SupportExecutor` registry. An operation may return
  information and an optional candidate action. It never receives the episode
  object and cannot call `episode.step` through this interface.
- `SupportBudget`, which limits operations per decision and task, additional
  model calls and tokens per task, and controller decision depth per
  environment decision. Controller and operation calls both consume the
  additional model/token budgets. A model adapter must reserve counted input
  tokens and its maximum output tokens before generation and report actual
  usage within that reservation.
- `SupportTrace`, which records each controller view, decision, request,
  result, action revision, cost, stop reason, and linked environment outcome.
  Controller, operation, and environment-step elapsed seconds are reported
  separately; model inference seconds and total episode elapsed time remain
  distinct measures.

`agent_support/autonomous_runner.py` owns the task-agent proposal and final
environment action. It parses every proposed revision and revalidates the
selected action before the single environment step. On a malformed request,
controller error, failed operation, or invalid revision, it uses the original
validated task-agent action. If the original proposal is invalid, the
benchmark's existing parse-error behavior applies. A budget stop may use the
last valid revised candidate; it is still revalidated before execution.

Only operations registered in code and named in the run config can execute.
Arguments must be JSON data and pass the selected operation's validation.
Controller text cannot name a shell command, Python snippet, import, or file
path for dynamic execution. The framework does not turn TextCraft `inventory`
into a support operation; it remains a budgeted environment action.

## Configuration and extension

`configs/textcraft_autonomous_support_framework.json` is a framework smoke
config. Its `execute_immediately_v1` controller requests no support. A new
controller can be registered in `CONTROLLERS` or injected through
`run_episode(..., support_controller=...)`. A trusted operation factory can be
registered with `register_support_operation`, listed in
`autonomous_support.operations`, or injected with
`run_episode(..., support_executor=...)`. `run(...)` also accepts per-task
controller and executor factories for CPU experiments. These hooks keep
controller selection separate from task prompts and benchmark adapters.
Registered operation factories receive `(config, controller_model)`; the
prompted controller's model can therefore be shared with model-backed supports.

For a model-driven controller, implement strict output parsing in the
controller adapter and return a `ControllerResponse` with the measured cost.
For a model-backed operation, count its prompt with the selected model's chat
template, reserve its configured output ceiling, and return its measured cost
even if its generated answer is malformed. An operation that raises an
exception is recorded as failed with zero *reported* cost; adapters that can
consume inference before failure should catch that case and return an error
result with the consumed cost. This accounting contract is required before
GPU evaluation.

## Minimal multi-support example

The CPU test `test_multiple_different_supports_are_chosen_from_previous_result`
uses a mock controller and two mock operations. Its decision trace is:

```text
task agent proposes: get 1 stone
controller requests inspect(query="recent failure")
inspect returns: {"need": "revision"}
controller observes that result and requests revise(query="wood")
revise returns candidate: get 1 wood
controller explicitly chooses execute
runner validates get 1 wood, steps the environment once, and logs success
```

The second request is conditional on the first result; no sequence is stored
in the executor. This validates framework behavior with mocks. It does not
claim that a learned or LLM controller has made the choices.

Run the repository tests with `python3 -m pytest -q tests`. Running bare
`pytest` from the repository root also collects the ignored
`data/adapt_worktree/tests` checkout and can produce duplicate module names;
the tracked test suite is `tests/`.

The checked-out `main` runner writes each completed episode atomically and
updates `summary.json`. It has no mid-episode or schema-v2 checkpoint/resume
facility. The new condition uses that existing per-episode persistence; a
future move to the modular schema-v2 branch can reuse its verified completed
episode checkpoints. A preempted support loop cannot currently resume from
the middle of an episode.
