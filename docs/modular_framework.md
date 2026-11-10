# Modular development experiments

The schema-v2 runner keeps the historical Qwen, TextCraft, ALFWorld, M2,
reconsideration, trajectory-memory, and ADaPT implementations. Historical
schema-v1 configs and run files remain readable. Schema-v2 experiments add
explicit component selection, budgets, provenance, and verified recovery.

## Components and decision flow

1. A benchmark driver builds the existing prompt and parses a model response.
2. A policy receives a `TurnContext` snapshot before and after the first
   proposal. The snapshot contains only the current observation, previous
   actions and feedback, available supports, and remaining budgets.
3. The policy returns `SelectionDecision` with support IDs, a reason, trigger,
   and budget snapshot. The runner validates phase and availability.
4. A support adapter handles its own retrieval and prompt. The runner records
   its event, invokes the model for a revision when requested, and applies the
   benchmark parser. ADaPT remains an episode-level controller with its own
   planner and executor loop.

The default `fixed_by_condition_v1` schedule is unchanged for schema-v1 runs.
The new policies are `never_v1`, `always_v1`, `random_v1`, and
`failure_triggered_v1`. Random policy seeds are derived from its configured
seed, task ID, and policy ID, so task scheduling does not alter a task's draw
sequence. The failure policy counts consecutive **previous** failed
environment actions and applies a cooldown after an executed selection.

## Development configuration

[`configs/textcraft_modular_development.json`](../configs/textcraft_modular_development.json)
is a development-only example. It pins Qwen2.5-7B-Instruct to revision
`a09a35458c702b33eeacc393d103063234e8bc28`, the full commit reported by
the [model repository API](https://huggingface.co/api/models/Qwen/Qwen2.5-7B-Instruct).
It uses the committed 20-task development manifest, checks overlap with the
committed held-out manifest, and selects the existing reconsideration support
with probability 0.25. It has not been run on a GPU. Review and, if needed,
replace the pinned revision before a new experiment; do not silently change
it inside a frozen run.

From the repository root, with the benchmark and CUDA dependencies installed:

```bash
PYTHONHASHSEED=0 python -m agent_support.runner \
  --config configs/textcraft_modular_development.json --limit 3
```

Remove `--limit` only for a separately identified full development run. The
config names the model adapter, exact revision, benchmark manifest, permitted
supports and versions, policy and parameters, seeds, action/model/support
limits, support-token budget, and output root. For a custom package, list its
Python import names under `component_modules`; the runner imports and hashes
those source files before registry validation.

`max_model_calls` and `max_support_calls` are enforced before work begins.
All action, model-call, support-call, and support-token limits are per episode.
For `max_support_tokens`, a model adapter must count the full chat-template
input. The runner reserves that input plus `max_new_tokens` before support
generation, then records actual input and output tokens afterward. If actual
usage exceeds the reservation, the run fails visibly. Retrieval-only support
can consume a support call without generating tokens. `compute_cost` records
tokens and inference/retrieval seconds; no monetary price is inferred for
local GPU inference.

## Add a model

Create an importable module and call `register_model("your_model", factory)`
from `agent_support.model` in it. The factory receives the config and returns
an adapter with:

```python
def generate(self, messages: list[dict], max_new_tokens: int | None = None) -> dict:
    # Return action-bearing text and measured usage.
    return {"text": final_text, "input_tokens": input_count,
            "output_tokens": output_count, "inference_seconds": elapsed}

def metadata(self) -> dict:
    return {"model_id": exact_id, "model_revision": exact_weights_revision}

def count_messages_tokens(self, messages: list[dict]) -> int:
    # Count the full rendered chat template when support-token budgets are used.
    ...

def count_tokens(self, text: str) -> int:
    # Used by trajectory-memory prompt trimming.
    ...
```

The adapter owns its tokenizer, chat template, loading and device placement,
decoding settings, token accounting, and any reasoning/final-answer split.
`text` must contain the final answer for the benchmark parser. The runner
checks `metadata()` against the configured identifier and revision before
inference. A future GPT-OSS adapter can use this contract without changing
the episode loop. Add the module to `component_modules` and set `model_type`,
`model_id`, and `model_revision` in a new config.

## Add a per-decision support

Create an adapter with `support_id`, `phase` (`before_proposal` or
`after_proposal`), `version`, `requirements`, and
`invoke(context, messages, action) -> SupportResult`. Register its factory:

```python
from agent_support.supports import SupportResult, register_support

class MySupport:
    support_id = "my_support"
    phase = "after_proposal"
    version = "my_support_v1"
    requirements = ("valid_proposal",)

    def invoke(self, context, messages, action):
        revision_prompt = messages + [{"role": "user", "content": "Recheck the action."}]
        event = {"mechanism": self.support_id, "proposed_action": action,
                 "action_changed": False}
        return SupportResult(messages, event, calls=1,
                             revision_messages=revision_prompt)

register_support("my_support", lambda config, model, resources: MySupport())
```

Return a modified `messages` value for a pre-proposal support. Return
`revision_messages=None` when no extra model call is needed. An adapter that
reads an external artifact must validate its compatibility and expose
`provenance()` with content hashes; the runner verifies this on restart. List
the support in `supports`, pin its `support_versions` entry, and refer to its
ID from a policy. No runner edit is needed. The CPU
[`extensibility test`](../tests/test_framework_extensibility.py) loads a
temporary module with a model, support, and policy and completes a paired run.

## Add a policy, benchmark, or controller

For a policy, implement `policy_id` and `select(context) -> SelectionDecision`,
then call `register_policy(id, class_or_factory)`. The factory receives the
policy config and task ID. Return an ordered tuple of support IDs in the
decision; the runner checks registration, phase, and availability. Add the
policy ID and parameters under `selection_policy` in a config. The policy
should read only `TurnContext`, never the environment or future outcome.

For a benchmark, call `register_benchmark(id, episode_factory, driver,
manifest_validator)`. The driver supplies `prompt`, `parse`, `stop_reason`,
and `parse_error_ends_episode`. An episode supplies `reset`, `step`, `close`,
`observation`, `initial_observation`, and `task_prompt`. Custom manifest tasks
include a stable `task_fingerprint` for overlap audits. The CPU extensibility
test runs a registered benchmark end to end.

For a recursive or multi-step method, call `register_controller(id, runner,
validator)`.
Its runner receives `(task, config, model, episode_factory)` and returns the
common episode metrics plus method-specific fields. A schema-v2 controller
must provide its own budget and compatibility validator before inference.
ADaPT uses the controller route in its existing development config and keeps
its original planner/action limits; it is not squeezed into the per-action
support contract.

## Checkpoints and comparison

Every run gets a unique directory containing `config.json`, `manifest.json`,
`run.json`, `summary.json`, `checkpoint.json`, and immutable episode JSON
files. `run.json` records the Git commit and dirty-state hashes, source and
snapshot hashes, model revision and runtime, prompt/support versions, seeds,
dependency versions, policy parameters, limits, and restart history. An
episode is written atomically and cannot replace a prior result.

For a stable Slurm job key, the first invocation creates a directory; a
restart must pass the same key with `--resume`:

```bash
PYTHONHASHSEED=0 python -m agent_support.runner \
  --config configs/textcraft_modular_development.json \
  --run-key dev_001
PYTHONHASHSEED=0 python -m agent_support.runner \
  --config configs/textcraft_modular_development.json \
  --run-key dev_001 --resume
```

Recovery rejects completed runs, changed code/config/manifests/artifacts,
unexpected or modified episode files, and different runtime metadata. It
verifies saved episodes before skipping them. If an episode was published just
before checkpoint metadata was updated, recovery logs it as an
`uncheckpointed_episodes` after validation. A run interrupted during an episode
restarts that episode; its partial in-memory trajectory is not claimed as a
checkpoint.

On Locust, submit [`scripts/locust_modular_job.sh`](../scripts/locust_modular_job.sh)
with your allocation's account, partition, GPU, memory, and time flags. Set
`AGENT_SUPPORT_DATA`, `AGENT_SUPPORT_CONFIG`, and `VENV_PATH` as needed before
`sbatch`. The script uses the Slurm job ID as the stable key, requests a
60-second termination signal, and verifies any existing run before resuming.
It does not launch a job by itself.

Compare two completed schema-v2 runs on the same manifest with:

```bash
python -m agent_support.analysis LEFT_RUN RIGHT_RUN --modular-comparison
```

The report includes task-level outcomes, actions, calls, tokens, support
invocations, and policy triggers, plus aggregate and paired helped/hurt
counts. The original episode traces retain every structured policy decision.
Historical schema-v1 analysis commands keep their existing behavior.

Identical seeds and greedy decoding do not guarantee bitwise-identical GPU
results across GPU types, library versions, process restarts, or nondeterministic
kernels. The framework records the relevant provenance and detects incompatible
restarts; it does not claim bitwise replay.
