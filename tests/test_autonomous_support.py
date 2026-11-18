"""CPU contracts for sequential support without an LLM controller or GPU."""

import json

import pytest

from agent_support.autonomous_support import (
    ControllerResponse, SupportBudget, SupportContext, SupportCost, SupportDecision,
    SupportExecutor, SupportRequest, SupportReservation, SupportResult,
    run_support_loop,
)
from agent_support.common import digest_text, validate_config
from agent_support.runner import run, run_episode


def config(**budget_changes):
    budget = {"max_operations_per_decision": 4, "max_operations_per_task": 8,
              "max_additional_model_calls": 20, "max_additional_tokens": 200,
              "max_depth": 6}
    budget.update(budget_changes)
    return {"benchmark": "textcraft", "condition": "autonomous_support_v1",
            "model_id": "fake", "quantization": "nf4_4bit", "max_actions": 2,
            "max_decisions": 2, "history_window": 2, "max_new_tokens": 10,
            "autonomous_support": {"controller": "execute_immediately_v1",
                                   "operations": [], "budget": budget}}


def task():
    observation = "Crafting commands:\ncraft 1 plank using 1 wood\n\nGoal: craft plank."
    return {"task_id": "textcraft:seed:42", "seed": 42,
            "initial_observation": observation, "observation_sha256": digest_text(observation)}


class Episode:
    def __init__(self, item):
        self.item = item
        self.initial_observation = self.observation = self.task_prompt = ""
        self.actions = []
        self.executed = []
        self.closed = False

    def reset(self):
        self.initial_observation = self.observation = self.task_prompt = self.item["initial_observation"]

    def step(self, action):
        self.executed.append(action)
        self.observation = "Done" if action == "get 1 wood" else "Could not craft"
        return {"observation": self.observation, "done": True,
                "success": action == "get 1 wood", "failed_action": action != "get 1 wood"}

    def close(self):
        self.closed = True


class Model:
    def __init__(self, output="Action: get 1 stone"):
        self.output = output
        self.calls = 0

    def generate(self, messages):
        self.calls += 1
        return {"text": self.output, "input_tokens": 10,
                "output_tokens": 4, "inference_seconds": 0.1}


class Controller:
    controller_id = "mock_controller"

    def __init__(self, decide):
        self.choose = decide
        self.contexts = []

    def reservation(self, context):
        return SupportReservation(model_calls=1, input_tokens=2, output_tokens=1)

    def decide(self, context):
        self.contexts.append(context.as_dict())
        return ControllerResponse(self.choose(context), SupportCost(1, 2, 1, 0.02))


class Operation:
    def __init__(self, name, handler, reservation=None):
        self.operation_id = name
        self.handler = handler
        self.maximum = reservation or SupportReservation()
        self.calls = []

    def validate_arguments(self, arguments, context):
        if set(arguments) != {"query"} or not isinstance(arguments["query"], str):
            raise ValueError("query must be text")

    def reservation(self, arguments, context):
        return self.maximum

    def execute(self, arguments, context):
        self.calls.append((arguments.copy(), len(context.results)))
        return self.handler(arguments, context)


def execute(reason="ready"):
    return SupportDecision("execute", reason=reason)


def request(name, query):
    return SupportDecision("request", SupportRequest(name, {"query": query}, "inspect the task"))


def episode_run(controller=None, executor=None, *, settings=None, output="Action: get 1 stone"):
    item = task()
    episode = Episode(item)
    result = run_episode("textcraft", item, settings or config(), Model(output),
                         episode_factory=lambda *_: episode,
                         support_controller=controller, support_executor=executor)
    return result, episode


def test_no_support_executes_original_action_and_logs_stop():
    result, episode = episode_run()
    assert episode.executed == ["get 1 stone"]
    assert result["model_calls"] == result["proposal_model_calls"] == 1
    assert result["support_calls"] == result["controller_model_calls"] == 0
    assert result["trace"][0]["support_trace"]["termination_reason"] == "execute"
    assert result["trace"][0]["support_trace"]["environment_outcome"]["failed_action"] is True
    assert episode.closed


def test_one_support_can_revise_action_and_costs_are_separate():
    controller = Controller(lambda state: request("recommend", "wood") if not state.results else execute())
    operation = Operation("recommend", lambda args, state: SupportResult(
        "ok", {"recommendation": "get wood"}, proposed_action="Action: get 1 wood",
        cost=SupportCost(1, 3, 2, 0.04), metadata={"source": "mock"}),
        SupportReservation(1, 3, 2))
    result, episode = episode_run(controller, SupportExecutor({"recommend": operation}))
    assert episode.executed == ["get 1 wood"] and result["success"]
    assert result["support_calls"] == 1 and result["controller_model_calls"] == 2
    assert result["support_model_calls"] == 1 and result["model_calls"] == 4
    assert (result["proposal_input_tokens"], result["controller_input_tokens"],
            result["support_input_tokens"], result["input_tokens"]) == (10, 4, 3, 17)
    step = result["trace"][0]
    assert step["support_trace"]["events"][0]["action_revision"] == {
        "from": "get 1 stone", "to": "get 1 wood"}
    assert step["support_trace"]["events"][1]["controller_decision"]["kind"] == "execute"
    assert step["support_trace"]["environment_outcome"]["success"] is True
    assert "operation_elapsed_seconds" in step["support_trace"]["events"][0]
    assert "controller_elapsed_seconds" in step["support_trace"]["events"][0]
    assert "environment_seconds" in step


def test_multiple_different_supports_are_chosen_from_previous_result():
    def choose(state):
        if not state.results:
            return request("inspect", "recent failure")
        if len(state.results) == 1 and state.results[0].information["need"] == "revision":
            return request("revise", "wood")
        return execute()

    controller = Controller(choose)
    inspect = Operation("inspect", lambda args, state: SupportResult("ok", {"need": "revision"}))
    revise = Operation("revise", lambda args, state: SupportResult(
        "ok", {"plan": "gather wood"}, proposed_action="Action: get 1 wood"))
    result, episode = episode_run(controller, SupportExecutor({"inspect": inspect, "revise": revise}))
    assert episode.executed == ["get 1 wood"]
    assert [event["request"]["operation"] for event in result["trace"][0]["support_trace"]["events"]
            if "request" in event] == ["inspect", "revise"]
    assert len(controller.contexts[1]["results"]) == 1
    assert controller.contexts[1]["results"][0]["information"] == {"need": "revision"}
    assert controller.contexts[2]["current_action"] == "get 1 wood"


def test_same_operation_may_repeat_with_different_arguments():
    def choose(state):
        if len(state.results) < 2:
            return request("search", f"query-{len(state.results)}")
        return execute()

    operation = Operation("search", lambda args, state: SupportResult("ok", args["query"]))
    result, _ = episode_run(Controller(choose), SupportExecutor({"search": operation}))
    assert operation.calls == [({"query": "query-0"}, 0), ({"query": "query-1"}, 1)]
    assert result["support_calls"] == 2


def test_operation_budget_is_shared_across_environment_decisions():
    class TwoStepEpisode(Episode):
        def step(self, action):
            self.executed.append(action)
            self.observation = "Continue" if len(self.executed) == 1 else "Done"
            return {"observation": self.observation, "done": len(self.executed) == 2,
                    "success": len(self.executed) == 2, "failed_action": False}

    item = task()
    episode = TwoStepEpisode(item)
    controller = Controller(lambda state: request("inspect", "state"))
    operation = Operation("inspect", lambda args, state: SupportResult("ok", "seen"))
    settings = config(max_operations_per_task=1, max_operations_per_decision=1)
    result = run_episode("textcraft", item, settings, Model(),
                         episode_factory=lambda *_: episode, support_controller=controller,
                         support_executor=SupportExecutor({"inspect": operation}))
    assert result["support_calls"] == 1 and len(episode.executed) == 2
    assert [record["support_trace"]["termination_reason"] for record in result["trace"]] == [
        "operation_budget", "operation_budget"]


@pytest.mark.parametrize("bad_request", [
    SupportRequest("unregistered", {"query": "wood"}),
    SupportRequest("search", {"unexpected": "wood"}),
])
def test_invalid_request_falls_back_to_original_action(bad_request):
    controller = Controller(lambda state: SupportDecision("request", bad_request))
    operation = Operation("search", lambda args, state: SupportResult("ok", "unreachable"))
    result, episode = episode_run(controller, SupportExecutor({"search": operation}))
    assert episode.executed == ["get 1 stone"] and not operation.calls
    assert result["support_calls"] == 0
    assert result["trace"][0]["support_trace"]["termination_reason"] == "invalid_request"


@pytest.mark.parametrize("limit,expected", [
    ({"max_operations_per_decision": 1}, "operation_budget"),
    ({"max_operations_per_task": 1}, "operation_budget"),
    ({"max_additional_model_calls": 1}, "model_call_budget"),
    ({"max_additional_tokens": 3}, "token_budget"),
    ({"max_depth": 1}, "depth_budget"),
])
def test_budget_exhaustion_is_explicit_and_never_overruns(limit, expected):
    controller = Controller(lambda state: request("search", f"q{len(state.results)}"))
    operation = Operation("search", lambda args, state: SupportResult("ok", args["query"]))
    result, episode = episode_run(controller, SupportExecutor({"search": operation}),
                                  settings=config(**limit))
    assert episode.executed == ["get 1 stone"]
    assert result["trace"][0]["support_trace"]["termination_reason"] == expected
    assert result["support_calls"] <= min(limit.get("max_operations_per_decision", 4),
                                         limit.get("max_operations_per_task", 8))
    assert result["support_budget"]["used"]["additional_tokens"] <= limit.get("max_additional_tokens", 200)


def test_support_failure_after_revision_restores_original_validated_action():
    def choose(state):
        return request("revise", "wood") if not state.results else request("broken", "check")

    revise = Operation("revise", lambda args, state: SupportResult(
        "ok", "new candidate", proposed_action="Action: get 1 wood"))
    def fail(args, state):
        raise RuntimeError("support failed")
    broken = Operation("broken", fail)
    result, episode = episode_run(Controller(choose), SupportExecutor({"revise": revise, "broken": broken}))
    assert episode.executed == ["get 1 stone"]
    support_trace = result["trace"][0]["support_trace"]
    assert support_trace["termination_reason"] == "support_failure"
    assert support_trace["events"][1]["result"]["status"] == "error"
    assert support_trace["final_action"] == "get 1 stone"


def test_operation_cost_above_reservation_fails_closed():
    controller = Controller(lambda state: request("expensive", "wood"))
    operation = Operation("expensive", lambda args, state: SupportResult(
        "ok", "candidate", proposed_action="Action: get 1 wood",
        cost=SupportCost(1, 10, 2)), SupportReservation(1, 2, 2))
    result, episode = episode_run(controller, SupportExecutor({"expensive": operation}))
    assert episode.executed == ["get 1 stone"]
    assert result["trace"][0]["support_trace"]["termination_reason"] == "accounting_error"
    assert result["support_calls"] == 0


def test_invalid_revision_and_invalid_original_never_step_environment():
    controller = Controller(lambda state: request("revise", "bad") if not state.results else execute())
    operation = Operation("revise", lambda args, state: SupportResult("ok", "bad", proposed_action="rm -rf /"))
    settings = config()
    settings["max_decisions"] = 1
    result, episode = episode_run(controller, SupportExecutor({"revise": operation}),
                                  settings=settings, output="invalid model output")
    assert episode.executed == [] and result["parser_errors"] == 1
    assert result["trace"][0]["support_trace"]["termination_reason"] == "support_failure"
    assert result["trace"][0]["support_trace"]["environment_outcome"]["executed"] is False


def test_persisted_new_run_and_legacy_config_compatibility(tmp_path):
    item = task()
    manifest = {"schema_version": 1, "benchmark": "textcraft",
                "partition": "development", "tasks": [item]}
    new_config = config()
    output = run(new_config, manifest, tmp_path / "new", model=Model("Action: get 1 wood"),
                 episode_factory=lambda *_: Episode(item))
    saved = json.loads(next((output / "episodes").glob("*.json")).read_text())
    metadata = json.loads((output / "run.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    assert metadata["autonomous_support"]["controller"] == "execute_immediately_v1"
    assert saved["trace"][0]["support_trace"]["termination_reason"] == "execute"
    assert summary["controller_model_calls"] == summary["support_calls"] == 0
    legacy = {key: value for key, value in new_config.items() if key != "autonomous_support"}
    legacy["condition"] = "baseline"
    validate_config(legacy, "textcraft")
    old_output = run(legacy, manifest, tmp_path / "old", model=Model("Action: get 1 wood"),
                     episode_factory=lambda *_: Episode(item))
    old_saved = json.loads(next((old_output / "episodes").glob("*.json")).read_text())
    assert old_saved["success"] and "support_trace" not in old_saved["trace"][0]


def test_alfworld_revision_uses_current_numbered_action_list():
    class AlfEpisode:
        def __init__(self):
            self.initial_observation = self.observation = "Your task is to: open drawer"
            self.task_prompt = "open drawer"
            self.actions = ["look", "open drawer"]
            self.executed = []
        def reset(self):
            pass
        def step(self, action):
            self.executed.append(action)
            return {"observation": "Done", "done": True, "success": True, "failed_action": False}
        def close(self):
            pass

    settings = config()
    settings["benchmark"] = "alfworld"
    settings["max_actions"] = settings["max_decisions"] = 1
    controller = Controller(lambda state: request("recommend", "drawer") if not state.results else execute())
    operation = Operation("recommend", lambda args, state: SupportResult(
        "ok", "choose second admissible action", proposed_action="2"))
    episode = AlfEpisode()
    result = run_episode("alfworld", {"task_id": "alfworld:one"}, settings, Model("1"),
                         episode_factory=lambda *_: episode, support_controller=controller,
                         support_executor=SupportExecutor({"recommend": operation}))
    assert episode.executed == ["open drawer"] and result["success"]
    assert result["trace"][0]["support_trace"]["events"][0]["action_revision"]["to"] == "open drawer"


def test_malformed_controller_response_and_reserved_cost_fail_closed():
    context = SupportContext("textcraft", "task", "craft plank", "start", (), (), 1,
                             "get 1 stone", "get 1 stone")
    class Malformed:
        controller_id = "malformed"
        def reservation(self, state):
            return SupportReservation()
        def decide(self, state):
            return {"kind": "request", "operation": "anything"}
    trace = run_support_loop(Malformed(), SupportExecutor(), SupportBudget(1, 1, 0, 0, 2),
                             context, lambda value: value)
    assert trace.termination_reason == "controller_error"
    assert trace.final_action == "get 1 stone"

    class Underreserved(Malformed):
        def decide(self, state):
            return ControllerResponse(execute(), SupportCost(1, 1, 1))
    trace = run_support_loop(Underreserved(), SupportExecutor(), SupportBudget(1, 1, 2, 10, 2),
                             context, lambda value: value)
    assert trace.termination_reason == "controller_error"
    assert trace.final_action == "get 1 stone"
