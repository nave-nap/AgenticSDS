"""CPU tests for the first prompted controller and its safe capabilities."""

import io
import json
from pathlib import Path

import pytest

from agent_support.autonomous_support import (SupportContext, make_support_controller,
                                              make_support_executor)
from agent_support.common import validate_config
from agent_support.gpt_oss_controller_worker import serve
from agent_support.llm_support_controller import parse_controller_decision
from agent_support.runner import run, run_episode


MANIFEST = Path(__file__).resolve().parents[1] / "manifests/textcraft/development.json"
CONFIG = Path(__file__).resolve().parents[1] / "configs/textcraft_prompted_gptoss_controller.json"


def research_config():
    return json.loads(CONFIG.read_text())


def textcraft_task(seed):
    return next(item for item in json.loads(MANIFEST.read_text())["tasks"] if item["seed"] == seed)


class CountedControllerModel:
    max_new_tokens = 1024

    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.messages = []
        self.limits = []

    def count_messages_tokens(self, messages):
        return 50

    def generate(self, messages, max_new_tokens=None):
        self.messages.append(messages)
        self.limits.append(max_new_tokens)
        output = next(self.outputs)
        return {"text": output, "input_tokens": 50, "output_tokens": 20,
                "inference_seconds": 0.25, "generation_status": "complete"}


class TaskModel:
    def __init__(self, action="Action: get 1 spruce logs"):
        self.action = action
    def generate(self, messages):
        return {"text": self.action, "input_tokens": 20, "output_tokens": 7,
                "inference_seconds": 0.1}


class Episode:
    def __init__(self, task):
        self.item = task
        self.initial_observation = self.observation = self.task_prompt = ""
        self.actions = []
        self.executed = []
    def reset(self):
        self.initial_observation = self.observation = self.task_prompt = self.item["initial_observation"]
    def step(self, action):
        self.executed.append(action)
        self.observation = "Got 1 spruce logs"
        return {"observation": self.observation, "done": True, "success": True,
                "failed_action": False}
    def close(self):
        pass


def run_with_controller(outputs, *, config=None, action="Action: get 1 spruce logs"):
    settings = config or research_config()
    settings["max_decisions"] = settings["max_actions"] = 1
    task = textcraft_task(130)
    episode = Episode(task)
    controller_model = CountedControllerModel(outputs)
    result = run_episode("textcraft", task, settings, TaskModel(action),
                         episode_factory=lambda *_: episode, controller_model=controller_model)
    return result, episode, controller_model


def decision(kind, **fields):
    return json.dumps({"kind": kind, **fields})


def test_realistic_textcraft_states_expose_only_visible_context():
    cases = [(130, "Could not find planks", "get 6 planks"),
             (72, "Could not find a valid recipe", "craft 4 planks using 1 logs"),
             (51, "Got 1 lapis lazuli", "get 1 lapis lazuli")]
    settings = research_config()
    model = CountedControllerModel([decision("execute", action=None, reason="proceed")] * len(cases))
    controller = make_support_controller(settings, model)
    executor = make_support_executor(settings, model)
    for seed, feedback, prior_action in cases:
        task = textcraft_task(seed)
        context = SupportContext("textcraft", task["task_id"], task["initial_observation"],
                                 feedback, (), ((prior_action, feedback),), 2,
                                 "get 1 spruce logs", "get 1 spruce logs",
                                 capabilities=executor.catalog(),
                                 remaining_actions=29, remaining_decisions=29,
                                 remaining_budget={"operations_task": 6, "additional_tokens": 80000})
        controller.decide(context)
        prompt = json.loads(model.messages[-1][-1]["content"])
        assert "task_id" not in prompt
        assert task["initial_observation"] == prompt["task_prompt"]
        assert prompt["history"][-1] == {"action": prior_action, "observation": feedback}
        assert {item["operation"] for item in prompt["capabilities"]} >= {
            "inspect_history_v1", "reason_over_artifacts_v1"}
        reasoning = next(item for item in prompt["capabilities"]
                         if item["operation"] == "reason_over_artifacts_v1")
        assert reasoning["model_calls_per_use"] == 1
        assert reasoning["max_output_tokens_per_use"] == 256
        assert prompt["remaining_actions"] == 29


def test_strict_parser_rejects_prose_unknown_operations_and_extra_fields():
    valid = decision("request", operation="inspect_history_v1", arguments={"limit": 2},
                     objective="review feedback", reason="uncertain")
    assert parse_controller_decision(valid, {"inspect_history_v1"}).request.arguments == {"limit": 2}
    for raw in ("```json\n" + valid + "\n```", valid + " extra",
                decision("request", operation="shell", arguments={}, objective="run", reason="x"),
                decision("execute", action=None, reason="ok", extra=1),
                decision(["execute"], action=None, reason="ok")):
        with pytest.raises(ValueError):
            parse_controller_decision(raw, {"inspect_history_v1"})


def test_immediate_execute_uses_no_support_and_counts_controller():
    result, episode, model = run_with_controller([
        decision("execute", action=None, reason="current action is sufficient")])
    assert episode.executed == ["get 1 spruce logs"]
    assert result["support_calls"] == 0 and result["controller_model_calls"] == 1
    assert result["trace"][0]["support_trace"]["termination_reason"] == "execute"
    assert model.limits == [1024]


def test_one_support_then_execute_with_controller_action():
    outputs = [
        decision("request", operation="inspect_candidate_v1",
                 arguments={"action": "get 1 spruce logs"}, objective="check candidate", reason="check"),
        decision("execute", action="Action: get 1 oak logs", reason="choose an alternative"),
    ]
    result, episode, model = run_with_controller(outputs)
    assert episode.executed == ["get 1 oak logs"]
    assert result["support_calls"] == 1 and result["controller_model_calls"] == 2
    assert len(model.messages) == 2
    assert '"format_valid":true' in model.messages[1][-1]["content"]


def test_multiple_supports_reassess_after_result_and_share_model():
    outputs = [
        decision("request", operation="inspect_history_v1", arguments={"limit": 2},
                 objective="review recent feedback", reason="need context"),
        decision("request", operation="reason_over_artifacts_v1",
                 arguments={"objective": "assess the next useful resource action",
                            "output_type": "candidate_action"},
                 objective="check next action", reason="an assessment may help"),
        json.dumps({"output_type": "candidate_action", "content": "Action: get 1 oak logs"}),
        decision("execute", action=None, reason="candidate is usable"),
    ]
    result, episode, model = run_with_controller(outputs)
    assert episode.executed == ["get 1 oak logs"]
    events = result["trace"][0]["support_trace"]["events"]
    assert [item["request"]["operation"] for item in events if "request" in item] == [
        "inspect_history_v1", "reason_over_artifacts_v1"]
    assert '"status":"empty"' in model.messages[1][-1]["content"]
    assert '"candidate_action"' in model.messages[3][-1]["content"]
    assert result["controller_model_calls"] == 3 and result["support_model_calls"] == 1
    assert result["model_calls"] == 5 and model.limits == [1024, 1024, 256, 1024]


def test_support_result_can_change_next_controller_choice():
    class ConditionalModel(CountedControllerModel):
        def __init__(self):
            super().__init__([])
        def generate(self, messages, max_new_tokens=None):
            state = json.loads(messages[-1]["content"])
            if not state["results"]:
                raw = decision("request", operation="inspect_candidate_v1",
                               arguments={"action": "craft 6 planks using 1 spruce logs"},
                               objective="check recipe", reason="uncertain")
            elif len(state["results"]) == 1 and state["results"][-1]["information"]["displayed_recipe"] is False:
                raw = decision("request", operation="inspect_history_v1", arguments={"limit": 1},
                               objective="review feedback", reason="recipe not displayed")
            else:
                raw = decision("execute", action=None, reason="proceed")
            self.messages.append(messages)
            return {"text": raw, "input_tokens": 50, "output_tokens": 20,
                    "inference_seconds": 0.25, "generation_status": "complete"}

    settings = research_config()
    settings["max_actions"] = settings["max_decisions"] = 1
    task = textcraft_task(130)
    episode = Episode(task)
    model = ConditionalModel()
    result = run_episode("textcraft", task, settings, TaskModel(),
                         episode_factory=lambda *_: episode, controller_model=model)
    requests = [item["request"]["operation"] for item in
                result["trace"][0]["support_trace"]["events"] if "request" in item]
    assert requests == ["inspect_candidate_v1", "inspect_history_v1"]
    assert result["trace"][0]["support_trace"]["termination_reason"] == "execute"


def test_budget_exhaustion_executes_original_valid_action():
    settings = research_config()
    settings["autonomous_support"]["budget"]["max_additional_tokens"] = 1000
    result, episode, model = run_with_controller([], config=settings)
    assert episode.executed == ["get 1 spruce logs"]
    assert result["trace"][0]["support_trace"]["termination_reason"] == "token_budget"
    assert result["controller_model_calls"] == 0 and model.messages == []


def test_malformed_output_recovers_and_charges_controller_cost():
    result, episode, model = run_with_controller(["not json"])
    assert episode.executed == ["get 1 spruce logs"]
    event = result["trace"][0]["support_trace"]["events"][0]
    assert event["controller_metadata"]["parse_error"]
    assert event["controller_cost"]["output_tokens"] == 20
    assert result["controller_model_calls"] == 1
    assert result["trace"][0]["support_trace"]["termination_reason"] == "malformed_controller_output"


def test_controller_backend_error_recovers_with_reserved_input_count():
    class BrokenModel(CountedControllerModel):
        def __init__(self):
            super().__init__([])
        def generate(self, messages, max_new_tokens=None):
            raise RuntimeError("worker disconnected")

    settings = research_config()
    settings["max_actions"] = settings["max_decisions"] = 1
    item = textcraft_task(130)
    episode = Episode(item)
    result = run_episode("textcraft", item, settings, TaskModel(),
                         episode_factory=lambda *_: episode, controller_model=BrokenModel())
    assert episode.executed == ["get 1 spruce logs"]
    assert result["controller_model_calls"] == 1
    assert result["controller_input_tokens"] == 50
    assert result["trace"][0]["support_trace"]["termination_reason"] == "controller_backend_error"


def test_run_shares_one_controller_model_across_tasks(tmp_path):
    settings = research_config()
    settings["max_actions"] = settings["max_decisions"] = 1
    tasks = [textcraft_task(130), textcraft_task(72)]
    manifest = {"schema_version": 1, "benchmark": "textcraft",
                "partition": "development", "tasks": tasks}
    model = CountedControllerModel([
        decision("execute", action=None, reason="proceed"),
        decision("execute", action=None, reason="proceed")])
    directory = run(settings, manifest, tmp_path, model=TaskModel(),
                    episode_factory=lambda benchmark, task, config: Episode(task),
                    controller_model=model)
    summary = json.loads((directory / "summary.json").read_text())
    assert len(model.messages) == 2
    assert summary["controller_model_calls"] == 2
    assert summary["tasks"] == 2


def test_config_and_worker_json_protocol_with_fake_model():
    settings = research_config()
    validate_config(settings, "textcraft")
    class FakeWorkerModel:
        def __init__(self, config):
            self.config = config
        def metadata(self):
            return {"model_type": "gpt_oss_20b"}
        def count_messages_tokens(self, messages):
            return 3
        def generate(self, messages, max_new_tokens=None):
            return {"text": "{}", "input_tokens": 3, "output_tokens": 2,
                    "inference_seconds": 0.01}
    requests = [
        {"op": "init", "config": {"model_id": "fake"}},
        {"op": "count_messages_tokens", "messages": [{"role": "user", "content": "x"}]},
        {"op": "generate", "messages": [{"role": "user", "content": "x"}], "max_new_tokens": 9},
        {"op": "shutdown"},
    ]
    source = io.StringIO("".join(json.dumps(item) + "\n" for item in requests))
    sink = io.StringIO()
    serve(source, sink, model_factory=FakeWorkerModel)
    responses = [json.loads(line) for line in sink.getvalue().splitlines()]
    assert len(responses) == 4 and all(item["ok"] for item in responses)
    assert responses[1]["count"] == 3 and responses[2]["generation"]["text"] == "{}"
