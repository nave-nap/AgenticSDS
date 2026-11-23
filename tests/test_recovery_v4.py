from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_support.autonomous_support import (ControllerResponse, SupportBudget,
    SupportContext, SupportDecision, SupportExecutor, SupportRequest,
    SupportReservation, make_support_controller, run_support_loop)
from agent_support.autonomous_operations import ReasonOverArtifacts
from agent_support.controller_backend import (ControllerWorkerFatalCUDA,
    ControllerWorkerOOM, ControllerWorkerUnavailable)
from agent_support.policy import parse_textcraft
from agent_support.recovery_v4 import (AnalyzeFailures, RecoveryState, VerifyPrerequisites,
    SYSTEM, adaptive_loop, parse_decision, recovery_executor, run_recovery_episode)
from agent_support.recovery_v4_eval import config_for, manifest

REAL_V4 = {item["label"]: item for item in json.loads((Path(__file__).parent /
    "fixtures/recovery_v4_failed_controller_fixtures.json").read_text())["cases"]}


def choice(kind, **kwargs):
    value = {"kind": kind, "reason": "visible evidence", "state_update": {},
             "progress_assessment": None, "progress_evidence": "", "support_assessment": None}
    value.update(kwargs)
    return json.dumps(value)


class Gpt:
    max_new_tokens = 1024

    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.messages = []

    def count_messages_tokens(self, messages):
        return 20

    def generate(self, messages, max_new_tokens=None):
        self.messages.append(messages)
        return {"text": next(self.outputs), "raw_completion": "full Harmony",
                "generation_status": "complete", "input_tokens": 20,
                "output_tokens": 10, "inference_seconds": 0.1}


class ErrorGpt(Gpt):
    def generate(self, messages, max_new_tokens=None):
        value = next(self.outputs)
        if isinstance(value, Exception):
            raise value
        self.messages.append(messages)
        return {"text": value, "raw_completion": "full Harmony",
                "generation_status": "complete", "input_tokens": 20,
                "output_tokens": 10, "inference_seconds": 0.1}


class Qwen:
    def generate(self, messages):
        return {"text": "Action: inventory", "input_tokens": 10,
                "output_tokens": 4, "inference_seconds": 0.1}


class Episode:
    def __init__(self):
        self.initial_observation = "Crafting commands:\ncraft 1 plank using 1 log\n\nGoal: craft plank."
        self.task_prompt = self.initial_observation
        self.observation = self.initial_observation
        self.steps = []

    def reset(self):
        pass

    def step(self, action):
        self.steps.append(action)
        self.observation = "Goal completed" if action == "get 1 log" else "Could not craft"
        return {"observation": self.observation, "done": action == "get 1 log",
                "success": action == "get 1 log", "failed_action": action != "get 1 log"}

    def close(self):
        pass


def factory(*_):
    return Episode()


def test_manifest_and_budget_are_frozen():
    selected = manifest()
    assert len(selected["tasks"]) == 30
    assert [task["seed"] for task in selected["tasks"][20:]] == list(range(200, 210))
    for arm in ("expanded_v3_v1", "autonomous_recovery_v4"):
        config = config_for(arm)
        assert (config["max_actions"], config["max_decisions"]) == (150, 220)
        assert config["limits"] == {"wall_seconds": 2400, "inference_seconds": 2100,
                                     "max_model_calls_total": 720}
        assert config["autonomous_support"]["budget"]["max_operations_per_task"] == 100


def test_prompt_separates_environment_actions_from_support_requests():
    assert "Request is only for supports" in SYSTEM
    assert "get, craft, and inventory are environment actions" in SYSTEM
    assert "Compare a pending predicted_outcome with observed_feedback" in SYSTEM
    assert "update strategy or plan in state_update" in SYSTEM
    assert "Otherwise use null or omit" in SYSTEM


def test_strict_decision_and_bounded_interpretations():
    valid = choice("execute", action="inventory", expected_outcome="inventory feedback",
                   state_update={"strategy": "inspect visible inventory",
                                 "hypotheses": ["missing base item"]})
    assert parse_decision(valid, set())["state_update"]["hypotheses"] == ["missing base item"]
    with pytest.raises(ValueError):
        parse_decision('{"kind":"execute","kind":"execute"}', set())
    with pytest.raises(ValueError):
        parse_decision(choice("execute", action="inventory", expected_outcome="x",
                              state_update={"verified_facts": "secret"}), set())
    state = RecoveryState(objective="craft plank")
    changes = state.apply({"strategy": "gather log", "hypotheses": ["log may be needed"]})
    assert "strategy" in changes and state.version == 1
    assert "epistemic_status" in state.snapshot()


def test_visible_supports_do_not_query_environment():
    episode = Episode()
    context = SupportContext("textcraft", "dev", episode.task_prompt, "Could not craft", (),
                             (("craft 1 plank using 1 log", "Could not craft"),
                              ("craft 1 plank using 1 log", "Could not craft")),
                             1, "craft 1 plank using 1 log", "craft 1 plank using 1 log")
    failures = AnalyzeFailures().execute({"limit": 5}, context)
    assert failures.information["repeat_counts"]["craft 1 plank using 1 log"] == 2
    prerequisites = VerifyPrerequisites().execute({"action": "craft 1 plank using 1 log"}, context)
    assert prerequisites.information["displayed_recipe"] == "craft 1 plank using 1 log"
    assert prerequisites.information["last_inventory_feedback"] is None
    assert prerequisites.metadata["no_environment_query"] is True
    fresh = SupportContext("textcraft", "dev", episode.task_prompt, "Inventory: [log] (1)", (),
                           (("inventory", "Inventory: [log] (1)"),), 2,
                           "craft 1 plank using 1 log", "craft 1 plank using 1 log")
    verified = VerifyPrerequisites().execute({"action": "craft 1 plank using 1 log"}, fresh)
    assert verified.information["all_prerequisites_observed_sufficient"] is True
    assert verified.information["ingredients"] == [{"item": "log", "required": 1,
                                                  "observed": 1, "sufficient": True}]


def test_controller_chains_support_then_revises_without_fixed_trigger():
    gpt = Gpt([
        choice("request", operation="analyze_failures_v1", arguments={"limit": 3},
               uncertainty="why craft failed", expected_information="failure feedback",
               state_update={"strategy": "diagnose failure"}),
        choice("request", operation="verify_prerequisites_v1",
               arguments={"action": "craft 1 plank using 1 log"},
               uncertainty="whether prerequisites are available",
               expected_information="recipe and observed inventory",
               support_assessment={"useful": True, "resolved": True,
                                   "reason": "failure feedback identified missing ingredients"}),
        choice("execute", action="get 1 log", expected_outcome="collect log",
               support_assessment={"useful": True, "resolved": True,
                                   "reason": "recipe identifies the required log"},
               state_update={"strategy": "gather prerequisite"})])
    config = config_for("autonomous_recovery_v4")
    executor = recovery_executor({**config, "autonomous_support": {
        **config["autonomous_support"],
        "operations": ["analyze_failures_v1", "verify_prerequisites_v1"]}}, gpt)
    state = RecoveryState(objective="craft plank")
    context = SupportContext("textcraft", "dev", Episode().task_prompt, "Could not craft", (),
        (("craft 1 plank using 1 log", "Could not craft"),), 1,
        "craft 1 plank using 1 log", "craft 1 plank using 1 log", remaining_actions=150,
        remaining_decisions=220)
    budget = SupportBudget(7, 100, 500, 3000000, 10)
    trace, stop = adaptive_loop(gpt, executor, budget, context, state,
                                deadline=10**12, inference_limit=2100, inference_used=0)
    assert stop is None and trace.final_action == "get 1 log"
    assert trace.operation_count == 2 and budget.controller_calls_used == 3
    assert budget.operations_used == 2
    assert state.support_findings[-1]["assessment"]["resolved"] is True
    assert state.strategy == "gather prerequisite"
    assert len(gpt.messages) == 3


def test_episode_success_is_environment_verified_and_checkpointed():
    gpt = Gpt([choice("execute", action="get 1 log", expected_outcome="collect log",
                      state_update={"strategy": "gather first"})])
    config = config_for("autonomous_recovery_v4")
    config["max_actions"] = config["max_decisions"] = 1
    config["autonomous_support"]["operations"] = []
    checkpoints = []
    result = run_recovery_episode({"task_id": "dev", "initial_observation": Episode().initial_observation},
        config, Qwen(), gpt, episode_factory=factory, checkpoint=checkpoints.append)
    assert result["success"] is True and result["status"] == "success"
    assert result["environment_actions"] == 1
    assert (result["qwen_model_calls"], result["controller_model_calls"], result["support_calls"]) == (1, 1, 0)
    assert result["strategy_changes"] == 1
    assert checkpoints[-1]["environment_actions"] == 1
    assert result["trace"][0]["action"] == "get 1 log"


class Memory:
    max_top_k = 3
    max_example_tokens = 1200
    corpus_sha256 = "corpus"
    index_sha256 = "index"

    def __init__(self):
        self.queries = []

    def search(self, benchmark, query, scope, top_k):
        self.queries.append((benchmark, query, scope, top_k))
        return ([{"task_id": "prior", "summary": "gather log"}], 0.01)


@pytest.mark.parametrize("operation,arguments", [
    ("inspect_state_v1", {"scope": "all", "limit": 2}),
    ("inspect_history_v1", {"limit": 2, "scope": "recent"}),
    ("retrieve_episodic_memory_v1", {"query": "how to craft plank", "scope": "both", "top_k": 1}),
])
def test_compact_request_dispatches_selected_support(operation, arguments):
    memory = Memory()
    gpt = Gpt([
        json.dumps({"kind": "request", "reason": "resolve uncertainty",
                    "operation": operation, "arguments": arguments,
                    "state_update": {"strategy": "inspect evidence"}}),
        json.dumps({"kind": "execute", "reason": "act on evidence", "action": "get 1 log",
                    "support_assessment": {"useful": True, "resolved": True,
                                           "reason": "result supplied visible evidence"}}),
    ])
    config = config_for("autonomous_recovery_v4")
    config["autonomous_support"]["operations"] = [operation]
    executor = recovery_executor(config, gpt, memory)
    state = RecoveryState(objective="craft plank")
    context = SupportContext("textcraft", "dev", Episode().task_prompt, "Could not craft", (),
        (("craft 1 plank using 1 log", "Could not craft"),), 1,
        "inventory", "inventory", remaining_actions=150, remaining_decisions=220)
    budget = SupportBudget(7, 100, 500, 3000000, 10)
    trace, stop = adaptive_loop(gpt, executor, budget, context, state,
                                deadline=10**12, inference_limit=2100, inference_used=0)
    assert stop is None and trace.termination_reason == "execute"
    assert trace.final_action == "get 1 log" and trace.operation_count == 1
    assert budget.operations_used == 1 and budget.controller_calls_used == 2
    assert trace.events[0]["result"]["status"] in {"ok", "empty"}
    assert trace.events[0]["recovery_state_after"]["support_findings"][0]["operation"] == operation
    assert state.support_findings[0]["assessment"]["resolved"] is True
    assert state.strategy == "inspect evidence" and state.version == 1
    assert memory.queries == ([('textcraft', 'how to craft plank', 'both', 1)]
                              if operation == "retrieve_episodic_memory_v1" else [])


def test_progress_assessment_and_state_update_survive_episode_checkpoints():
    gpt = Gpt([
        choice("execute", action="inventory", expected_outcome="inventory appears",
               state_update={"plan": "inspect then gather"}),
        choice("execute", action="get 1 log", expected_outcome="log collected",
               progress_assessment="stalled", progress_evidence="inventory did not advance crafting",
               state_update={"plan": "gather log", "attempted_approaches": ["inspected inventory"]}),
    ])
    config = config_for("autonomous_recovery_v4")
    config["max_actions"] = config["max_decisions"] = 2
    config["autonomous_support"]["operations"] = []
    checkpoints = []
    result = run_recovery_episode({"task_id": "dev", "initial_observation": Episode().initial_observation},
        config, Qwen(), gpt, episode_factory=factory, checkpoint=checkpoints.append)
    assert result["success"] and result["environment_actions"] == 2
    assert result["progress_assessments"] == 1
    assert result["recovery_state_final"]["plan"] == "gather log"
    assert result["trace"][1]["support_trace"]["events"][0]["progress_assessment"]["assessment"] == "stalled"
    assert checkpoints[-1]["recovery_state"]["attempted_approaches"] == ["inspected inventory"]


@pytest.mark.parametrize("decision,detail", [
    ({"kind": "request", "reason": "look", "operation": "unknown_v1", "arguments": {}},
     "Unknown support operation"),
    ({"kind": "request", "operation": "get", "arguments": {"item": "cocoa beans"}},
     "Unknown support operation: 'get'"),
    ({"kind": "request", "reason": "look", "operation": "inspect_state_v1",
      "arguments": {"scope": "all", "limit": 0}, "state_update": {"strategy": "must not persist"}},
     "inspect_state requires"),
    ({"kind": "execute", "reason": "act", "action": "craft everything"},
     "Invalid native TextCraft action"),
])
def test_invalid_decision_falls_back_once_with_precise_detail(decision, detail):
    gpt = Gpt([json.dumps(decision)])
    config = config_for("autonomous_recovery_v4")
    config["max_actions"] = config["max_decisions"] = 1
    config["autonomous_support"]["operations"] = ["inspect_state_v1"]
    result = run_recovery_episode({"task_id": "dev", "initial_observation": Episode().initial_observation},
        config, Qwen(), gpt, episode_factory=factory)
    record = result["trace"][0]
    assert result["environment_actions"] == 1 and result["support_calls"] == 0
    assert result["controller_model_calls"] == 1 and result["controller_errors"] == 1
    assert result["controller_fallbacks"] == 1
    assert record["controller_fallback_reason"] == "controller_parse_error"
    assert detail in record["controller_fallback_detail"]
    assert record["action"] == "inventory"
    assert result["recovery_state_final"]["strategy"] == ""


def test_extra_closing_brace_is_parse_error_with_one_fallback():
    gpt = Gpt(['{"kind":"execute","action":"get 1 log"}}'])
    config = config_for("autonomous_recovery_v4")
    config["max_actions"] = config["max_decisions"] = 1
    config["autonomous_support"]["operations"] = []
    result = run_recovery_episode({"task_id": "dev", "initial_observation": Episode().initial_observation},
        config, Qwen(), gpt, episode_factory=factory)
    record = result["trace"][0]
    assert result["controller_model_calls"] == 1 and result["controller_errors"] == 1
    assert result["controller_fallbacks"] == 1 and result["support_calls"] == 0
    assert record["controller_fallback_reason"] == "controller_parse_error"
    assert "Extra data" in record["controller_fallback_detail"]
    assert record["action"] == "inventory"


@pytest.mark.parametrize("useful", [True, False])
def test_support_assessment_is_recorded_and_counted(useful):
    gpt = Gpt([
        choice("request", operation="inspect_state_v1", arguments={"scope": "recipes", "limit": 2},
               uncertainty="which recipe is visible", expected_information="displayed recipe"),
        choice("execute", action="get 1 log", expected_outcome="collect log",
               support_assessment={"useful": useful, "resolved": useful,
                                   "reason": "displayed recipe was inspected"}),
    ])
    config = config_for("autonomous_recovery_v4")
    config["max_actions"] = config["max_decisions"] = 1
    config["autonomous_support"]["operations"] = ["inspect_state_v1"]
    checkpoints = []
    result = run_recovery_episode({"task_id": "dev", "initial_observation": Episode().initial_observation},
        config, Qwen(), gpt, episode_factory=factory, checkpoint=checkpoints.append)
    assert result["success"] and result["support_calls"] == 1
    assert result["support_assessments"] == 1
    assert result["support_useful_judgments"] == int(useful)
    finding = result["recovery_state_final"]["support_findings"][0]
    assert finding["assessment"]["useful"] is useful
    assert checkpoints[-1]["recovery_state"]["support_findings"][0]["assessment"] == finding["assessment"]
    assert result["trace"][0]["support_trace"]["events"][1]["support_assessment"]["useful"] is useful


def test_only_revisions_of_existing_plan_count_as_replanning():
    gpt = Gpt([
        choice("execute", action="inventory", expected_outcome="inventory feedback",
               state_update={"plan": "inspect inventory", "strategy": "inspect"}),
        choice("execute", action="inventory", expected_outcome="inventory feedback",
               progress_assessment="stalled", progress_evidence="no material acquired",
               state_update={"plan": "inspect inventory", "strategy": "inspect"}),
        choice("execute", action="inventory", expected_outcome="inventory feedback",
               progress_assessment="stalled", progress_evidence="still no material",
               state_update={"plan": "gather log", "strategy": "gather"}),
        choice("execute", action="get 1 log", expected_outcome="log acquired",
               progress_assessment="stalled", progress_evidence="inventory did not gather log"),
    ])
    config = config_for("autonomous_recovery_v4")
    config["max_actions"] = config["max_decisions"] = 4
    config["autonomous_support"]["operations"] = []
    result = run_recovery_episode({"task_id": "dev", "initial_observation": Episode().initial_observation},
        config, Qwen(), gpt, episode_factory=factory)
    assert result["success"] and result["environment_actions"] == 4
    assert result["strategy_changes"] == 2
    assert result["replanning_events"] == 1
    assert result["progress_assessments"] == 3
    events = [record["support_trace"]["events"][0] for record in result["trace"]]
    assert events[0]["state_changes"]["plan"] == {"before": "", "after": "inspect inventory"}
    assert "plan" not in events[1]["state_changes"]
    assert events[2]["state_changes"]["plan"] == {
        "before": "inspect inventory", "after": "gather log"}
    assert result["recovery_state_final"]["plan"] == "gather log"


def test_real_seed51_request_executes_inspect_state():
    gpt = Gpt([REAL_V4["request_inspect_state"]["text"],
               json.dumps({"kind": "execute", "action": "get 1 log"})])
    config = config_for("autonomous_recovery_v4")
    config["autonomous_support"]["operations"] = ["inspect_state_v1"]
    executor = recovery_executor(config, gpt)
    state = RecoveryState(objective="craft cyan dye",
        last_progress=REAL_V4["request_inspect_state"]["recovery_state_before"]["last_progress"])
    context = SupportContext("textcraft", "dev", Episode().task_prompt, "Could not craft", (),
        (("inventory", "Inventory: empty"),), 3, "inventory", "inventory",
        remaining_actions=150, remaining_decisions=218)
    budget = SupportBudget(7, 100, 500, 3000000, 10)
    trace, stop = adaptive_loop(gpt, executor, budget, context, state,
                                deadline=10**12, inference_limit=2100, inference_used=0)
    assert stop is None and trace.final_action == "get 1 log"
    assert trace.operation_count == 1 and budget.operations_used == 1
    assert trace.events[0]["request"]["operation"] == "inspect_state_v1"
    assert trace.events[0]["result"]["status"] == "ok"
    assert state.support_findings[0]["operation"] == "inspect_state_v1"
    assert state.last_progress["assessment"] is None


def test_real_seed126_execute_with_pending_progress_is_accepted():
    gpt = Gpt([REAL_V4["execute_decision"]["text"]])
    config = config_for("autonomous_recovery_v4")
    config["autonomous_support"]["operations"] = []
    state = RecoveryState(objective="craft brown dye",
        last_progress=REAL_V4["execute_decision"]["recovery_state_before"]["last_progress"])
    context = SupportContext("textcraft", "dev", Episode().task_prompt, "Got 1 cocoa beans", (),
        (("get 1 cocoa beans", "Got 1 cocoa beans"),), 3, "inventory", "inventory",
        remaining_actions=150, remaining_decisions=218)
    budget = SupportBudget(7, 100, 500, 3000000, 10)
    trace, stop = adaptive_loop(gpt, recovery_executor(config, gpt), budget, context, state,
                                deadline=10**12, inference_limit=2100, inference_used=0)
    assert stop is None and trace.final_action == "craft 1 brown dye using 1 cocoa beans"
    assert state.last_progress["assessment"] is None
    assert trace.events[0]["recovery_state_after"]["plan"] == "Craft brown dye from cocoa beans"


def test_real_seed126_execute_reaches_environment_and_persists_state():
    gpt = Gpt([REAL_V4["execute_decision"]["text"]])
    config = config_for("autonomous_recovery_v4")
    config["max_actions"] = config["max_decisions"] = 1
    config["autonomous_support"]["operations"] = []
    result = run_recovery_episode({"task_id": "dev", "initial_observation": Episode().initial_observation},
        config, Qwen(), gpt, episode_factory=factory)
    assert result["environment_actions"] == 1 and result["controller_errors"] == 0
    assert result["controller_fallbacks"] == 0
    assert result["trace"][0]["action"] == "craft 1 brown dye using 1 cocoa beans"
    assert result["recovery_state_final"]["subgoal"] == "obtain brown dye"
    assert result["recovery_state_final"]["expected_outcome"] == (
        "Brown dye appears in inventory as it has the required ingredient.")


def test_real_malformed_header_generation_falls_back_with_status():
    class Malformed(Gpt):
        def generate(self, messages, max_new_tokens=None):
            self.messages.append(messages)
            return {"text": "", "raw_completion": REAL_V4["malformed_final_header"]["raw_completion"],
                    "generation_status": "malformed_final_header", "input_tokens": 20,
                    "output_tokens": 10, "inference_seconds": 0.1}
    config = config_for("autonomous_recovery_v4")
    config["max_actions"] = config["max_decisions"] = 1
    config["autonomous_support"]["operations"] = []
    result = run_recovery_episode({"task_id": "dev", "initial_observation": Episode().initial_observation},
        config, Qwen(), Malformed([]), episode_factory=factory)
    assert result["controller_fallbacks"] == 1 and result["controller_errors"] == 1
    assert result["support_calls"] == 0 and result["environment_actions"] == 1
    assert result["trace"][0]["controller_fallback_reason"] == "controller_generation_failure"
    assert result["trace"][0]["controller_fallback_detail"] == "malformed_final_header"


def test_two_malformed_final_headers_count_as_two_fallbacks():
    class MalformedTwice(Gpt):
        def generate(self, messages, max_new_tokens=None):
            self.messages.append(messages)
            return {"text": "", "raw_completion": "invalid final header",
                    "generation_status": "malformed_final_header", "input_tokens": 20,
                    "output_tokens": 10, "inference_seconds": 0.1}
    config = config_for("autonomous_recovery_v4")
    config["max_actions"] = config["max_decisions"] = 2
    config["autonomous_support"]["operations"] = []
    result = run_recovery_episode({"task_id": "dev", "initial_observation": Episode().initial_observation},
        config, Qwen(), MalformedTwice([]), episode_factory=factory)
    assert result["controller_model_calls"] == 2
    assert result["controller_errors"] == result["controller_fallbacks"] == 2
    assert result["support_calls"] == 0 and result["environment_actions"] == 2
    assert [record["controller_fallback_detail"] for record in result["trace"]] == [
        "malformed_final_header", "malformed_final_header"]


def test_controller_cuda_oom_terminates_episode_with_estimated_usage():
    detail = "CUDA out of memory. Tried to allocate 2 GiB"
    gpt = ErrorGpt([ControllerWorkerOOM(detail)])
    config = config_for("autonomous_recovery_v4")
    config["autonomous_support"]["operations"] = []
    episode = Episode()
    checkpoints = []
    result = run_recovery_episode({"task_id": "dev", "initial_observation": episode.initial_observation},
        config, Qwen(), gpt, episode_factory=lambda *_: episode, checkpoint=checkpoints.append)
    assert result["status"] == result["termination_reason"] == "controller_cuda_oom"
    assert result["success"] is False and result["environment_actions"] == 0
    assert episode.steps == [] and len(result["trace"]) == 1
    assert result["qwen_model_calls"] == result["controller_model_calls"] == 1
    assert result["controller_input_tokens"] == 20 and result["controller_output_tokens"] == 0
    assert result["controller_errors"] == 1 and result["controller_fallbacks"] == 0
    record = result["trace"][0]
    assert record["proposed_action"] == "inventory" and record["action"] is None
    assert record["termination_reason"] == "controller_cuda_oom"
    assert record["support_trace"]["final_action"] is None
    assert record["support_trace"]["termination_reason"] == "controller_cuda_oom"
    event = record["support_trace"]["events"][0]
    assert event["error_detail"] == detail and detail in event["error"]
    assert event["usage_estimated"] is True
    assert event["controller_cost"] == {"model_calls": 1, "input_tokens": 20,
                                         "output_tokens": 0, "inference_seconds": 0.0}
    assert record["support_budget_before"]["additional_model_calls"] == 0
    assert record["support_budget_after"]["additional_model_calls"] == 1
    assert checkpoints[-1]["environment_actions"] == 0
    assert checkpoints[-1]["trace"][0]["action"] is None


def test_controller_cuda_oom_after_prior_action_stops_further_steps():
    gpt = ErrorGpt([choice("execute", action="inventory", expected_outcome="inventory feedback"),
                    ControllerWorkerOOM("allocation failed")])
    config = config_for("autonomous_recovery_v4")
    config["autonomous_support"]["operations"] = []
    episode = Episode()
    result = run_recovery_episode({"task_id": "dev", "initial_observation": episode.initial_observation},
        config, Qwen(), gpt, episode_factory=lambda *_: episode)
    assert result["status"] == "controller_cuda_oom"
    assert episode.steps == ["inventory"]
    assert result["environment_actions"] == 1 and result["model_decisions"] == 2
    assert result["qwen_model_calls"] == result["controller_model_calls"] == 2
    assert result["trace"][1]["action"] is None
    assert result["trace"][1]["support_trace"]["events"][0]["usage_estimated"] is True


def test_fatal_cuda_after_prior_action_stops_before_repeated_worker_error():
    detail = "CUDA error: an illegal memory access was encountered"
    gpt = ErrorGpt([choice("execute", action="inventory", expected_outcome="inventory feedback"),
                    ControllerWorkerFatalCUDA("AcceleratorError", detail, "kernel stack"),
                    RuntimeError("repeated poisoned worker error")])
    config = config_for("autonomous_recovery_v4")
    config["autonomous_support"]["operations"] = []
    episode = Episode()
    checkpoints = []
    result = run_recovery_episode({"task_id": "dev", "initial_observation": episode.initial_observation},
        config, Qwen(), gpt, episode_factory=lambda *_: episode, checkpoint=checkpoints.append)
    assert result["status"] == result["termination_reason"] == "controller_cuda_fatal"
    assert episode.steps == ["inventory"]
    assert result["environment_actions"] == 1 and result["model_decisions"] == 2
    assert result["controller_model_calls"] == 2 and result["controller_fallbacks"] == 0
    assert result["controller_input_tokens"] == 40 and result["controller_output_tokens"] == 10
    record = result["trace"][1]
    assert record["decision"] == 2 and record["action"] is None
    assert record["termination_reason"] == "controller_cuda_fatal"
    event = record["support_trace"]["events"][0]
    assert event["error_detail"] == detail and "AcceleratorError" in event["error"]
    assert event["worker_traceback"] == "kernel stack"
    assert event["usage_estimated"] is True
    assert event["controller_cost"] == {"model_calls": 1, "input_tokens": 20,
                                         "output_tokens": 0, "inference_seconds": 0.0}
    assert checkpoints[-1]["trace"][1]["action"] is None


def test_worker_oom_during_reservation_terminates_without_model_charge():
    class OOMAtCount(ErrorGpt):
        def count_messages_tokens(self, messages):
            raise ControllerWorkerOOM("token-count allocation failed")

    config = config_for("autonomous_recovery_v4")
    config["autonomous_support"]["operations"] = []
    episode = Episode()
    result = run_recovery_episode({"task_id": "dev", "initial_observation": episode.initial_observation},
        config, Qwen(), OOMAtCount([]), episode_factory=lambda *_: episode)
    assert result["status"] == "controller_cuda_oom" and episode.steps == []
    assert result["environment_actions"] == 0 and result["controller_model_calls"] == 0
    event = result["trace"][0]["support_trace"]["events"][0]
    assert event["stage"] == "reservation"
    assert event["error_detail"] == "token-count allocation failed"


def test_worker_oom_during_support_reservation_is_terminal():
    class OOMSupport:
        operation_id = "oom_support_v1"
        description = "Model-backed support that fails while counting tokens"
        argument_schema = {}
        model_calls_per_use = 1

        def validate_arguments(self, arguments, context):
            assert arguments == {}

        def reservation(self, arguments, context):
            raise ControllerWorkerOOM("support token-count allocation failed")

    gpt = Gpt([choice("request", operation="oom_support_v1", arguments={},
                      uncertainty="unknown", expected_information="evidence")])
    config = config_for("autonomous_recovery_v4")
    episode = Episode()
    result = run_recovery_episode({"task_id": "dev", "initial_observation": episode.initial_observation},
        config, Qwen(), gpt, episode_factory=lambda *_: episode,
        executor=SupportExecutor({"oom_support_v1": OOMSupport()}))
    assert result["status"] == "controller_cuda_oom" and episode.steps == []
    assert result["environment_actions"] == result["support_calls"] == 0
    assert result["controller_model_calls"] == 1
    event = result["trace"][0]["support_trace"]["events"][0]
    assert event["stage"] == "support_reservation"
    assert event["request"]["operation"] == "oom_support_v1"
    assert event["error_detail"] == "support token-count allocation failed"


def test_non_oom_backend_error_still_uses_qwen_fallback():
    gpt = ErrorGpt([RuntimeError("CUDA out of memory text in a non-OOM error")])
    config = config_for("autonomous_recovery_v4")
    config["max_actions"] = config["max_decisions"] = 1
    config["autonomous_support"]["operations"] = []
    episode = Episode()
    result = run_recovery_episode({"task_id": "dev", "initial_observation": episode.initial_observation},
        config, Qwen(), gpt, episode_factory=lambda *_: episode)
    assert episode.steps == ["inventory"] and result["environment_actions"] == 1
    assert result["controller_fallbacks"] == result["controller_errors"] == 1
    assert result["trace"][0]["controller_fallback_reason"] == "controller_backend_error"
    assert result["trace"][0]["support_trace"]["termination_reason"] == "controller_backend_error"


def test_expanded_v3_controller_oom_keeps_existing_fallback():
    gpt = ErrorGpt([ControllerWorkerOOM("CUDA allocation failed")])
    config = config_for("expanded_v3_v1")
    config["max_actions"] = config["max_decisions"] = 1
    episode = Episode()
    result = run_recovery_episode({"task_id": "dev", "initial_observation": episode.initial_observation},
        config, Qwen(), gpt, episode_factory=lambda *_: episode,
        controller=make_support_controller(config, gpt), executor=SupportExecutor({}))
    assert episode.steps == ["inventory"] and result["environment_actions"] == 1
    assert result["trace"][0]["support_trace"]["termination_reason"] == "controller_backend_error"
    assert result["status"] == "max_actions"


def test_expanded_v3_fatal_cuda_keeps_existing_fallback():
    gpt = ErrorGpt([ControllerWorkerFatalCUDA("AcceleratorError",
        "CUDA error: an illegal memory access was encountered")])
    config = config_for("expanded_v3_v1")
    config["max_actions"] = config["max_decisions"] = 1
    episode = Episode()
    result = run_recovery_episode({"task_id": "dev", "initial_observation": episode.initial_observation},
        config, Qwen(), gpt, episode_factory=lambda *_: episode,
        controller=make_support_controller(config, gpt), executor=SupportExecutor({}))
    assert result["status"] == "max_actions" and episode.steps == ["inventory"]
    assert result["trace"][0]["support_trace"]["termination_reason"] == "controller_backend_error"


def test_reason_over_artifacts_oom_preserves_cost_and_stops_v4():
    detail = "support generation CUDA allocation failed"
    gpt = ErrorGpt([
        choice("request", operation="reason_over_artifacts_v1",
               arguments={"objective": "inspect evidence", "output_type": "plan"},
               uncertainty="which plan works", expected_information="evidence-based plan"),
        ControllerWorkerOOM(detail),
    ])
    config = config_for("autonomous_recovery_v4")
    config["autonomous_support"]["operations"] = ["reason_over_artifacts_v1"]
    episode = Episode()
    checkpoints = []
    result = run_recovery_episode({"task_id": "dev", "initial_observation": episode.initial_observation},
        config, Qwen(), gpt, episode_factory=lambda *_: episode, checkpoint=checkpoints.append)
    assert result["status"] == "controller_cuda_oom" and episode.steps == []
    assert result["environment_actions"] == 0 and result["controller_fallbacks"] == 0
    assert result["controller_model_calls"] == result["support_model_calls"] == 1
    assert result["support_calls"] == 1 and result["model_calls"] == 3
    assert result["controller_input_tokens"] == result["support_input_tokens"] == 20
    event = result["trace"][0]["support_trace"]["events"][0]
    assert event["request"]["operation"] == "reason_over_artifacts_v1"
    assert event["result"]["status"] == "error"
    assert event["result"]["cost"] == {"model_calls": 1, "input_tokens": 20,
                                         "output_tokens": 0, "inference_seconds": 0.0}
    assert event["result"]["metadata"]["worker_failure"] == {"kind": "cuda_oom", "detail": detail}
    assert event["error_detail"] == detail
    assert result["recovery_state_final"]["support_findings"][0]["status"] == "error"
    assert result["trace"][0]["action"] is None
    assert checkpoints[-1]["trace"][0]["support_trace"]["events"][0]["result"] == event["result"]


def test_reason_over_artifacts_fatal_cuda_preserves_typed_marker():
    failure = ControllerWorkerFatalCUDA("AcceleratorError",
        "CUDA error: an illegal memory access was encountered", "kernel stack")
    gpt = ErrorGpt([
        choice("request", operation="reason_over_artifacts_v1",
               arguments={"objective": "inspect evidence", "output_type": "plan"}),
        failure,
    ])
    config = config_for("autonomous_recovery_v4")
    config["autonomous_support"]["operations"] = ["reason_over_artifacts_v1"]
    episode = Episode()
    result = run_recovery_episode({"task_id": "dev", "initial_observation": episode.initial_observation},
        config, Qwen(), gpt, episode_factory=lambda *_: episode)
    assert result["status"] == "controller_cuda_fatal" and episode.steps == []
    assert result["support_calls"] == 1 and result["support_model_calls"] == 1
    event = result["trace"][0]["support_trace"]["events"][0]
    assert event["result"]["metadata"]["worker_failure"] == {
        "kind": "cuda_fatal", "detail": failure.detail,
        "error_type": "AcceleratorError", "worker_traceback": "kernel stack"}
    assert event["worker_traceback"] == "kernel stack"
    assert result["trace"][0]["action"] is None


@pytest.mark.parametrize("failure,status,kind", [
    (ControllerWorkerOOM("registered support OOM"), "controller_cuda_oom", "cuda_oom"),
    (ControllerWorkerFatalCUDA("AcceleratorError", "CUDA error: device-side assert triggered"),
     "controller_cuda_fatal", "cuda_fatal"),
    (ControllerWorkerUnavailable("registered support stream closed"),
     "controller_worker_unavailable", "unavailable"),
])
def test_registered_support_worker_failure_uses_reserved_cost_once(failure, status, kind):
    class BrokenSupport:
        operation_id = "broken_support_v1"
        description = "Registered model support"
        argument_schema = {}
        model_calls_per_use = 1

        def validate_arguments(self, arguments, context):
            assert arguments == {}

        def reservation(self, arguments, context):
            return SupportReservation(1, 13, 64)

        def execute(self, arguments, context):
            raise failure

    gpt = Gpt([choice("request", operation="broken_support_v1", arguments={})])
    config = config_for("autonomous_recovery_v4")
    episode = Episode()
    result = run_recovery_episode({"task_id": "dev", "initial_observation": episode.initial_observation},
        config, Qwen(), gpt, episode_factory=lambda *_: episode,
        executor=SupportExecutor({"broken_support_v1": BrokenSupport()}))
    assert result["status"] == status and episode.steps == []
    assert result["controller_model_calls"] == result["support_model_calls"] == 1
    assert result["support_calls"] == 1 and result["model_calls"] == 3
    assert result["support_input_tokens"] == 13 and result["support_output_tokens"] == 0
    event = result["trace"][0]["support_trace"]["events"][0]
    assert event["result"]["cost"] == {"model_calls": 1, "input_tokens": 13,
                                         "output_tokens": 0, "inference_seconds": 0.0}
    assert event["result"]["metadata"]["usage_estimated"] is True
    marker = event["result"]["metadata"]["worker_failure"]
    assert marker["kind"] == kind and marker["detail"] == failure.detail
    if kind == "cuda_fatal":
        assert marker["error_type"] == "AcceleratorError"
    assert event["error_detail"] == failure.detail
    assert result["trace"][0]["action"] is None


def test_support_oom_after_nonfailed_environment_action_stops_next_action():
    class NonfailedInventory(Episode):
        def step(self, action):
            if action != "inventory":
                return super().step(action)
            self.steps.append(action)
            self.observation = "Inventory: empty"
            return {"observation": self.observation, "done": False,
                    "success": False, "failed_action": False}

    gpt = ErrorGpt([
        choice("execute", action="inventory", expected_outcome="visible inventory"),
        choice("request", operation="reason_over_artifacts_v1",
               arguments={"objective": "inspect evidence", "output_type": "plan"}),
        ControllerWorkerOOM("support allocation failed"),
    ])
    config = config_for("autonomous_recovery_v4")
    config["autonomous_support"]["operations"] = ["reason_over_artifacts_v1"]
    episode = NonfailedInventory()
    result = run_recovery_episode({"task_id": "dev", "initial_observation": episode.initial_observation},
        config, Qwen(), gpt, episode_factory=lambda *_: episode)
    assert result["status"] == "controller_cuda_oom"
    assert episode.steps == ["inventory"] and result["failed_actions"] == 0
    assert result["environment_actions"] == 1 and result["model_decisions"] == 2
    assert result["trace"][1]["action"] is None


@pytest.mark.parametrize("at_count", [False, True])
def test_worker_unavailable_stops_v4_controller_without_fallback(at_count):
    class UnavailableAtCount(ErrorGpt):
        def count_messages_tokens(self, messages):
            if at_count:
                raise ControllerWorkerUnavailable("worker exited")
            return 20

    gpt = UnavailableAtCount([ControllerWorkerUnavailable("worker closed protocol stream")])
    config = config_for("autonomous_recovery_v4")
    config["autonomous_support"]["operations"] = []
    episode = Episode()
    result = run_recovery_episode({"task_id": "dev", "initial_observation": episode.initial_observation},
        config, Qwen(), gpt, episode_factory=lambda *_: episode)
    assert result["status"] == "controller_worker_unavailable" and episode.steps == []
    assert result["environment_actions"] == result["controller_fallbacks"] == 0
    assert result["controller_model_calls"] == (0 if at_count else 1)
    event = result["trace"][0]["support_trace"]["events"][0]
    assert event["error_detail"] == ("worker exited" if at_count else "worker closed protocol stream")
    assert result["trace"][0]["action"] is None


def test_non_oom_reason_support_error_still_allows_next_decision():
    gpt = ErrorGpt([
        choice("request", operation="reason_over_artifacts_v1",
               arguments={"objective": "inspect evidence", "output_type": "plan"}),
        RuntimeError("ordinary support failure"),
        choice("execute", action="get 1 log", expected_outcome="collect log"),
    ])
    config = config_for("autonomous_recovery_v4")
    config["autonomous_support"]["operations"] = ["reason_over_artifacts_v1"]
    episode = Episode()
    result = run_recovery_episode({"task_id": "dev", "initial_observation": episode.initial_observation},
        config, Qwen(), gpt, episode_factory=lambda *_: episode)
    assert result["status"] == "success" and episode.steps == ["get 1 log"]
    assert result["support_calls"] == 1 and result["controller_fallbacks"] == 0
    event = result["trace"][0]["support_trace"]["events"][0]
    assert event["result"]["status"] == "error"
    assert "worker_failure" not in event["result"]["metadata"]


def test_expanded_v3_reason_support_oom_keeps_support_failure_fallback():
    class RequestController:
        controller_id = "test_request_controller"

        def reservation(self, context):
            return SupportReservation()

        def decide(self, context):
            request = SupportRequest("reason_over_artifacts_v1",
                {"objective": "inspect evidence", "output_type": "plan"})
            return ControllerResponse(SupportDecision("request", request=request))

    gpt = ErrorGpt([ControllerWorkerOOM("support allocation failed")])
    config = config_for("expanded_v3_v1")
    operation = ReasonOverArtifacts(config, gpt)
    context = SupportContext("textcraft", "dev", Episode().task_prompt,
        Episode().observation, (), (), 1, "inventory", "inventory")
    budget = SupportBudget(7, 100, 500, 3000000, 10)
    trace = run_support_loop(RequestController(),
        SupportExecutor({"reason_over_artifacts_v1": operation}), budget, context, parse_textcraft)
    assert trace.termination_reason == "support_failure"
    assert trace.final_action == "inventory" and trace.operation_count == 1
    assert trace.events[0]["result"]["status"] == "error"
    assert budget.support_model_calls_used == 1
