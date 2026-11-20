from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_support.autonomous_support import SupportBudget, SupportContext
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
