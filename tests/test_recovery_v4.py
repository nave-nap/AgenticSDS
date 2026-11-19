from __future__ import annotations

import json

import pytest

from agent_support.autonomous_support import SupportBudget, SupportContext
from agent_support.recovery_v4 import (AnalyzeFailures, RecoveryState, VerifyPrerequisites,
    adaptive_loop, parse_decision, recovery_executor, run_recovery_episode)
from agent_support.recovery_v4_eval import config_for, manifest


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
