"""CPU-only checks for evidence-based terminal-stop recovery."""
from __future__ import annotations

import json

from agent_support.autonomous_support import SupportBudget, SupportContext
from agent_support.controller_backend import ControllerWorkerFatalCUDA
from agent_support.recovery_accuracy_policy import (AccuracyFirstTerminalPolicy,
                                                    StateAwarePolicy)
from agent_support.recovery_v4 import (RecoveryState, adaptive_loop, recovery_executor,
                                       run_recovery_episode)
from agent_support.recovery_v4_eval import config_for
from agent_support.recovery_policy_telemetry import decision_telemetry


PROMPT = "Crafting commands:\ncraft 1 plank using 1 log\nGoal: craft plank."


class StopModel:
    max_new_tokens = 1024

    def __init__(self, candidates=("get 1 log", "inventory"), fatal=False):
        self.candidates = candidates
        self.fatal = fatal
        self.controller_calls = 0
        self.support_calls = 0

    def count_messages_tokens(self, messages):
        return 20

    def generate(self, messages, max_new_tokens=None):
        payload = json.loads(messages[-1]["content"])
        if "requested_analysis" in payload:
            if self.fatal:
                raise ControllerWorkerFatalCUDA("AcceleratorError",
                    "CUDA driver error: an illegal memory access was encountered",
                    worker_traceback="Traceback: model.generate", request_id=13,
                    failed_phase="model_generate")
            index = min(self.support_calls, len(self.candidates) - 1)
            self.support_calls += 1
            answer = {"output_type": "candidate_action", "content": self.candidates[index]}
        else:
            self.controller_calls += 1
            answer = {"kind": "stop", "stop_reason": "no_viable_strategy",
                      "reason": "Cannot get the ingredient from visible recipes"}
        return {"text": json.dumps(answer), "generation_status": "complete",
                "input_tokens": 20, "output_tokens": 10, "inference_seconds": 0.1}


def make_executor(model):
    config = config_for("autonomous_recovery_v4")
    config["autonomous_support"]["operations"] = ["inspect_state_v1",
        "reason_over_artifacts_v1", "inspect_candidate_v1", "verify_prerequisites_v1"]
    return recovery_executor(config, model)


def run_loop(model, *, history=(), policy=None):
    context = SupportContext("textcraft", "dev", PROMPT,
        history[-1][1] if history else PROMPT, (), history, 1,
        "craft 1 plank using 1 log", "craft 1 plank using 1 log")
    budget = SupportBudget(7, 100, 500, 3000000, 10)
    trace, stop = adaptive_loop(model, make_executor(model), budget, context,
        RecoveryState(objective="craft plank"), deadline=10**12,
        inference_limit=2100, inference_used=0, policy=policy)
    return trace, stop, budget


def test_decision_one_stop_is_reviewed_and_rescued_with_checked_action():
    model = StopModel()
    trace, stop, budget = run_loop(model, policy=AccuracyFirstTerminalPolicy())
    assert stop is None and trace.final_action == "get 1 log"
    assert budget.operations_used > 0
    assert trace.events[0]["controller_decision"]["kind"] == "stop"
    assert trace.events[0]["terminal_review"]["outcome"] == "request"
    assert trace.events[0]["terminal_review"]["evidence"]["inventory_observed"] is False
    assert trace.events[-1]["terminal_review"]["outcome"] == "rescue"
    assert trace.events[-1]["policy_action_selection"]["to"] == "get 1 log"
    operations = [item["request"]["operation"] for item in trace.events if "request" in item]
    assert operations.count("reason_over_artifacts_v1") == 2
    assert "inspect_state_v1" in operations
    assert "inspect_candidate_v1" in operations
    telemetry = decision_telemetry({"decision": 1, "input_observation": PROMPT,
        "proposed_action": "craft 1 plank using 1 log", "action": trace.final_action,
        "support_trace": trace.as_dict()}, "accuracy_first_terminal_v1")
    assert telemetry["terminal_reviews"][-1]["outcome"] == "rescue"
    assert any(item["source"] == "terminal_review" for item in telemetry["candidate_actions"])


def test_terminal_rescue_can_produce_environment_verified_success():
    class Qwen:
        def generate(self, messages):
            return {"text": "Action: craft 1 plank using 1 log",
                    "input_tokens": 10, "output_tokens": 5, "inference_seconds": 0.1}

    class Episode:
        initial_observation = task_prompt = PROMPT

        def __init__(self):
            self.observation = PROMPT
            self.actions = []

        def reset(self):
            pass

        def step(self, action):
            self.actions.append(action)
            self.observation = "Got 1 log"
            return {"observation": self.observation, "done": True,
                    "success": action == "get 1 log", "failed_action": False}

        def close(self):
            pass

    model, episode = StopModel(), Episode()
    config = config_for("autonomous_recovery_v4")
    result = run_recovery_episode({"task_id": "dev", "initial_observation": PROMPT},
        config, Qwen(), model, episode_factory=lambda *_: episode,
        executor=make_executor(model), policy=AccuracyFirstTerminalPolicy())
    assert result["success"] is True and result["status"] == "success"
    assert result["environment_actions"] == 1 and episode.actions == ["get 1 log"]
    assert result["trace"][0]["support_trace"]["events"][0]["controller_decision"]["kind"] == "stop"


def test_exhausted_task_accepts_stop_after_duplicate_candidate_review():
    history = (("inventory", "Inventory: empty"),
               ("get 1 log", "Could not get 1 log"))
    model = StopModel(("get 1 log", "get 1 log"))
    trace, stop, budget = run_loop(model, history=history,
                                   policy=AccuracyFirstTerminalPolicy())
    assert stop == trace.termination_reason == "no_viable_strategy"
    assert trace.final_action is None
    assert budget.operations_used <= 7
    assert sum(item.get("request", {}).get("operation") == "inspect_candidate_v1"
               for item in trace.events) == 1
    assert trace.events[-1]["terminal_review"]["outcome"] == "accept"
    assert trace.events[-1]["terminal_review"]["evidence"]["candidate_actions"] == ["get 1 log"]


def test_terminal_craft_requires_observed_sufficient_ingredients():
    craft = "craft 1 plank using 1 log"
    model = StopModel((craft, craft))
    trace, stop, _ = run_loop(model,
        history=(("inventory", "Inventory: [log] (1)"),),
        policy=AccuracyFirstTerminalPolicy())
    assert stop is None and trace.final_action == craft
    assert sum(item.get("request", {}).get("operation") == "verify_prerequisites_v1"
               for item in trace.events) == 1
    assert trace.events[-1]["terminal_review"]["outcome"] == "rescue"


def test_terminal_review_retrieves_memory_once_when_stop_reason_changes():
    class ChangingStopModel(StopModel):
        def generate(self, messages, max_new_tokens=None):
            result = super().generate(messages, max_new_tokens=max_new_tokens)
            payload = json.loads(messages[-1]["content"])
            if "requested_analysis" not in payload:
                answer = json.loads(result["text"])
                answer["reason"] = f"Impossible for reason {self.controller_calls}"
                result["text"] = json.dumps(answer)
            return result

    class Memory:
        max_top_k = 3
        max_example_tokens = 1200
        corpus_sha256 = "frozen-corpus"
        index_sha256 = "frozen-index"

        def __init__(self):
            self.calls = []

        def search(self, benchmark, query, scope, top_k):
            self.calls.append((benchmark, query, scope, top_k))
            return ([{"summary": "get one log"}], 0.01)

    model, memory = ChangingStopModel(), Memory()
    config = config_for("autonomous_recovery_v4")
    config["autonomous_support"]["operations"] = ["inspect_state_v1",
        "retrieve_episodic_memory_v1", "reason_over_artifacts_v1",
        "inspect_candidate_v1", "verify_prerequisites_v1"]
    context = SupportContext("textcraft", "dev", PROMPT, PROMPT, (), (), 1,
        "craft 1 plank using 1 log", "craft 1 plank using 1 log")
    trace, stop = adaptive_loop(model, recovery_executor(config, model, memory),
        SupportBudget(7, 100, 500, 3000000, 10), context,
        RecoveryState(objective="craft plank"), deadline=10**12,
        inference_limit=2100, inference_used=0, policy=AccuracyFirstTerminalPolicy())
    assert stop is None and trace.final_action == "get 1 log"
    assert len(memory.calls) == 1
    assert trace.events[-1]["terminal_review"]["evidence"]["claim_reason"] == "Impossible for reason 1"


def test_terminal_review_respects_finite_support_depth():
    model = StopModel()
    context = SupportContext("textcraft", "dev", PROMPT, PROMPT, (), (), 1,
        "craft 1 plank using 1 log", "craft 1 plank using 1 log")
    budget = SupportBudget(7, 100, 500, 3000000, 1)
    trace, stop = adaptive_loop(model, make_executor(model), budget, context,
        RecoveryState(objective="craft plank"), deadline=10**12,
        inference_limit=2100, inference_used=0, policy=AccuracyFirstTerminalPolicy())
    assert stop == "no_viable_strategy" and trace.final_action is None
    assert budget.operations_used == 0
    assert trace.events[-1]["terminal_review"]["outcome"] == "accept_at_support_limit"


def test_fatal_cuda_during_terminal_search_remains_terminal():
    model = StopModel(fatal=True)
    trace, stop, _ = run_loop(model, policy=AccuracyFirstTerminalPolicy())
    assert stop == trace.termination_reason == "controller_cuda_fatal"
    assert trace.final_action is None
    assert trace.events[-1]["worker_traceback"] == "Traceback: model.generate"
    assert not any(item.get("policy_action_selection") for item in trace.events)


def test_original_v4_and_state_aware_stops_are_unchanged():
    for policy in (None, StateAwarePolicy()):
        model = StopModel()
        trace, stop, budget = run_loop(model, policy=policy)
        assert stop == "no_viable_strategy" and trace.final_action is None
        assert budget.operations_used == model.support_calls == 0
        assert len(trace.events) == 1
        assert "terminal_review" not in trace.events[0]
