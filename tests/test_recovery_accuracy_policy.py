"""CPU-only checks for the opt-in v4 selector and its controller integration."""
import json

from agent_support.autonomous_support import (SupportBudget, SupportContext,
                                              SupportRequest, SupportResult)
from agent_support.recovery_accuracy_policy import (AccuracyFirstPolicy,
    PolicyDirective, StateAwarePolicy)
from agent_support.recovery_v4 import (RecoveryState, adaptive_loop, recovery_executor,
                                       run_recovery_episode)
from agent_support.recovery_v4_eval import config_for


CRAFT = "craft 1 plank using 1 log"
PROMPT = f"Crafting commands:\n{CRAFT}\nGoal: craft plank."


class FakeModel:
    max_new_tokens = 1024

    def __init__(self):
        self.calls = []

    def count_messages_tokens(self, messages):
        return 20

    def generate(self, messages, max_new_tokens=None):
        payload = json.loads(messages[-1]["content"])
        self.calls.append(payload)
        if "requested_analysis" in payload:
            objective = payload["requested_analysis"]["objective"]
            action = "inventory" if "meaningfully different" in objective else "get 1 log"
            answer = {"output_type": "candidate_action", "content": action}
        else:
            answer = {"kind": "execute", "action": CRAFT, "reason": "try craft"}
        return {"text": json.dumps(answer), "generation_status": "complete",
                "input_tokens": 20, "output_tokens": 10, "inference_seconds": 0.1}


def executor(model):
    config = config_for("autonomous_recovery_v4")
    config["autonomous_support"]["operations"] = [
        "analyze_failures_v1", "verify_prerequisites_v1",
        "reason_over_artifacts_v1", "inspect_candidate_v1"]
    return recovery_executor(config, model)


def test_stalled_action_gets_competing_candidates_and_checked_alternative():
    model = FakeModel()
    context = SupportContext("textcraft", "dev", PROMPT, "Could not craft", (),
        ((CRAFT, "Could not find enough items to craft plank"),), 2,
        CRAFT, CRAFT, remaining_actions=149, remaining_decisions=219)
    state = RecoveryState(objective="craft plank")
    budget = SupportBudget(7, 100, 500, 3000000, 10)
    trace, stop = adaptive_loop(model, executor(model), budget, context, state,
        deadline=10**12, inference_limit=2100, inference_used=0,
        policy=AccuracyFirstPolicy())
    operations = [event["request"]["operation"] for event in trace.events
                  if "request" in event]
    assert stop is None and trace.final_action == "get 1 log"
    assert operations.count("reason_over_artifacts_v1") == 2
    objectives = [call["requested_analysis"]["objective"] for call in model.calls
                  if "requested_analysis" in call]
    assert "get 1 log" in objectives[1]
    assert "analyze_failures_v1" in operations
    assert "verify_prerequisites_v1" in operations
    assert operations.count("inspect_candidate_v1") >= 2
    assert budget.operations_used == trace.operation_count == len(operations)
    assert budget.controller_calls_used == trace.controller_decisions
    assert any(event.get("policy_action_selection", {}).get("to") == "get 1 log"
               for event in trace.events)
    assert all(call.get("policy_guidance", {}).get("policy_id") == "accuracy_first_v1"
               for call in model.calls if "requested_analysis" not in call)


def test_no_failure_uses_verification_without_multi_candidate_search():
    model = FakeModel()
    context = SupportContext("textcraft", "dev", PROMPT, PROMPT, (), (), 1,
        CRAFT, CRAFT)
    budget = SupportBudget(7, 100, 500, 3000000, 10)
    trace, stop = adaptive_loop(model, executor(model), budget, context,
        RecoveryState(objective="craft plank"), deadline=10**12,
        inference_limit=2100, inference_used=0, policy=AccuracyFirstPolicy())
    operations = [event["request"]["operation"] for event in trace.events
                  if "request" in event]
    assert stop is None and trace.final_action == CRAFT
    assert operations == ["verify_prerequisites_v1", "inspect_candidate_v1"]


def test_state_aware_comparator_omits_candidate_generation_on_stall():
    model = FakeModel()
    context = SupportContext("textcraft", "dev", PROMPT, "Could not craft", (),
        ((CRAFT, "Could not craft"),), 2, CRAFT, CRAFT)
    budget = SupportBudget(7, 100, 500, 3000000, 10)
    trace, stop = adaptive_loop(model, executor(model), budget, context,
        RecoveryState(objective="craft plank"), deadline=10**12,
        inference_limit=2100, inference_used=0, policy=StateAwarePolicy())
    operations = [event["request"]["operation"] for event in trace.events
                  if "request" in event]
    assert stop is None and "reason_over_artifacts_v1" not in operations
    assert operations == ["analyze_failures_v1", "verify_prerequisites_v1",
                          "inspect_candidate_v1"]


def test_stagnation_retrieves_frozen_memory_when_available():
    class Memory:
        max_top_k = 3
        max_example_tokens = 1200
        corpus_sha256 = "frozen-corpus"
        index_sha256 = "frozen-index"

        def __init__(self):
            self.queries = []

        def search(self, benchmark, query, scope, top_k):
            self.queries.append((benchmark, query, scope, top_k))
            return ([{"summary": "collect a log first"}], 0.01)

    model, memory = FakeModel(), Memory()
    config = config_for("autonomous_recovery_v4")
    config["autonomous_support"]["operations"] = [
        "analyze_failures_v1", "verify_prerequisites_v1",
        "retrieve_episodic_memory_v1", "inspect_candidate_v1"]
    context = SupportContext("textcraft", "dev", PROMPT, "Could not craft", (),
        ((CRAFT, "Could not craft"),), 2, CRAFT, CRAFT)
    budget = SupportBudget(7, 100, 500, 3000000, 10)
    trace, stop = adaptive_loop(model, recovery_executor(config, model, memory), budget,
        context, RecoveryState(objective="craft plank"), deadline=10**12,
        inference_limit=2100, inference_used=0, policy=StateAwarePolicy())
    assert stop is None
    assert memory.queries and memory.queries[0][0] == "textcraft"
    assert budget.support_model_calls_used == 1
    assert any(event.get("request", {}).get("operation") == "retrieve_episodic_memory_v1"
               for event in trace.events)


def test_action_arbitration_requires_checked_visible_alternative():
    policy = AccuracyFirstPolicy()
    history = ((CRAFT, "Could not craft"),)
    unchecked = SupportContext("textcraft", "dev", PROMPT, "Could not craft", (),
        history, 2, CRAFT, CRAFT,
        results=(SupportResult("ok", proposed_action="get 1 log"),))
    assert policy.choose_action(unchecked, CRAFT) == "inventory"
    checked = SupportContext("textcraft", "dev", PROMPT, "Could not craft", (),
        history, 2, CRAFT, CRAFT,
        results=unchecked.results + (SupportResult("ok", {
            "parsed_action": "get 1 log", "format_valid": True,
            "admissible": True, "displayed_recipe": None}),))
    assert policy.choose_action(checked, CRAFT) == "get 1 log"


def test_successful_collection_clears_old_failure_signal():
    policy = AccuracyFirstPolicy()
    context = SupportContext("textcraft", "dev", PROMPT, "Got 1 log", (),
        ((CRAFT, "Could not craft"), ("get 1 log", "Got 1 log")),
        3, CRAFT, CRAFT)
    assert policy.choose_action(context, CRAFT) == CRAFT
    assert policy.guidance(context, RecoveryState(objective="craft plank"))["stalled"] is False


def test_alternative_craft_needs_observed_sufficient_prerequisites():
    other = "craft 1 stick using 1 plank"
    policy = AccuracyFirstPolicy()
    results = (SupportResult("ok", proposed_action=other),
               SupportResult("ok", {"parsed_action": other, "format_valid": True,
                                     "admissible": True, "displayed_recipe": True}),
               SupportResult("ok", {"displayed_recipe": other,
                                     "all_prerequisites_observed_sufficient": None}))
    context = SupportContext("textcraft", "dev", PROMPT + "\n" + other,
        "Could not craft", (), ((CRAFT, "Could not craft"),), 2,
        CRAFT, CRAFT, results=results)
    assert policy.choose_action(context, CRAFT) == "inventory"
    sufficient = SupportResult("ok", {"displayed_recipe": other,
                                      "all_prerequisites_observed_sufficient": True})
    assert policy.choose_action(SupportContext("textcraft", "dev", PROMPT + "\n" + other,
        "Could not craft", (), ((CRAFT, "Could not craft"),), 2,
        CRAFT, CRAFT, results=results + (sufficient,)), CRAFT) == other


def test_policy_runs_through_episode_runner_without_gpu():
    class Qwen:
        def generate(self, messages):
            return {"text": f"Action: {CRAFT}", "input_tokens": 10,
                    "output_tokens": 5, "inference_seconds": 0.1}

    class Episode:
        initial_observation = task_prompt = PROMPT

        def __init__(self):
            self.observation = PROMPT
            self.actions = []

        def reset(self):
            pass

        def step(self, action):
            self.actions.append(action)
            success = action == "get 1 log"
            self.observation = "Got 1 log" if success else "Could not craft"
            return {"observation": self.observation, "done": success,
                    "success": success, "failed_action": not success}

        def close(self):
            pass

    model = FakeModel()
    episode = Episode()
    config = config_for("autonomous_recovery_v4")
    config["max_actions"] = config["max_decisions"] = 2
    config["autonomous_support"]["operations"] = [
        "analyze_failures_v1", "verify_prerequisites_v1",
        "reason_over_artifacts_v1", "inspect_candidate_v1"]
    result = run_recovery_episode({"task_id": "dev", "initial_observation": PROMPT},
        config, Qwen(), model, episode_factory=lambda *_: episode,
        executor=recovery_executor(config, model), policy=AccuracyFirstPolicy())
    assert result["success"] is True
    assert result["support_policy_id"] == "accuracy_first_v1"
    assert episode.actions == [CRAFT, "get 1 log"]
    assert result["support_calls"] > 2


def test_invalid_policy_request_is_rejected_by_existing_parser():
    class InvalidPolicy(AccuracyFirstPolicy):
        def select(self, context, state, available):
            return PolicyDirective(SupportRequest("get", {"amount": 1}),
                                   "invalid operation", "invalid operation")

    model = FakeModel()
    context = SupportContext("textcraft", "dev", PROMPT, PROMPT, (), (), 1,
        CRAFT, CRAFT)
    budget = SupportBudget(7, 100, 500, 3000000, 10)
    trace, stop = adaptive_loop(model, executor(model), budget, context,
        RecoveryState(objective="craft plank"), deadline=10**12,
        inference_limit=2100, inference_used=0, policy=InvalidPolicy())
    assert stop == trace.termination_reason == "policy_error"
    assert trace.operation_count == budget.operations_used == 0
    assert "Unknown support operation" in trace.events[-1]["error"]
