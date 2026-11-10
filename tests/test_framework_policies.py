"""Selection uses only the current turn and enforces declared budgets."""

from dataclasses import replace

from agent_support.runner import run_episode
from agent_support.common import validate_config
from agent_support.model import MODEL_REGISTRY
from agent_support.support_policy import (FailureTriggeredPolicy, RandomPolicy, TurnContext,
                                          make_support_policy, selection_decision)
from agent_support.supports import SUPPORT_REGISTRY, SupportResult


class Episode:
    task_prompt = initial_observation = observation = "Goal: craft plank."
    actions = []
    def reset(self):
        pass
    def step(self, action):
        self.observation = "Could not craft"
        return {"observation": self.observation, "done": False, "success": False, "failed_action": True}
    def close(self):
        pass


class Model:
    def __init__(self):
        self.calls = 0
    def generate(self, messages):
        self.calls += 1
        return {"text": "Action: inventory", "input_tokens": 4, "output_tokens": 2,
                "inference_seconds": 0.01}
    def count_messages_tokens(self, messages):
        return 4


class MockSupport:
    support_id = "mock"
    phase = "after_proposal"
    version = "mock-v1"
    requirements = ()
    def invoke(self, context, messages, action):
        return SupportResult(messages, {"mechanism": "mock", "action_changed": False}, calls=1,
                             revision_messages=messages)


def config(policy, **extra):
    return {"schema_version": 2, "condition": "modular_v1", "benchmark": "textcraft",
            "max_actions": 3, "max_decisions": 3, "max_model_calls": 6,
            "max_support_calls": 3, "max_support_tokens": 30,
            "history_window": 1, "max_new_tokens": 2, "seed": 5,
            "supports": ["mock"], "selection_policy": policy, **extra}


def context():
    return TurnContext("after_proposal", "textcraft", "modular_v1", 1, "task", "observation",
                       (), (), 0, 3, 3, 0, proposed_action="inventory",
                       available_supports=("mock",))


def test_seeded_random_is_independent_of_task_schedule(monkeypatch):
    monkeypatch.setitem(SUPPORT_REGISTRY, "mock", lambda spec, model, resources: MockSupport())
    spec = {"id": "random_v1", "support_id": "mock", "probability": 0.5, "seed": 991}
    a = make_support_policy({"selection_policy": spec}, "task-a")
    b = make_support_policy({"selection_policy": spec}, "task-b")
    a_sequence = [selection_decision(a, replace(context(), decision=i)).support_ids for i in range(1, 21)]
    _ = [selection_decision(b, replace(context(), decision=i)) for i in range(1, 16)]
    again = make_support_policy({"selection_policy": spec}, "task-a")
    assert a_sequence == [selection_decision(again, replace(context(), decision=i)).support_ids
                          for i in range(1, 21)]


def test_failure_threshold_and_cooldown_use_only_prior_outcomes(monkeypatch):
    monkeypatch.setitem(SUPPORT_REGISTRY, "mock", lambda spec, model, resources: MockSupport())
    policy = FailureTriggeredPolicy({"support_id": "mock", "threshold": 2, "cooldown": 2})
    assert not selection_decision(policy, replace(context(), recent_failures=(True,))).support_ids
    chosen = selection_decision(policy, replace(context(), recent_failures=(True, True)))
    assert chosen.support_ids == ("mock",)
    policy.on_executed(1)
    assert selection_decision(policy, replace(context(), decision=2,
                                              recent_failures=(True, True, True))).reason == "cooldown"
    assert selection_decision(policy, replace(context(), decision=4,
                                              recent_failures=(True, True, True))).support_ids == ("mock",)


def test_always_policy_budget_checks_before_generation(monkeypatch):
    monkeypatch.setitem(SUPPORT_REGISTRY, "mock", lambda spec, model, resources: MockSupport())
    policy = {"id": "always_v1", "support_id": "mock"}
    model = Model()
    result = run_episode("textcraft", {"task_id": "a"}, config(policy, max_support_calls=0),
                         model, episode_factory=lambda *args: Episode())
    assert model.calls == 3 and result["support_calls"] == 0
    assert result["trace"][0]["support_policy"]["blocked"][0]["reason"] == "support_call_budget"

    model = Model()
    result = run_episode("textcraft", {"task_id": "a"}, config(policy, max_support_tokens=5),
                         model, episode_factory=lambda *args: Episode())
    assert model.calls == 3 and result["support_calls"] == 3
    assert result["support_model_calls"] == 0
    assert result["trace"][0]["support_policy"]["blocked"][0]["reason"] == "support_token_budget"

    model = Model()
    result = run_episode("textcraft", {"task_id": "a"}, config(policy, max_model_calls=1),
                         model, episode_factory=lambda *args: Episode())
    assert model.calls == 1 and result["status"] == "max_model_calls"
    assert result["support_calls"] == 0


def test_v2_configuration_pins_components_and_policy_parameters(monkeypatch):
    monkeypatch.setitem(SUPPORT_REGISTRY, "mock", lambda spec, model, resources: MockSupport())
    monkeypatch.setitem(MODEL_REGISTRY, "mock", lambda spec: Model())
    value = config({"id": "random_v1", "support_id": "mock", "probability": 0.25, "seed": 7},
                   manifest="manifests/textcraft/development.json", output_root="results/development",
                   model_type="mock", model_id="mock", model_revision="mock-v1",
                   support_versions={"mock": "mock-v1"},
                   partition_manifests={"development": "manifests/textcraft/development.json"})
    validate_config(value, "textcraft")
    value["selection_policy"]["probability"] = 1.2
    try:
        validate_config(value, "textcraft")
    except ValueError as exc:
        assert "probability" in str(exc)
    else:
        raise AssertionError("Invalid probability was accepted")
