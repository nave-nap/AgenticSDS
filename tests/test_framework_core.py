"""CPU contracts for component registration and legacy support dispatch."""

import pytest

from agent_support.benchmarks import BENCHMARK_REGISTRY, make_episode
from agent_support.controllers import CONTROLLER_REGISTRY, run_controller
from agent_support.model import MODEL_REGISTRY, make_model
from agent_support.supports import SUPPORT_REGISTRY, SupportResult, make_supports


def test_model_factory_allows_external_adapter(monkeypatch):
    class MockModel:
        def generate(self, messages, max_new_tokens=None):
            return {"text": "Action: inventory", "input_tokens": 2, "output_tokens": 1,
                    "inference_seconds": 0.0}
        def metadata(self):
            return {"model_id": "mock", "model_revision": "mock-v1"}
        def count_tokens(self, text):
            return len(text.split())

    monkeypatch.setitem(MODEL_REGISTRY, "mock", lambda config: MockModel())
    assert make_model({"model_type": "mock"}).metadata()["model_revision"] == "mock-v1"
    with pytest.raises(ValueError, match="Unknown model type"):
        make_model({"model_type": "absent"})


def test_support_and_benchmark_registries_allow_external_components(monkeypatch):
    class MockSupport:
        support_id = "mock"
        phase = "after_proposal"
        version = "mock-v1"
        requirements = ()
        def invoke(self, context, messages, action):
            return SupportResult(messages, {"mechanism": "mock", "action_changed": False}, calls=1)

    monkeypatch.setitem(SUPPORT_REGISTRY, "mock", lambda config, model, resources: MockSupport())
    monkeypatch.setitem(BENCHMARK_REGISTRY, "mock", lambda task, config: task)
    assert make_supports({"supports": ["mock"]}, None, {})["mock"].version == "mock-v1"
    assert make_episode("mock", {"task_id": "one"}, {}) == {"task_id": "one"}


def test_controller_registry_keeps_recursive_controller_separate(monkeypatch):
    monkeypatch.setitem(CONTROLLER_REGISTRY, "mock", lambda task, config, model, factory: {"task_id": task["task_id"]})
    assert run_controller("mock", {"task_id": "one"}, {}, None, None) == {"task_id": "one"}
    with pytest.raises(ValueError, match="Unknown controller"):
        run_controller("absent", {}, {}, None, None)
