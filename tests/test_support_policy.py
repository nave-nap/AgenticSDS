import json

import pytest

from agent_support.common import canonical_digest, digest_text, write_json_atomic
from agent_support.memory import build_index
from agent_support.runner import run, run_episode
from agent_support.support_policy import (ACTION_TRAJECTORY, LEGACY_ACTION_MEMORY, TASK_TRAJECTORY,
                                          POLICY_REGISTRY, FixedConditionPolicy, TurnContext, make_support_policy,
                                          selected_supports)
from agent_support.trajectory_memory import build_trajectory_index


def context(condition, phase, action=None):
    return TurnContext(phase=phase, benchmark="textcraft", condition=condition, decision=1,
                       task_prompt="Goal: craft plank.", observation="Start", available_actions=(),
                       history=(), actions_taken=0, remaining_actions=3, remaining_decisions=3,
                       support_calls_so_far=0, proposed_action=action)


def test_fixed_policy_preserves_condition_schedule():
    policy = FixedConditionPolicy()
    assert selected_supports(policy, context("baseline", "before_proposal")) == ()
    assert selected_supports(policy, context("expel_retrieval_only_v1", "before_proposal")) == (TASK_TRAJECTORY,)
    assert selected_supports(policy, context("m2_always", "after_proposal", "get 1 log")) == (LEGACY_ACTION_MEMORY,)
    assert selected_supports(policy, context("m2_trajectory_v1", "after_proposal", "get 1 log")) == (ACTION_TRAJECTORY,)
    assert selected_supports(policy, context("m2_trajectory_v1", "after_proposal")) == ()
    with pytest.raises(ValueError, match="Unknown support policy"):
        make_support_policy({"support_policy": "unregistered"})


def test_new_policy_can_be_selected_by_config(monkeypatch):
    class SkipPolicy:
        policy_id = "skip_v1"
        def select(self, _context):
            return ()

    monkeypatch.setitem(POLICY_REGISTRY, SkipPolicy.policy_id, SkipPolicy)
    assert make_support_policy({"support_policy": "skip_v1"}).policy_id == "skip_v1"


def test_policy_rejects_invalid_or_misphased_choices():
    class BadPolicy:
        policy_id = "bad"
        def select(self, _context):
            return (ACTION_TRAJECTORY,)

    with pytest.raises(ValueError, match="wrong phase"):
        selected_supports(BadPolicy(), context("baseline", "before_proposal"))
    with pytest.raises(ValueError, match="without a valid proposal"):
        selected_supports(BadPolicy(), context("baseline", "after_proposal"))


class FakeEpisode:
    task_prompt = initial_observation = observation = "Goal: craft plank."
    actions = []

    def reset(self):
        pass

    def step(self, action):
        return {"observation": "Done", "done": True, "success": action == "get 1 sugar cane",
                "failed_action": False}

    def close(self):
        pass


class FakeModel:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = 0

    def generate(self, _messages):
        self.calls += 1
        return {"text": next(self.outputs), "input_tokens": 10,
                "output_tokens": 3, "inference_seconds": 0.1}


def test_replacement_policy_can_skip_support_without_changing_mechanism():
    class NoSupport:
        policy_id = "no_support_v1"
        def select(self, _context):
            return ()

    class WouldFailIfCalled:
        def retrieve(self, *_args):
            raise AssertionError("retrieval should not be called")

    model = FakeModel(["Action: get 1 sugar cane"])
    result = run_episode("textcraft", {"task_id": "textcraft:seed:42"},
                         {"condition": "m2_trajectory_v1", "max_actions": 1,
                          "max_decisions": 1, "history_window": 1}, model,
                         episode_factory=lambda *_: FakeEpisode(),
                         trajectory_support=WouldFailIfCalled(), support_policy=NoSupport())
    assert result["success"] and model.calls == 1
    assert result["support_calls"] == result["support_model_calls"] == 0
    assert result["trace"][0]["support_policy"]["policy_id"] == "no_support_v1"


def test_policy_can_select_two_supports_in_order():
    class TwoSupports:
        policy_id = "two_supports_v1"
        def select(self, turn):
            return ((LEGACY_ACTION_MEMORY, ACTION_TRAJECTORY)
                    if turn.phase == "after_proposal" and turn.proposed_action else ())

    class Legacy:
        def retrieve(self, *_args):
            return ([{"memory_id": "old", "similarity": 0.9, "task_prompt": "Goal: craft plank.",
                      "observation": "Start", "action": "get 1 log", "feedback": "Got log",
                      "failed_action": False, "episode_success": True}], 0.01)

    class Trajectory:
        def retrieve(self, *_args):
            return ([{"memory_id": "new", "similarity": 0.8,
                      "content": "Past successful task: craft plank.\n1. Action: get 1 sugar cane"}], 0.02)

    model = FakeModel(["Action: get 1 plank", "Action: get 1 log", "Action: get 1 sugar cane"])
    result = run_episode("textcraft", {"task_id": "textcraft:seed:42"},
                         {"condition": "baseline", "max_actions": 1,
                          "max_decisions": 1, "history_window": 1}, model,
                         episode_factory=lambda *_: FakeEpisode(),
                         support=Legacy(), trajectory_support=Trajectory(), support_policy=TwoSupports())
    assert result["success"] and model.calls == 3
    assert result["support_calls"] == result["support_model_calls"] == 2
    assert result["actions_changed_by_support"] == 1
    record = result["trace"][0]
    assert record["proposed_action"] == "get 1 plank"
    assert [event["mechanism"] for event in record["support_events"]] == ["m2_action_memory", "m2_trajectory_memory"]
    assert [event["proposed_action"] for event in record["support_events"]] == ["get 1 plank", "get 1 log"]


def test_task_examples_are_retrieved_once_and_reused_each_turn():
    class TwoStepEpisode(FakeEpisode):
        def __init__(self):
            self.steps = 0
        def step(self, _action):
            self.steps += 1
            return {"observation": "Continue" if self.steps == 1 else "Done",
                    "done": self.steps == 2, "success": self.steps == 2, "failed_action": False}

    class Memory:
        def __init__(self):
            self.calls = 0
        def retrieve(self, *_args):
            self.calls += 1
            return ([{"memory_id": "example", "content": "Past successful task: craft plank."}], 0.01)

    memory = Memory()
    model = FakeModel(["Action: get 1 log", "Action: get 1 sugar cane"])
    result = run_episode("textcraft", {"task_id": "textcraft:seed:42"},
                         {"condition": "expel_retrieval_only_v1", "max_actions": 2,
                          "max_decisions": 2, "history_window": 2}, model,
                         episode_factory=lambda *_: TwoStepEpisode(), trajectory_support=memory)
    assert result["success"] and model.calls == 2 and memory.calls == 1
    assert result["support_calls"] == 1 and result["support_model_calls"] == 0
    assert [step["support"]["retrieval_event"] for step in result["trace"]] == [True, False]


def test_run_loads_both_memories_for_a_replaceable_policy(monkeypatch, tmp_path):
    class TwoSupports:
        policy_id = "both_memories_v1"
        def select(self, turn):
            return ((LEGACY_ACTION_MEMORY, ACTION_TRAJECTORY)
                    if turn.phase == "after_proposal" and turn.proposed_action else ())

    class FakeEmbedder:
        def __init__(self, model_id="fake", revision="v1"):
            self.model_id = model_id
            self.revision = revision
        def encode(self, texts):
            return [[1.0, 0.0] for _ in texts]

    monkeypatch.setitem(POLICY_REGISTRY, TwoSupports.policy_id, TwoSupports)
    monkeypatch.setattr("agent_support.supports.SentenceEmbedder", FakeEmbedder)
    monkeypatch.setattr("agent_support.supports.MPNetEmbedder", FakeEmbedder)
    observation = "Goal: craft plank."
    task = {"task_id": "textcraft:seed:42", "seed": 42, "initial_observation": observation,
            "observation_sha256": digest_text(observation)}
    manifest = {"schema_version": 1, "benchmark": "textcraft", "partition": "development", "tasks": [task]}
    legacy_corpus = {"excluded_manifest_sha256": {"dev": canonical_digest(manifest)},
                     "records": [{"memory_id": "old", "benchmark": "textcraft", "task_id": "textcraft:seed:1000",
                                  "task_prompt": observation, "observation": "Start", "action": "get 1 plank",
                                  "feedback": "Accepted", "failed_action": False, "episode_success": True}]}
    trajectory_corpus = {"excluded_manifest_sha256": {"dev": canonical_digest(manifest)},
                         "records": [{"memory_id": "new", "benchmark": "textcraft", "task_id": "textcraft:seed:1001",
                                      "task_fingerprint": digest_text("Goal: craft paper."),
                                      "task_prompt": "craft paper.", "initial_observation": "Goal: craft paper.",
                                      "steps": [{"observation": "Start", "action": "get 1 sugar cane",
                                                 "feedback": "Accepted", "failed_action": False}]}]}
    legacy_index = build_index(legacy_corpus, FakeEmbedder())
    trajectory_index = build_trajectory_index(trajectory_corpus, FakeEmbedder())
    for name, data in (("legacy_corpus", legacy_corpus), ("legacy_index", legacy_index),
                       ("trajectory_corpus", trajectory_corpus), ("trajectory_index", trajectory_index)):
        write_json_atomic(tmp_path / f"{name}.json", data)
    config = {"benchmark": "textcraft", "condition": "m2_trajectory_v1", "support_policy": "both_memories_v1",
              "model_id": "fake", "quantization": "nf4_4bit", "max_actions": 1,
              "max_decisions": 1, "history_window": 1, "max_new_tokens": 10,
              "memory_corpus": str(tmp_path / "legacy_corpus.json"),
              "memory_index": str(tmp_path / "legacy_index.json"), "memory_top_k": 1,
              "trajectory_corpus": str(tmp_path / "trajectory_corpus.json"),
              "trajectory_index": str(tmp_path / "trajectory_index.json"),
              "trajectory_top_k": 1, "trajectory_token_budget": 100}
    model = FakeModel(["Action: get 1 plank", "Action: get 1 log", "Action: get 1 sugar cane"])
    directory = run(config, manifest, tmp_path / "results", model=model,
                    episode_factory=lambda *_: FakeEpisode())
    metadata = json.loads((directory / "run.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    assert metadata["status"] == "complete" and metadata["support_policy_id"] == "both_memories_v1"
    assert metadata["memory_corpus_sha256"] == canonical_digest(legacy_corpus)
    assert metadata["trajectory_corpus_sha256"] == canonical_digest(trajectory_corpus)
    assert summary["support_calls"] == 2 and summary["support_model_calls"] == 2
