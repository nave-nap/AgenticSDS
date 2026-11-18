"""CPU checks for the first composable autonomous support capabilities."""

import json
from pathlib import Path

import pytest

from agent_support.autonomous_memory import FrozenTrajectoryRetrieval
from agent_support.autonomous_operations import (InspectHistory, InspectState,
                                                 ReasonOverArtifacts, RetrieveEpisodicMemory)
from agent_support.autonomous_support import SupportContext, make_support_executor
from agent_support.common import canonical_digest, validate_config
from agent_support.runner import run, run_episode


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/textcraft_prompted_gptoss_capabilities.json"
MANIFEST = ROOT / "manifests/textcraft/development.json"


def settings():
    return json.loads(CONFIG.read_text())


def task():
    return next(item for item in json.loads(MANIFEST.read_text())["tasks"] if item["seed"] == 130)


class FakeEmbedder:
    model_id = "fake-mpnet"
    revision = "frozen-revision"

    def encode(self, texts):
        return [[0.0, 1.0] if "bed" in text else [1.0, 0.0] for text in texts]


def toy_memory(token_budget=300):
    records = [
        {"memory_id": "bee-memory", "benchmark": "textcraft", "task_id": "training:bee",
         "source_run_id": "frozen-training-run", "task_prompt": "craft beehive",
         "steps": [{"observation": "Crafting commands: craft beehive", "action": "get 1 oak logs",
                    "feedback": "Got 1 oak logs", "failed_action": False}]},
        {"memory_id": "bed-memory", "benchmark": "textcraft", "task_id": "training:bed",
         "source_run_id": "frozen-training-run", "task_prompt": "craft bed",
         "steps": [{"observation": "Crafting commands: craft bed", "action": "get 1 wool",
                    "feedback": "Got 1 wool", "failed_action": False}]},
    ]
    corpus = {"schema_version": 1, "kind": "successful_trajectory_retrieval_only", "records": records}
    index = {"schema_version": 2, "embedding_model": "fake-mpnet",
             "embedding_revision": "frozen-revision", "corpus_sha256": canonical_digest(corpus),
             "memory_ids": ["bee-memory", "bed-memory"],
             "task_vectors": [[1.0, 0.0], [0.0, 1.0]],
             "step_memory_ids": ["bee-memory", "bed-memory"],
             "step_vectors": [[1.0, 0.0], [0.0, 1.0]]}
    return FrozenTrajectoryRetrieval(corpus, index, FakeEmbedder(), 2,
                                     token_budget, lambda value: len(value.split()))


def context(**changes):
    item = task()
    values = dict(benchmark="textcraft", task_id=item["task_id"],
                  task_prompt=item["initial_observation"], observation=item["initial_observation"],
                  available_actions=(), history=(), decision=1,
                  original_action="get 1 spruce logs", current_action="get 1 spruce logs")
    values.update(changes)
    return SupportContext(**values)


def test_free_query_retrieval_returns_frozen_provenance_and_similarity():
    memory = toy_memory()
    hits, elapsed = memory.search("textcraft", "how to craft beehive", "both", 1)
    assert elapsed >= 0 and len(hits) == 1
    assert hits[0]["memory_id"] == "bee-memory"
    assert hits[0]["source_run_id"] == "frozen-training-run"
    assert hits[0]["source_task_id"] == "training:bee"
    assert hits[0]["similarity"] == hits[0]["task_similarity"] == hits[0]["step_similarity"] == 1.0
    second, _ = memory.search("textcraft", "how to craft bed", "steps", 1)
    assert second[0]["memory_id"] == "bed-memory"
    operation = RetrieveEpisodicMemory(toy_memory(token_budget=1))
    arguments = {"query": "how to craft beehive", "scope": "task", "top_k": 1}
    operation.validate_arguments(arguments, context())
    result = operation.execute(arguments, context())
    assert result.status == "empty" and result.information["hits"] == []
    assert result.metadata["corpus_sha256"] == memory.corpus_sha256
    assert result.cost.model_calls == 1 and result.cost.inference_seconds >= 0


def test_history_scopes_and_visible_state_do_not_query_simulator():
    state = context(
        history=(("inventory", "Inventory: 1 spruce logs"),
                 ("craft 1 beehive using 6 planks, 3 honeycomb", "Could not craft beehive"),
                 ("craft 1 beehive using 6 planks, 3 honeycomb", "Could not craft beehive")),
        observation="Could not craft beehive")
    history = InspectHistory()
    failures = history.execute({"scope": "failures", "limit": 2}, state)
    repeated = history.execute({"scope": "repeated", "limit": 2}, state)
    assert failures.status == "ok" and failures.information["total_matches"] == 2
    assert all(item["failed_action"] is True for item in failures.information["interactions"])
    assert [item["action_occurrence"] for item in repeated.information["interactions"]] == [1, 2]
    inspection = InspectState()
    inventory = inspection.execute({"scope": "inventory", "limit": 3}, state)
    recipes = inspection.execute({"scope": "recipes", "limit": 3, "query": "beehive"}, state)
    assert inventory.information["inventory_feedback"] == "Inventory: 1 spruce logs"
    assert recipes.information["displayed_recipes"] == ["craft 1 beehive using 6 planks, 3 honeycomb"]
    assert inspection.execute({"scope": "inventory", "limit": 2}, context()).status == "empty"
    assert inventory.metadata["no_environment_query"] is True


class TaskModel:
    def generate(self, messages):
        return {"text": "Action: get 1 spruce logs", "input_tokens": 20,
                "output_tokens": 7, "inference_seconds": 0.01}


class Episode:
    def __init__(self, item):
        self.item = item
        self.actions = []
        self.executed = []
    def reset(self):
        self.initial_observation = self.observation = self.task_prompt = self.item["initial_observation"]
    def step(self, action):
        self.executed.append(action)
        return {"observation": "Got 1 oak logs", "done": True, "success": True,
                "failed_action": False}
    def close(self):
        pass


class ScriptedModel:
    max_new_tokens = 1024
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.prompts = []
    def count_messages_tokens(self, messages):
        return 50
    def generate(self, messages, max_new_tokens=None):
        self.prompts.append(json.loads(messages[-1]["content"]))
        return {"text": next(self.outputs), "input_tokens": 50,
                "output_tokens": 20, "inference_seconds": 0.01,
                "generation_status": "complete"}


def choice(kind, **fields):
    return json.dumps({"kind": kind, **fields})


def run_case(model):
    config = settings()
    config["max_actions"] = config["max_decisions"] = 1
    item = task()
    episode = Episode(item)
    executor = make_support_executor(config, model, toy_memory())
    result = run_episode("textcraft", item, config, TaskModel(),
                         episode_factory=lambda *_: episode,
                         support_executor=executor, controller_model=model)
    return result, episode, model


def test_controller_composes_all_capabilities_and_stops_with_valid_action():
    model = ScriptedModel([
        choice("request", operation="retrieve_episodic_memory_v1",
               arguments={"query": "craft beehive resources", "scope": "both", "top_k": 1},
               objective="find a relevant past task", reason="need evidence"),
        choice("request", operation="inspect_state_v1",
               arguments={"scope": "recipes", "limit": 4, "query": "beehive"},
               objective="check visible recipe", reason="compare evidence"),
        choice("request", operation="inspect_history_v1",
               arguments={"scope": "failures", "limit": 2},
               objective="check failures", reason="review recent feedback"),
        choice("request", operation="reason_over_artifacts_v1",
               arguments={"objective": "choose a resource action from the visible evidence",
                          "output_type": "candidate_action"},
               objective="consider candidate", reason="synthesize evidence"),
        json.dumps({"output_type": "candidate_action", "content": "Action: get 1 oak logs"}),
        choice("execute", action=None, reason="enough information"),
    ])
    result, episode, model = run_case(model)
    trace = result["trace"][0]["support_trace"]
    requests = [event["request"]["operation"] for event in trace["events"] if "request" in event]
    assert requests == ["retrieve_episodic_memory_v1", "inspect_state_v1",
                        "inspect_history_v1", "reason_over_artifacts_v1"]
    assert model.prompts[1]["results"][0]["information"]["hits"][0]["memory_id"] == "bee-memory"
    assert model.prompts[2]["results"][1]["information"]["matching_recipe_count"] == 1
    assert episode.executed == ["get 1 oak logs"] and trace["termination_reason"] == "execute"
    assert result["controller_model_calls"] == 5 and result["support_model_calls"] == 2
    assert result["retrieval_seconds"] >= 0


def test_retrieval_result_changes_next_controller_choice():
    class ConditionalModel(ScriptedModel):
        def __init__(self):
            super().__init__([])
        def generate(self, messages, max_new_tokens=None):
            state = json.loads(messages[-1]["content"])
            self.prompts.append(state)
            if not state["results"]:
                raw = choice("request", operation="retrieve_episodic_memory_v1",
                             arguments={"query": "beehive", "scope": "task", "top_k": 1},
                             objective="find a precedent", reason="uncertain")
            elif len(state["results"]) == 1 and state["results"][0]["information"]["hits"]:
                raw = choice("request", operation="inspect_state_v1",
                             arguments={"scope": "recipes", "limit": 2, "query": "beehive"},
                             objective="compare visible recipe", reason="memory found")
            else:
                raw = choice("execute", action=None, reason="support complete")
            return {"text": raw, "input_tokens": 50, "output_tokens": 20,
                    "inference_seconds": 0.01, "generation_status": "complete"}
    result, episode, _ = run_case(ConditionalModel())
    events = result["trace"][0]["support_trace"]["events"]
    assert [event["request"]["operation"] for event in events if "request" in event] == [
        "retrieve_episodic_memory_v1", "inspect_state_v1"]
    assert episode.executed == ["get 1 spruce logs"]


def test_invalid_retrieval_arguments_and_missing_state_are_safe():
    operation = RetrieveEpisodicMemory(toy_memory())
    with pytest.raises(ValueError):
        operation.validate_arguments({"query": "beehive", "scope": "both", "top_k": 99}, context())
    with pytest.raises(ValueError):
        InspectState().validate_arguments({"scope": "hidden_inventory", "limit": 2}, context())
    assert InspectHistory().execute({"scope": "failures", "limit": 2}, context()).status == "empty"
    validate_config(settings(), "textcraft")


def test_controller_written_reasoning_objective_returns_plan_without_acting():
    model = ScriptedModel([json.dumps({"output_type": "plan",
                                      "content": "Collect logs; use the displayed plank recipe; inspect feedback."})])
    operation = ReasonOverArtifacts(settings(), model)
    arguments = {"objective": "reconsider prerequisites for the beehive",
                 "output_type": "plan"}
    operation.validate_arguments(arguments, context())
    reservation = operation.reservation(arguments, context())
    result = operation.execute(arguments, context())
    assert reservation.model_calls == result.cost.model_calls == 1
    assert result.status == "ok" and result.information["output_type"] == "plan"
    assert result.proposed_action is None and "beehive" in model.prompts[0]["requested_analysis"]["objective"]


def test_supplied_200_trajectory_artifacts_validate_without_model_inference():
    spec = settings()["autonomous_support"]["memory"]
    corpus_path = Path(spec["corpus_path"]).expanduser()
    index_path = Path(spec["index_path"]).expanduser()
    if not corpus_path.exists() or not index_path.exists():
        pytest.skip("Frozen M2-200 artifacts are not mounted")
    corpus = json.loads(corpus_path.read_text())
    index = json.loads(index_path.read_text())
    assert len(corpus["records"]) == 200
    assert index["corpus_sha256"] == canonical_digest(corpus)
    class IndexedVectorEmbedder:
        model_id = index["embedding_model"]
        revision = index["embedding_revision"]
        def encode(self, texts):
            return [index["task_vectors"][0] for _ in texts]
    memory = FrozenTrajectoryRetrieval(corpus, index, IndexedVectorEmbedder(),
                                       1, 1200, lambda value: len(value.split()))
    hits, _ = memory.search("textcraft", "fixture query", "task", 1)
    assert len(hits) == 1 and hits[0]["source_run_id"]
    assert "similarity" in hits[0] and "content" in hits[0]


def test_run_wires_frozen_artifacts_into_controller_without_gpu(tmp_path, monkeypatch):
    spec = settings()["autonomous_support"]["memory"]
    if not Path(spec["corpus_path"]).expanduser().exists() or not Path(spec["index_path"]).expanduser().exists():
        pytest.skip("Frozen M2-200 artifacts are not mounted")
    index = json.loads(Path(spec["index_path"]).expanduser().read_text())
    class IndexedVectorEmbedder:
        def __init__(self, model_id, revision):
            self.model_id, self.revision = model_id, revision
        def encode(self, texts):
            return [index["task_vectors"][0] for _ in texts]
    monkeypatch.setattr("agent_support.runner.MPNetEmbedder", IndexedVectorEmbedder)
    config = settings()
    config["max_actions"] = config["max_decisions"] = 1
    controller_model = ScriptedModel([
        choice("request", operation="retrieve_episodic_memory_v1",
               arguments={"query": "beehive", "scope": "task", "top_k": 1},
               objective="find evidence", reason="uncertain"),
        choice("execute", action=None, reason="enough information"),
    ])
    directory = run(config, json.loads(MANIFEST.read_text()), tmp_path, limit=1,
                    model=TaskModel(), controller_model=controller_model,
                    episode_factory=lambda benchmark, item, settings: Episode(item))
    summary = json.loads((directory / "summary.json").read_text())
    metadata = json.loads((directory / "run.json").read_text())
    snapshot = json.loads((directory / "config.json").read_text())
    episode_trace = json.loads(next((directory / "episodes").glob("*.json")).read_text())["trace"][0]
    assert summary["support_calls"] == 1 and summary["tasks"] == 1
    assert metadata["autonomous_memory_trajectories"] == 200
    assert metadata["autonomous_memory_index_sha256"] == canonical_digest(index)
    assert snapshot["model_id"] == "Qwen/Qwen2.5-7B-Instruct"
    assert snapshot["autonomous_support"]["model"]["model"]["model_id"] == "openai/gpt-oss-20b"
    assert episode_trace["support_trace"]["termination_reason"] == "execute"
    assert episode_trace["support_trace"]["events"][0]["request"]["operation"] == (
        "retrieve_episodic_memory_v1")
    assert episode_trace["support_trace"]["events"][0]["result"]["status"] == "ok"
    assert episode_trace["support_trace"]["environment_outcome"]["executed"] is True


def test_run_rejects_changed_frozen_index_digest_before_loading_models(tmp_path):
    config = settings()
    spec = config["autonomous_support"]["memory"]
    if not Path(spec["corpus_path"]).expanduser().exists() or not Path(spec["index_path"]).expanduser().exists():
        pytest.skip("Frozen M2-200 artifacts are not mounted")
    spec["index_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="artifact digest differs"):
        run(config, json.loads(MANIFEST.read_text()), tmp_path, limit=1,
            model=TaskModel(), controller_model=ScriptedModel([]),
            episode_factory=lambda benchmark, item, settings: Episode(item))
    assert list(tmp_path.iterdir()) == []
