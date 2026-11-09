import json

import pytest

from agent_support.analysis import compare_trajectory
from agent_support.common import canonical_digest, digest_text, write_json_atomic
from agent_support.runner import run, run_episode
from agent_support.trajectory_memory import (TrajectoryMemory, build_trajectory_corpus,
                                             build_trajectory_index, memory_messages)


class FakeEmbedder:
    model_id = "fake-mpnet"
    revision = "fixed"

    def encode(self, texts):
        return [[1.0, 0.0] if "plank" in text else [0.0, 1.0] for text in texts]


def task(seed, observation):
    return {"task_id": f"textcraft:seed:{seed}", "seed": seed,
            "initial_observation": observation, "observation_sha256": digest_text(observation)}


def manifest(partition, tasks):
    return {"schema_version": 1, "benchmark": "textcraft", "partition": partition, "tasks": tasks}


def make_run(root, condition, task_manifest, outcomes, config=None, provenance=None):
    root.mkdir()
    (root / "episodes").mkdir()
    config = config or {"benchmark": "textcraft", "condition": condition, "model_id": "fake"}
    write_json_atomic(root / "config.json", config)
    write_json_atomic(root / "manifest.json", task_manifest)
    for n, (item, success) in enumerate(zip(task_manifest["tasks"], outcomes)):
        write_json_atomic(root / "episodes" / f"{n:04d}.json", {
            "task_id": item["task_id"], "task_prompt": item["initial_observation"],
            "initial_observation": item["initial_observation"], "success": success,
            "status": "success" if success else "max_actions", "environment_actions": 1,
            "model_decisions": 1, "parser_errors": 0, "failed_actions": 0,
            "input_tokens": 10, "output_tokens": 2, "inference_seconds": 0.1,
            "trace": [{"decision": 1, "action": "get 1 log", "input_observation": "Start",
                       "output_observation": "Got 1 log", "outcome": {"failed_action": False}}]})
    write_json_atomic(root / "run.json", {
        "status": "complete", "run_id": root.name, "benchmark": "textcraft", "condition": condition,
        "partition": task_manifest["partition"], "selected_task_count": len(outcomes),
        "total_manifest_task_count": len(outcomes), "limit": None,
        "selected_manifest_sha256": canonical_digest(task_manifest),
        "manifest_sha256": canonical_digest(task_manifest), "config_sha256": canonical_digest(config),
        "python_version": "3.13", "package_versions": {"textcraft": "0.0.3"}, **(provenance or {})})
    return root


def test_successful_trajectories_only_and_exclusion(tmp_path):
    training = manifest("training", [task(1000, "Goal: craft plank."), task(1001, "Goal: craft paper.")])
    source = make_run(tmp_path / "train", "baseline", training, [True, False])
    dev = tmp_path / "dev.json"
    heldout = tmp_path / "heldout.json"
    write_json_atomic(dev, manifest("development", [task(42, "Goal: craft stone.")]))
    write_json_atomic(heldout, manifest("heldout", [task(43, "Goal: craft iron.")]))
    corpus = build_trajectory_corpus([source], [dev, heldout])
    assert len(corpus["records"]) == 1
    assert corpus["records"][0]["steps"][0]["action"] == "get 1 log"
    assert corpus["records"][0]["task_prompt"] == "craft plank."
    write_json_atomic(dev, manifest("development", [task(42, "Goal: craft plank.")]))
    with pytest.raises(ValueError, match="overlap"):
        build_trajectory_corpus([source], [dev, heldout])


def test_shared_index_retrieval_budget_and_prompt():
    records = [{"memory_id": "plank", "task_id": "train:1", "benchmark": "textcraft",
                "task_prompt": "craft plank.", "steps": [{"observation": "Start", "action": "get 1 log", "feedback": "Got log",
                                                          "failed_action": False}]},
               {"memory_id": "paper", "task_id": "train:2", "benchmark": "textcraft",
                "task_prompt": "craft paper.", "steps": [{"observation": "Start", "action": "get 1 cane", "feedback": "Got cane",
                                                          "failed_action": False}]}]
    corpus = {"records": records}
    index = build_trajectory_index(corpus, FakeEmbedder())
    memory = TrajectoryMemory(corpus, index, FakeEmbedder(), 2, 100, lambda text: len(text.split()))
    task_hits, _ = memory.retrieve("textcraft", "Goal: craft plank.")
    proposal_hits, _ = memory.retrieve("textcraft", "Goal: craft plank.", "Start", "get 1 log")
    assert task_hits[0]["memory_id"] == proposal_hits[0]["memory_id"] == "plank"
    assert "Got log" in task_hits[0]["content"]
    rendered = memory_messages([{"role": "user", "content": "Choose the next command.\nAction:"}],
                               proposal_hits, "get 1 log")[-1]["content"]
    assert "Your proposed action was: get 1 log" in rendered
    assert rendered.endswith("\nAction:")
    tiny = TrajectoryMemory(corpus, index, FakeEmbedder(), 2, 1, lambda text: len(text.split()))
    assert tiny.retrieve("textcraft", "Goal: craft plank.")[0] == []
    with pytest.raises(ValueError, match="does not match"):
        TrajectoryMemory({"records": records[:1]}, index, FakeEmbedder(), 2, 100, len)


def test_action_context_can_change_trajectory_ranking():
    class ActionEmbedder(FakeEmbedder):
        def encode(self, texts):
            return [[0.0, 1.0] if "Action: get 1 cane" in text else [1.0, 0.0]
                    for text in texts]

    records = [{"memory_id": memory_id, "task_id": f"train:{memory_id}",
                "benchmark": "textcraft", "task_prompt": "craft plank.",
                "steps": [{"observation": "Start", "action": action,
                           "feedback": "Accepted", "failed_action": False}]}
               for memory_id, action in (("a", "get 1 log"), ("b", "get 1 cane"))]
    corpus = {"records": records}
    embedder = ActionEmbedder()
    memory = TrajectoryMemory(corpus, build_trajectory_index(corpus, embedder), embedder,
                              1, 100, lambda text: len(text.split()))
    assert memory.retrieve("textcraft", "Goal: craft plank.")[0][0]["memory_id"] == "a"
    assert memory.retrieve("textcraft", "Goal: craft plank.", "Start", "get 1 cane")[0][0]["memory_id"] == "b"


def test_retrieval_only_uses_one_model_call_and_m2_revises():
    class FakeEpisode:
        task_prompt = initial_observation = observation = "Goal: craft plank."
        actions = []
        def reset(self):
            pass
        def step(self, action):
            return {"observation": "Done", "done": True, "success": action == "get 1 log", "failed_action": False}
        def close(self):
            pass

    class FakeModel:
        def __init__(self, outputs):
            self.outputs = iter(outputs)
            self.calls = 0
        def generate(self, messages):
            self.calls += 1
            return {"text": next(self.outputs), "input_tokens": 10, "output_tokens": 3, "inference_seconds": 0.1}

    class FakeMemory:
        def retrieve(self, *_args):
            return [{"memory_id": "a", "content": "Past successful task: craft plank.\n1. Action: get 1 log"}], 0.01

    base = {"max_actions": 2, "max_decisions": 2, "history_window": 2}
    model = FakeModel(["Action: get 1 log"])
    task_result = run_episode("textcraft", {"task_id": "textcraft:seed:42"},
                              {**base, "condition": "expel_retrieval_only_v1"}, model,
                              episode_factory=lambda *_: FakeEpisode(), trajectory_support=FakeMemory())
    assert task_result["success"] and model.calls == 1
    assert task_result["support_calls"] == 1 and task_result["support_model_calls"] == 0
    model = FakeModel(["Action: get 1 plank", "Action: get 1 log"])
    action_result = run_episode("textcraft", {"task_id": "textcraft:seed:42"},
                                {**base, "condition": "m2_trajectory_v1"}, model,
                                episode_factory=lambda *_: FakeEpisode(), trajectory_support=FakeMemory())
    assert action_result["success"] and model.calls == 2
    assert action_result["actions_changed_by_support"] == 1


def test_trajectory_comparison_requires_shared_artifacts(tmp_path):
    dev = manifest("development", [task(42, "Goal: craft plank.")])
    base_config = {"benchmark": "textcraft", "condition": "baseline", "model_id": "fake"}
    support_keys = {"trajectory_corpus": "one", "trajectory_index": "two",
                    "trajectory_top_k": 2, "trajectory_token_budget": 1800}
    provenance = {"trajectory_corpus_sha256": "abc", "trajectory_index_sha256": "def",
                  "memory_selection_policy": "v1"}
    a = make_run(tmp_path / "a", "baseline", dev, [False], base_config)
    b = make_run(tmp_path / "b", "expel_retrieval_only_v1", dev, [True],
                 {**base_config, "condition": "expel_retrieval_only_v1", **support_keys}, provenance)
    c = make_run(tmp_path / "c", "m2_trajectory_v1", dev, [True],
                 {**base_config, "condition": "m2_trajectory_v1", **support_keys}, provenance)
    assert compare_trajectory(a, b)["paired"]["helped"] == 1
    assert compare_trajectory(b, c)["paired"]["same"] == 1
    run_meta = json.loads((c / "run.json").read_text())
    run_meta["support_policy_id"] = "different_policy"
    other_meta = json.loads((b / "run.json").read_text())
    other_meta["support_policy_id"] = "fixed_by_condition_v1"
    write_json_atomic(b / "run.json", other_meta)
    write_json_atomic(c / "run.json", run_meta)
    with pytest.raises(ValueError, match="support policies differ"):
        compare_trajectory(b, c)
    run_meta.pop("support_policy_id")
    write_json_atomic(c / "run.json", run_meta)
    with pytest.raises(ValueError, match="support policies differ"):
        compare_trajectory(b, c)
    run_meta["support_policy_id"] = "fixed_by_condition_v1"
    run_meta["package_versions"]["sentence-transformers"] = "different"
    write_json_atomic(c / "run.json", run_meta)
    with pytest.raises(ValueError, match="Package version differs"):
        compare_trajectory(b, c)
    run_meta["package_versions"].pop("sentence-transformers")
    run_meta["trajectory_index_sha256"] = "changed"
    write_json_atomic(c / "run.json", run_meta)
    with pytest.raises(ValueError, match="Trajectory memory differs"):
        compare_trajectory(b, c)


def test_runner_rejects_content_overlap_and_records_trajectory_hashes(monkeypatch, tmp_path):
    dev_task = task(42, "Goal: craft plank.")
    dev = manifest("development", [dev_task])
    record = {"memory_id": "train-1", "benchmark": "textcraft", "task_id": "textcraft:seed:1000",
              "task_fingerprint": digest_text("Goal: craft paper."),
              "task_prompt": "craft paper.", "initial_observation": "Goal: craft paper.",
              "steps": [{"observation": "Start", "action": "get 1 sugar cane", "feedback": "Got cane", "failed_action": False}]}
    corpus = {"records": [record], "excluded_manifest_sha256": {"dev": canonical_digest(dev)}}
    index = build_trajectory_index(corpus, FakeEmbedder())
    corpus_path, index_path = tmp_path / "corpus.json", tmp_path / "index.json"
    write_json_atomic(corpus_path, corpus)
    write_json_atomic(index_path, index)
    monkeypatch.setattr("agent_support.runner.MPNetEmbedder", lambda *_args, **_kwargs: FakeEmbedder())
    config = {"benchmark": "textcraft", "condition": "expel_retrieval_only_v1", "model_id": "fake",
              "quantization": "nf4_4bit", "max_actions": 1, "max_decisions": 1,
              "history_window": 1, "max_new_tokens": 10, "trajectory_corpus": str(corpus_path),
              "trajectory_index": str(index_path), "trajectory_top_k": 1,
              "trajectory_token_budget": 100}

    class FakeModel:
        def generate(self, messages):
            return {"text": "Action: get 1 log", "input_tokens": 10,
                    "output_tokens": 3, "inference_seconds": 0.1}

    class FakeEpisode:
        task_prompt = initial_observation = observation = "Goal: craft plank."
        actions = []
        def reset(self):
            pass
        def step(self, action):
            return {"observation": "Done", "done": True, "success": True, "failed_action": False}
        def close(self):
            pass

    output = run(config, dev, tmp_path / "results", model=FakeModel(),
                 episode_factory=lambda *_: FakeEpisode())
    metadata = json.loads((output / "run.json").read_text())
    assert metadata["status"] == "complete"
    assert metadata["trajectory_corpus_sha256"] == canonical_digest(corpus)
    assert metadata["trajectory_index_sha256"] == canonical_digest(index)
    assert metadata["support_policy_id"] == "fixed_by_condition_v1"
    assert json.loads((output / "summary.json").read_text())["support_model_calls"] == 0
    corpus["records"][0]["task_fingerprint"] = dev_task["observation_sha256"]
    write_json_atomic(corpus_path, corpus)
    with pytest.raises(ValueError, match="task content"):
        run(config, dev, tmp_path / "results", model=FakeModel(),
            episode_factory=lambda *_: FakeEpisode())
