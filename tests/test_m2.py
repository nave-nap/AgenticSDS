import json

import pytest

from agent_support.common import canonical_digest, digest_text, write_json_atomic
from agent_support.memory import ActionMemory, build_corpus, build_index, related_action, retrieval_text, support_messages
from agent_support.manifests import write_training_manifest
from agent_support.runner import run, run_episode


def task(seed, observation):
    return {"task_id": f"textcraft:seed:{seed}", "seed": seed,
            "initial_observation": observation, "observation_sha256": digest_text(observation)}


def manifest(partition, one_task):
    return {"schema_version": 1, "benchmark": "textcraft", "partition": partition,
            "tasks": [one_task]}


def test_corpus_rejects_development_overlap(tmp_path):
    training_task = task(1000, "Crafting commands:\ncraft 1 plank using 1 log\n\nGoal: craft plank.")
    source = tmp_path / "training_run"
    source.mkdir()
    training_manifest = manifest("training", training_task)
    from agent_support.common import canonical_digest
    write_json_atomic(source / "run.json", {
        "status": "complete", "run_id": "train-1", "benchmark": "textcraft", "condition": "baseline",
        "selected_task_count": 1, "total_manifest_task_count": 1, "limit": None,
        "selected_manifest_sha256": canonical_digest(training_manifest),
    })
    write_json_atomic(source / "config.json", {"condition": "baseline"})
    metadata = json.loads((source / "run.json").read_text())
    metadata["config_sha256"] = canonical_digest({"condition": "baseline"})
    write_json_atomic(source / "run.json", metadata)
    write_json_atomic(source / "manifest.json", training_manifest)
    (source / "episodes").mkdir()
    write_json_atomic(source / "episodes" / "0000.json", {
        "task_id": training_task["task_id"], "task_prompt": training_task["initial_observation"],
        "success": False, "trace": [{"decision": 1, "action": "get 1 log",
                                      "input_observation": "Start", "output_observation": "Got 1 log",
                                      "outcome": {"failed_action": False, "success": False}}],
    })
    dev = tmp_path / "dev.json"
    heldout = tmp_path / "heldout.json"
    write_json_atomic(dev, manifest("development", task(42, "Other task")))
    write_json_atomic(heldout, manifest("heldout", task(43, "Different task")))
    corpus = build_corpus([source], [dev, heldout])
    assert len(corpus["records"]) == 1
    assert corpus["records"][0]["task_id"] == training_task["task_id"]
    assert retrieval_text(corpus["records"][0]).startswith("Proposed action: get 1 log")
    write_json_atomic(dev, manifest("development", task(42, training_task["initial_observation"])))
    with pytest.raises(ValueError, match="overlap"):
        build_corpus([source], [dev, heldout])


def test_training_manifest_skips_duplicate_task_content(monkeypatch, tmp_path):
    dev = tmp_path / "dev.json"
    heldout = tmp_path / "heldout.json"
    write_json_atomic(dev, manifest("development", task(42, "Same task")))
    write_json_atomic(heldout, manifest("heldout", task(43, "Held-out task")))
    monkeypatch.setattr("agent_support.manifests.build_textcraft", lambda start, count: [
        task(1000, "Same task"), task(1001, "New task"), task(1002, "New task"),
        task(1003, "Another task")])
    result = write_training_manifest("textcraft", 2, tmp_path / "train.json", [dev, heldout])
    assert [item["task_id"] for item in result["tasks"]] == ["textcraft:seed:1001", "textcraft:seed:1003"]
    with pytest.raises(ValueError, match="Both development and held-out"):
        write_training_manifest("textcraft", 1, tmp_path / "other.json", [dev])


def test_index_and_support_prompt_keep_action_visible():
    record = {"memory_id": "a", "task_id": "textcraft:seed:1000",
              "task_prompt": "Crafting commands:\n" + "x\n" * 100 + "\nGoal: craft plank.",
              "observation": "Start", "action": "get 1 log", "feedback": "Got 1 log",
              "failed_action": False, "episode_success": True, "benchmark": "textcraft"}
    corpus = {"records": [record]}

    class FakeEmbedder:
        model_id = "fake"

        def encode(self, texts):
            return [[1.0, 0.0] for _ in texts]

    index = build_index(corpus, FakeEmbedder())
    assert index["memory_ids"] == ["a"]
    hits, _ = ActionMemory(corpus, index, FakeEmbedder()).retrieve(
        "textcraft", "Goal: craft plank.", "Start", "get 1 log")
    assert [hit["memory_id"] for hit in hits] == ["a"]
    messages = support_messages([{"role": "system", "content": "base"},
                                 {"role": "user", "content": "original"}], "get 1 log",
                                [{**record, "similarity": 0.9}], "textcraft")
    assert messages[0]["content"] == "base"
    assert "craft plank" in messages[-1]["content"]
    assert "Got 1 log" in messages[-1]["content"]


def test_retrieval_uses_only_successful_matching_actions():
    base = {"benchmark": "textcraft", "task_prompt": "Goal: craft plank.",
            "observation": "Start", "feedback": "Got 1 log"}
    records = [
        {**base, "memory_id": "good", "task_id": "textcraft:seed:1000", "action": "get 1 log",
         "failed_action": False, "episode_success": True},
        {**base, "memory_id": "failed", "task_id": "textcraft:seed:1001", "action": "get 1 log",
         "failed_action": True, "episode_success": True},
        {**base, "memory_id": "unfinished", "task_id": "textcraft:seed:1002", "action": "get 1 log",
         "failed_action": False, "episode_success": False},
        {**base, "memory_id": "wrong-item", "task_id": "textcraft:seed:1003", "action": "get 1 sand",
         "failed_action": False, "episode_success": True},
    ]

    class FakeEmbedder:
        model_id = "fake"

        def encode(self, texts):
            return [[1.0, 0.0] for _ in texts]

    corpus = {"records": records}
    memory = ActionMemory(corpus, build_index(corpus, FakeEmbedder()), FakeEmbedder())
    hits, _ = memory.retrieve("textcraft", "Goal: craft plank.", "Start", "get 2 log")
    assert [hit["memory_id"] for hit in hits] == ["good"]
    no_hits, _ = memory.retrieve("textcraft", "Goal: craft plank.", "Start", "craft 1 plank using 1 log")
    assert no_hits == []
    assert related_action("alfworld", "go to drawer 1", "go to drawer 2")
    assert not related_action("alfworld", "go to drawer 1", "go to fridge 1")


def test_m2_reconsideration_logs_extra_call_and_changed_action():
    class FakeModel:
        def __init__(self):
            self.outputs = iter(["Action: get 1 plank", "Action: get 1 log"])

        def generate(self, messages):
            return {"text": next(self.outputs), "input_tokens": 10,
                    "output_tokens": 4, "inference_seconds": 0.1}

    class FakeEpisode:
        task_prompt = "Goal: craft plank."
        initial_observation = observation = "Goal: craft plank."
        actions = []

        def reset(self):
            pass

        def step(self, action):
            assert action == "get 1 log"
            return {"observation": "Success", "done": True, "success": True, "failed_action": False}

        def close(self):
            pass

    class FakeSupport:
        def retrieve(self, benchmark, task_prompt, observation, action):
            assert action == "get 1 plank"
            return ([{"memory_id": "a", "similarity": 0.9, "task_prompt": "Goal: craft plank.",
                      "observation": "Start", "action": "get 1 log", "feedback": "Got 1 log",
                      "failed_action": False, "episode_success": True}], 0.01)

    config = {"max_actions": 3, "max_decisions": 3, "history_window": 2}
    result = run_episode("textcraft", {"task_id": "textcraft:seed:42"}, config, FakeModel(),
                         episode_factory=lambda *_: FakeEpisode(), support=FakeSupport())
    assert result["success"]
    assert result["model_decisions"] == 1
    assert result["model_calls"] == 2
    assert result["support_calls"] == 1
    assert result["actions_changed_by_support"] == 1
    assert result["input_tokens"] == 20
    assert result["trace"][0]["proposed_action"] == "get 1 plank"
    assert result["trace"][0]["action"] == "get 1 log"


def test_no_relevant_memory_keeps_original_proposal():
    class OneCallModel:
        calls = 0

        def generate(self, messages):
            self.calls += 1
            assert self.calls == 1
            return {"text": "Action: get 1 log", "input_tokens": 10,
                    "output_tokens": 4, "inference_seconds": 0.1}

    class FakeEpisode:
        initial_observation = observation = task_prompt = "Goal: craft plank."
        actions = []

        def reset(self):
            pass

        def step(self, action):
            assert action == "get 1 log"
            return {"observation": "Done", "done": True, "success": True, "failed_action": False}

        def close(self):
            pass

    class NoHits:
        def retrieve(self, benchmark, task_prompt, observation, action):
            return [], 0.0

    result = run_episode("textcraft", {"task_id": "textcraft:seed:42"},
                         {"max_actions": 1, "max_decisions": 1, "history_window": 1},
                         OneCallModel(), episode_factory=lambda *_: FakeEpisode(), support=NoHits())
    assert result["success"]
    assert result["support_calls"] == 1
    assert result["support_model_calls"] == 0
    assert result["model_calls"] == 1
    assert result["actions_changed_by_support"] == 0
    assert result["trace"][0]["action"] == result["trace"][0]["proposed_action"]
    assert result["trace"][0]["support"]["fallback_reason"] == "no_relevant_memory"


def test_supported_run_requires_exclusion_proof_and_records_memory_hashes(monkeypatch, tmp_path):
    dev_manifest = manifest("development", task(42, "Goal: craft plank."))
    record = {"memory_id": "train-step", "task_id": "textcraft:seed:1000", "benchmark": "textcraft",
              "task_prompt": "Goal: craft plank.", "observation": "Start", "action": "get 1 log",
              "feedback": "Got 1 log", "failed_action": False, "episode_success": True}
    corpus = {"records": [record], "excluded_manifest_sha256": {"dev": canonical_digest(dev_manifest)}}

    class FakeEmbedder:
        model_id = "fake"
        revision = None

        def __init__(self, *_args, **_kwargs):
            pass

        def encode(self, texts):
            return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr("agent_support.runner.SentenceEmbedder", FakeEmbedder)
    index = build_index(corpus, FakeEmbedder())
    corpus_path = tmp_path / "corpus.json"
    index_path = tmp_path / "index.json"
    write_json_atomic(corpus_path, corpus)
    write_json_atomic(index_path, index)
    config = {"benchmark": "textcraft", "condition": "m2_always", "model_id": "fake",
              "quantization": "nf4_4bit", "max_actions": 2, "max_decisions": 2,
              "history_window": 2, "max_new_tokens": 10, "memory_corpus": str(corpus_path),
              "memory_index": str(index_path), "memory_top_k": 1}

    class FakeModel:
        def __init__(self):
            self.outputs = iter(["Action: get 1 log", "Action: get 2 log"])

        def generate(self, messages):
            return {"text": next(self.outputs), "input_tokens": 10,
                    "output_tokens": 4, "inference_seconds": 0.1}

    class FakeEpisode:
        initial_observation = observation = task_prompt = "Goal: craft plank."
        actions = []

        def reset(self):
            pass

        def step(self, action):
            assert action == "get 2 log"
            return {"observation": "Done", "done": True, "success": True, "failed_action": False}

        def close(self):
            pass

    output = run(config, dev_manifest, tmp_path / "results", model=FakeModel(),
                 episode_factory=lambda *_: FakeEpisode())
    metadata = json.loads((output / "run.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    assert metadata["status"] == "complete"
    assert metadata["memory_corpus_sha256"] == canonical_digest(corpus)
    assert summary["model_calls"] == 2
    assert summary["actions_changed_by_support"] == 1
    corpus["excluded_manifest_sha256"] = {}
    write_json_atomic(corpus_path, corpus)
    with pytest.raises(ValueError, match="not built with this evaluation manifest"):
        run(config, dev_manifest, tmp_path / "results", model=FakeModel(),
            episode_factory=lambda *_: FakeEpisode())
