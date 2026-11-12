"""CPU-only checks for the development diagnostic and adapter entry point."""

import hashlib
import json
from collections import Counter
from pathlib import Path

from agent_support import delegation_diagnostic as diagnostic
from agent_support.model import MODEL_REGISTRY


def _jsonl(path, values):
    path.write_text("".join(json.dumps(value) + "\n" for value in values))


def _dataset_manifest(root, *names):
    (root / "dataset.json").write_text(json.dumps({
        "schema_version": 1, "partition": "development",
        "files_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                         for name in names}}))


def test_saved_diagnostic_has_40_blinded_inputs_and_distinct_labels():
    root = Path("diagnostics/textcraft_delegation_v1")
    metadata = diagnostic._verify_dataset_files(root / "inputs.jsonl",
                                                 root / "labels.jsonl", root / "replay.jsonl")
    inputs = diagnostic._jsonl(root / "inputs.jsonl")
    labels = diagnostic._jsonl(root / "labels.jsonl")
    prefixes = diagnostic._jsonl(root / "replay.jsonl")
    assert metadata["count"] == len(inputs) == len(labels) == len(prefixes) == 40
    assert len({item["task_id"] for item in inputs}) == 20
    assert Counter(item["category"] for item in labels) == diagnostic.CATEGORY_COUNTS
    assert Counter(item["qwen_action"].split()[0] for item in labels
                   if item["category"] == "productive_controls") == {"get": 4, "craft": 4}
    assert {item["sample_id"] for item in inputs} == {item["sample_id"] for item in labels}
    for item in inputs:
        assert set(item) == {"sample_id", "task_id", "decision", "task_goal",
                             "current_observation", "visible_history", "messages"}
        assert len(item["visible_history"]) <= 8
        assert item["messages"][-1]["content"].endswith("Action:")
        assert "qwen_action" not in item and "qwen_outcome" not in item
        assert not any(key in item for key in ("future_feedback", "reference_solution", "hidden_state"))


def test_prediction_passes_only_saved_messages_to_registered_adapter(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    root.mkdir()
    messages = [{"role": "user", "content": "Choose one action. Action:"}]
    _jsonl(root / "inputs.jsonl", [{"sample_id": "s1", "task_id": "task", "decision": 2,
                                     "messages": messages}])
    _dataset_manifest(root, "inputs.jsonl")
    config = {"model_type": "mock", "model_id": "mock/model", "model_revision": "revision"}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    class Model:
        def metadata(self):
            return {"model_id": "mock/model", "model_revision": "revision"}
        def generate(self, supplied):
            assert supplied == messages
            return {"text": "Action: get 1 sand", "input_tokens": 5,
                    "output_tokens": 6, "inference_seconds": 0.1}
    monkeypatch.setitem(MODEL_REGISTRY, "mock", lambda _: Model())
    monkeypatch.setattr(diagnostic, "git_state", lambda: {"git_commit": "test"})
    output = tmp_path / "predictions"
    diagnostic.predict(config_path, root / "inputs.jsonl", output)
    assert json.loads((output / "run.json").read_text())["status"] == "complete"
    assert json.loads((output / "000_s1.json").read_text())["action"] == "get 1 sand"


def test_replay_compares_two_resets_and_reports_only_immediate_feedback(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    root.mkdir()
    _jsonl(root / "inputs.jsonl", [{"sample_id": "s1", "current_observation": "after first"}])
    bad = {"observation": "Could not find item", "failed_action": True,
           "done": False, "success": False}
    _jsonl(root / "labels.jsonl", [{"sample_id": "s1", "category": "unavailable_items",
                                     "qwen_action": "get 1 item", "qwen_outcome": bad}])
    _jsonl(root / "replay.jsonl", [{"sample_id": "s1", "task": {"task_id": "task"},
                                     "prior_steps": [{"action": "get 1 sand", "observation": "after first",
                                                      "failed_action": False}]}])
    _dataset_manifest(root, "inputs.jsonl", "labels.jsonl", "replay.jsonl")
    class Episode:
        def __init__(self, task):
            self.observation = "initial"
        def reset(self):
            self.observation = "initial"
        def step(self, action):
            if action == "get 1 sand":
                self.observation = "after first"
                return {"observation": "after first", "failed_action": False,
                        "done": False, "success": False}
            if action == "get 1 item":
                return bad
            return {"observation": "Got 1 useful item", "failed_action": False,
                    "done": False, "success": False}
        def close(self):
            pass
    monkeypatch.setattr(diagnostic, "TextCraftEpisode", Episode)
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    (predictions / "run.json").write_text(json.dumps({
        "status": "complete", "git_commit": "test",
        "inputs_sha256": hashlib.sha256((root / "inputs.jsonl").read_bytes()).hexdigest(),
        "selected_sample_ids": ["s1"]}))
    (predictions / "000_s1.json").write_text(json.dumps({
        "sample_id": "s1", "action": "get 1 useful item"}))
    saved_run = json.loads((predictions / "run.json").read_text())
    saved_run["prediction_file_sha256"] = {
        "000_s1.json": hashlib.sha256((predictions / "000_s1.json").read_bytes()).hexdigest()}
    (predictions / "run.json").write_text(json.dumps(saved_run))
    report = diagnostic.replay(predictions, root / "inputs.jsonl", root / "labels.jsonl",
                               root / "replay.jsonl", tmp_path / "replay_report.json")
    assert report["scope"] == "immediate_action_only"
    assert report["by_category"]["unavailable_items"]["immediate_failure_avoided"] == 1
    assert report["observations"][0]["action_changed"] is True
