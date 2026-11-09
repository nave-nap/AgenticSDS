import copy
import json

import pytest

from agent_support.analysis import compare, load_run
from agent_support.common import digest_text
from agent_support.manifests import build_textcraft, write_splits
from agent_support.policy import parse_alfworld, parse_textcraft
from agent_support.runner import run


CONFIG = {
    "benchmark": "textcraft",
    "condition": "baseline",
    "model_id": "Qwen/Qwen2.5-7B-Instruct",
    "quantization": "nf4_4bit",
    "max_actions": 3,
    "max_decisions": 3,
    "history_window": 2,
    "max_new_tokens": 100,
}


class FakeModel:
    def __init__(self, outputs):
        self.outputs = iter(outputs)

    def generate(self, messages):
        return {"text": next(self.outputs), "input_tokens": 10, "output_tokens": 4, "inference_seconds": 0.1}


class FakeEpisode:
    def __init__(self, task):
        self.task = task
        self.initial_observation = ""
        self.observation = ""
        self.task_prompt = ""
        self.actions = []
        self.closed = False

    def reset(self):
        self.initial_observation = self.observation = self.task_prompt = self.task["initial_observation"]

    def step(self, action):
        self.observation = "Goal reached" if action == "get 1 wood" else "Could not craft"
        return {"observation": self.observation, "reward": 1.0 if action == "get 1 wood" else 0.0,
                "done": action == "get 1 wood", "success": action == "get 1 wood", "failed_action": action != "get 1 wood"}

    def close(self):
        self.closed = True


def fake_factory(benchmark, task, config):
    assert benchmark == "textcraft"
    return FakeEpisode(task)


def manifest():
    observation = "Craft wood"
    return {"schema_version": 1, "benchmark": "textcraft", "partition": "development", "tasks": [
        {"task_id": "textcraft:seed:42", "seed": 42, "initial_observation": observation,
         "observation_sha256": digest_text(observation)}
    ]}


def test_parsers_reject_invalid_actions():
    assert parse_alfworld("2", ["look", "open drawer"]) == "open drawer"
    assert parse_alfworld("Action: 2", ["look", "open drawer"]) is None
    assert parse_alfworld("3", ["look", "open drawer"]) is None
    assert parse_textcraft("Action: get 1 oak_log") == "get 1 oak log"
    assert parse_textcraft("Action: get 1 wood using 1 stone") is None
    assert parse_textcraft("Action: craft 1 plank using 1 wood") == "craft 1 plank using 1 wood"


def test_run_logs_parse_decision_without_environment_action(tmp_path):
    output = run(CONFIG, manifest(), tmp_path, model=FakeModel(["bad output", "Action: get 1 wood"]), episode_factory=fake_factory)
    metadata, config, task_manifest, episodes = load_run(output)
    result = next(iter(episodes.values()))
    assert metadata["status"] == "complete"
    assert result["success"]
    assert result["model_decisions"] == 2
    assert result["environment_actions"] == 1
    assert result["parser_errors"] == 1
    assert result["input_tokens"] == 20
    assert len(result["trace"]) == 2


def test_alfworld_numbered_choice_uses_same_runner(tmp_path):
    config = dict(CONFIG, benchmark="alfworld", max_new_tokens=32, alfworld_config=str(tmp_path / "base.yaml"))
    (tmp_path / "base.yaml").write_text("fixed: true\n")
    task_manifest = {"schema_version": 1, "benchmark": "alfworld", "tasks": [
        {"task_id": "alfworld:eval_in_distribution:one/game.tw-pddl", "split": "eval_in_distribution",
         "game_relpath": "one/game.tw-pddl", "game_sha256": "abc"}
    ]}

    class FakeALF(FakeEpisode):
        def reset(self):
            super().reset()
            self.actions = ["look", "open drawer"]

        def step(self, action):
            assert action == "open drawer"
            return {"observation": "Done", "score": 1.0, "done": True, "success": True, "failed_action": False}

    output = run(config, task_manifest, tmp_path, model=FakeModel(["2"]),
                 episode_factory=lambda benchmark, task, config: FakeALF({**task, "initial_observation": "Your task is to: open drawer"}))
    _, _, _, episodes = load_run(output)
    assert next(iter(episodes.values()))["environment_actions"] == 1


def test_compare_rejects_mismatched_task_sets_and_incomplete_runs(tmp_path):
    left = run(CONFIG, manifest(), tmp_path, model=FakeModel(["Action: get 1 wood"]), episode_factory=fake_factory)
    right = run(CONFIG, manifest(), tmp_path, model=FakeModel(["Action: get 1 wood"]), episode_factory=fake_factory)
    assert compare(left, right)["left"]["success_rate"] == 1.0
    task_manifest = json.loads((right / "manifest.json").read_text())
    task_manifest["tasks"][0]["task_id"] = "different"
    (right / "manifest.json").write_text(json.dumps(task_manifest))
    # Metadata still points to the original manifest, but result IDs reveal tampering.
    with pytest.raises(ValueError):
        compare(left, right)
    run_meta = json.loads((left / "run.json").read_text())
    run_meta["status"] = "running"
    (left / "run.json").write_text(json.dumps(run_meta))
    with pytest.raises(ValueError):
        load_run(left)


def test_textcraft_generation_fails_if_fresh_instances_disagree(monkeypatch):
    observations = iter(["Task A", "Task B"])
    monkeypatch.setattr("agent_support.manifests.textcraft_observation", lambda seed: next(observations))
    with pytest.raises(RuntimeError, match="different tasks"):
        build_textcraft(42, 1)


def test_existing_manifest_cannot_be_silently_replaced(tmp_path):
    tasks = manifest()["tasks"] + [{"task_id": "textcraft:seed:43", "seed": 43,
                                    "initial_observation": "Another task", "observation_sha256": digest_text("Another task")}]
    write_splits("textcraft", tasks, 1, 1, tmp_path, seed=42)
    changed = copy.deepcopy(tasks)
    changed[0]["initial_observation"] = "Changed"
    changed[0]["observation_sha256"] = digest_text("Changed")
    with pytest.raises(FileExistsError):
        write_splits("textcraft", changed, 1, 1, tmp_path, seed=42)
