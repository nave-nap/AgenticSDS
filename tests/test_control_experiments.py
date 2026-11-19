from __future__ import annotations

import json

import pytest

from agent_support import control_experiments as controls
from agent_support.common import canonical_digest


class Episode:
    def __init__(self):
        self.initial_observation = "Crafting commands:\ncraft 1 plank using 1 log\n\nGoal: craft plank."
        self.task_prompt = self.initial_observation
        self.observation = self.initial_observation
        self.steps = []

    def reset(self):
        pass

    def step(self, action):
        self.steps.append(action)
        if action == "get 1 log":
            self.observation = "Collected log"
            return {"observation": self.observation, "done": False, "success": False,
                    "failed_action": False}
        if action == "craft 1 plank using 1 log" and "get 1 log" in self.steps:
            self.observation = "Success"
            return {"observation": self.observation, "done": True, "success": True,
                    "failed_action": False}
        self.observation = "Could not craft"
        return {"observation": self.observation, "done": False, "success": False,
                "failed_action": True}

    def close(self):
        pass


class Model:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.messages = []

    def generate(self, messages, max_new_tokens=None):
        self.messages.append(messages)
        return {"text": next(self.outputs), "raw_completion": "full Harmony completion",
                "generation_status": "complete", "input_tokens": 10,
                "output_tokens": 5, "inference_seconds": 0.2}


def _config(arm):
    config = controls.config_for(arm)
    return {**config, "max_actions": 3, "max_decisions": 3}


def _factory(*_):
    return Episode()


def test_configs_and_strict_outputs():
    for arm in controls.ARMS:
        assert controls.config_for(arm)["gpt_oss"]["model"]["model_revision"] == "6cee5e81ee83917806bbde320786a8fb61efebee"
    assert controls._strict_gpt_action("Action: get 1 log") == "get 1 log"
    assert controls._strict_gpt_action("get 1 log") is None
    assert controls._strict_gpt_action("Action: get 1 log\nAction: inventory") is None
    assert controls._valid_review('{"decision":"request","operation":"inspect_state_v1"}', "inventory")[1:] == ("fallback", "invalid_review_decision")


def test_standalone_actions_failures_and_full_generation_record():
    gpt = Model(["Action: craft 1 plank using 1 log", "Action: get 1 log",
                 "Action: craft 1 plank using 1 log"])
    result = controls.run_episode({"task_id": "dev"}, _config(controls.ARMS[0]), gpt,
                                  episode_factory=_factory)
    assert result["success"] is True
    assert result["failed_actions"] == 1
    assert result["environment_actions"] == 3
    assert result["model_calls"] == result["gptoss_model_calls"] == 3
    assert (result["input_tokens"], result["output_tokens"]) == (30, 15)
    assert result["inference_seconds"] == pytest.approx(0.6)
    assert result["external_support_operations"] == 0
    assert result["trace"][0]["gptoss_generation"]["raw_completion"]
    assert all("support" not in json.dumps(messages) for messages in gpt.messages)


def test_review_only_one_call_per_valid_proposal_and_fallback():
    qwen = Model(["malformed", "Action: craft 1 plank using 1 log",
                  "Action: get 1 log"])
    gpt = Model(['{"decision":"revise","action":"get 1 log"}',
                 '{"decision":"request","operation":"inspect_state_v1"}'])
    result = controls.run_episode({"task_id": "dev"}, _config(controls.ARMS[1]),
                                  gpt, qwen, _factory)
    assert result["parser_errors"] == 1
    assert result["qwen_model_calls"] == 3
    assert result["gptoss_model_calls"] == 2
    assert result["action_revisions"] == 1
    assert result["review_parse_errors"] == 1
    assert result["external_support_operations"] == 0
    assert result["trace"][0]["review_decision"] == "skipped_invalid_qwen_proposal"
    assert result["trace"][1]["action"] == "get 1 log"
    assert result["trace"][2]["action"] == "get 1 log"
    assert result["input_tokens"] == 50
    assert result["output_tokens"] == 25


def test_generation_failure_is_saved_without_environment_action():
    class Truncated(Model):
        def generate(self, messages, max_new_tokens=None):
            return {"text": "", "raw_completion": "unfinished analysis",
                    "generation_status": "truncated_before_final", "input_tokens": 12,
                    "output_tokens": 1024, "inference_seconds": 2.5}

    config = _config(controls.ARMS[0])
    config["max_decisions"] = 1
    result = controls.run_episode({"task_id": "dev"}, config, Truncated([]),
                                  episode_factory=_factory)
    assert result["gptoss_generation_failures"] == 1
    assert result["environment_actions"] == 0
    assert result["parser_errors"] == 1
    assert result["gptoss_output_tokens"] == 1024
    assert result["trace"][0]["gptoss_generation"]["raw_completion"] == "unfinished analysis"


def test_resume_does_not_repeat_completed_task(monkeypatch, tmp_path):
    arm = controls.ARMS[0]
    plan = {"git_commit": "fixed", "config_sha256": {arm: canonical_digest(controls.config_for(arm))},
            "python_version": "fixed", "package_versions": {}}
    frozen = {"tasks": [{"task_id": str(i), "initial_observation": f"observation {i}"}
                        for i in range(50)]}
    monkeypatch.setattr(controls, "checked_plan", lambda *_: plan)
    monkeypatch.setattr(controls, "manifests", lambda: ({}, frozen))
    monkeypatch.setattr(controls, "_runtime_lock", lambda: {
        "python_version": "fixed", "package_versions": {}})
    executed = []

    class Worker:
        def __init__(self, *_):
            pass

        def metadata(self):
            return {}

        def begin_episode(self, *_):
            pass

        def close(self):
            pass

    def episode(task, *_args):
        executed.append(task["task_id"])
        return {"task_id": task["task_id"], "initial_observation": task["initial_observation"],
                "success": False, "status": "max_decisions", "environment_actions": 0,
                "external_support_operations": 0, "trace": []}

    monkeypatch.setattr(controls, "run_episode", episode)
    first = controls.run_batch(tmp_path, arm, 0, 1, model_factory=Worker)
    second = controls.run_batch(tmp_path, arm, 0, 1, model_factory=Worker)
    assert first["completed"] == 1
    assert second["completed"] == 2
    assert executed == ["0", "1"]
    assert controls.status(tmp_path, arm, 0)["missing_indices"] == [2, 3, 4]
