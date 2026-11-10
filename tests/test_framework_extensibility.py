"""A configured extension module adds a model and support without runner edits."""

from pathlib import Path

from agent_support.analysis import compare_modular, load_run
from agent_support.common import digest_text, read_json, write_json_atomic
from agent_support.runner import run
from agent_support.benchmarks import (BENCHMARK_DRIVERS, BENCHMARK_REGISTRY,
                                      MANIFEST_VALIDATORS, register_benchmark)
from agent_support.model import MODEL_REGISTRY, register_model


class Episode:
    actions = []
    def __init__(self, task):
        self.initial_observation = self.observation = self.task_prompt = task["initial_observation"]
    def reset(self):
        pass
    def step(self, action):
        return {"observation": "Done", "done": True, "success": action == "get 1 wood",
                "failed_action": False}
    def close(self):
        pass


def manifest(partition, number):
    observation = f"Crafting commands:\nget 1 wood\n\nGoal: craft {number} plank."
    return {"schema_version": 1, "benchmark": "textcraft", "partition": partition,
            "tasks": [{"task_id": f"textcraft:seed:{number}", "seed": number,
                       "initial_observation": observation, "observation_sha256": digest_text(observation)}]}


def test_configured_module_registers_mock_model_and_support(monkeypatch, tmp_path):
    module = tmp_path / "research_extension.py"
    module.write_text('''
from agent_support.model import register_model
from agent_support.supports import register_support, SupportResult
from agent_support.support_policy import register_policy, SelectionDecision

class Model:
    def __init__(self, config):
        self.config = config
    def metadata(self):
        return {"model_id": self.config["model_id"], "model_revision": self.config["model_revision"]}
    def count_messages_tokens(self, messages):
        return 3
    def count_tokens(self, text):
        return len(text.split())
    def generate(self, messages, max_new_tokens=None):
        answer = "Action: get 1 wood" if len(messages) > 2 else "Action: inventory"
        return {"text": answer, "input_tokens": 3, "output_tokens": 2, "inference_seconds": 0.01}

class Support:
    support_id = "example_revision"
    phase = "after_proposal"
    version = "example_revision_v1"
    requirements = ("valid_proposal",)
    def invoke(self, context, messages, action):
        prompt = messages + [{"role": "user", "content": "Try gathering wood."}]
        event = {"mechanism": self.support_id, "proposed_action": action,
                 "action_changed": False, "retrieved": [], "retrieval_seconds": 0.0}
        return SupportResult(messages, event, calls=1, revision_messages=prompt)

register_model("example_model", Model)
register_support("example_revision", lambda config, model, resources: Support())

class ExamplePolicy:
    policy_id = "example_policy_v1"
    def __init__(self, spec, task_id):
        self.support_id = spec["support_id"]
    def select(self, context):
        if context.phase == "after_proposal" and context.proposed_action:
            return SelectionDecision((self.support_id,), "example_trigger", "first_proposal")
        return SelectionDecision(reason="waiting_for_proposal")

register_policy("example_policy_v1", ExamplePolicy)
''', encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    dev, heldout = manifest("development", 1), manifest("heldout", 2)
    dev_path, heldout_path = tmp_path / "dev.json", tmp_path / "heldout.json"
    write_json_atomic(dev_path, dev)
    write_json_atomic(heldout_path, heldout)
    config = {"schema_version": 2, "benchmark": "textcraft", "condition": "modular_v1",
              "manifest": str(dev_path), "partition_manifests": {"development": str(dev_path),
                                                                  "heldout": str(heldout_path)},
              "component_modules": ["research_extension"], "model_type": "example_model",
              "model_id": "mock-model", "model_revision": "revision-1", "seed": 9,
              "max_actions": 1, "max_decisions": 1, "max_model_calls": 2,
              "history_window": 1, "max_new_tokens": 5, "max_support_calls": 1,
              "do_sample": False,
              "max_support_tokens": 10, "supports": ["example_revision"],
              "support_versions": {"example_revision": "example_revision_v1"},
              "selection_policy": {"id": "always_v1", "support_id": "example_revision"},
              "output_root": str(tmp_path / "runs")}
    directory = run(config, dev, tmp_path / "runs", episode_factory=lambda _, task, config: Episode(task))
    metadata, saved_config, saved_manifest, episodes = load_run(directory)
    result = episodes[dev["tasks"][0]["task_id"]]
    assert result["success"] and result["model_calls"] == 2 and result["support_calls"] == 1
    assert result["trace"][0]["support_policy"]["after_decision"]["reason"] == "always"
    assert result["trace"][0]["support_events"][0]["mechanism"] == "example_revision"
    assert metadata["component_modules"]["research_extension"]["sha256"]
    assert saved_config == config and saved_manifest == dev
    control_config = {**config, "selection_policy": {"id": "never_v1"}}
    control = run(control_config, dev, tmp_path / "runs",
                  episode_factory=lambda _, task, config: Episode(task))
    report = compare_modular(control, directory)
    assert report["paired"] == {"helped": 1, "hurt": 0}
    assert report["right"]["totals"]["support_calls"] == 1
    custom_config = {**config, "selection_policy": {"id": "example_policy_v1",
                                                    "support_id": "example_revision"}}
    custom = run(custom_config, dev, tmp_path / "runs",
                 episode_factory=lambda _, task, config: Episode(task))
    assert load_run(custom)[3][dev["tasks"][0]["task_id"]]["success"]


def test_registered_benchmark_runs_with_its_own_driver_and_manifest_validator(tmp_path):
    class Driver:
        def prompt(self, episode, history, window):
            return ["finish"], [{"role": "user", "content": episode.observation}]
        def parse(self, raw, available):
            return raw if raw in available else None
        def stop_reason(self, episode, history):
            return None
        def parse_error_ends_episode(self):
            return True

    class World:
        def __init__(self, task):
            self.initial_observation = self.observation = self.task_prompt = task["prompt"]
        def reset(self):
            pass
        def step(self, action):
            return {"observation": "Done", "done": True, "success": action == "finish",
                    "failed_action": False}
        def close(self):
            pass

    class Model:
        def __init__(self, config):
            self.config = config
        def metadata(self):
            return {"model_id": "world-model", "model_revision": "world-r1"}
        def generate(self, messages):
            return {"text": "finish", "input_tokens": 1, "output_tokens": 1,
                    "inference_seconds": 0.0}

    def validate_world_manifest(manifest):
        if any("task_fingerprint" not in task for task in manifest["tasks"]):
            raise ValueError("missing fingerprint")

    register_benchmark("mockworld", lambda task, config: World(task), Driver(), validate_world_manifest)
    register_model("world_model", Model)
    try:
        dev = {"schema_version": 2, "benchmark": "mockworld", "partition": "development",
               "tasks": [{"task_id": "world:1", "prompt": "start", "task_fingerprint": "hash-1"}]}
        heldout = {"schema_version": 2, "benchmark": "mockworld", "partition": "heldout",
                   "tasks": [{"task_id": "world:2", "prompt": "other", "task_fingerprint": "hash-2"}]}
        dev_path, heldout_path = tmp_path / "world_dev.json", tmp_path / "world_heldout.json"
        write_json_atomic(dev_path, dev)
        write_json_atomic(heldout_path, heldout)
        config = {"schema_version": 2, "benchmark": "mockworld", "condition": "modular_v1",
                  "manifest": str(dev_path), "partition_manifests": {"development": str(dev_path),
                                                                      "heldout": str(heldout_path)},
                  "model_type": "world_model", "model_id": "world-model", "model_revision": "world-r1",
                  "seed": 1, "max_actions": 1, "max_decisions": 1, "max_model_calls": 1,
                  "history_window": 1, "max_new_tokens": 1, "max_support_calls": 0,
                  "do_sample": False,
                  "max_support_tokens": 0, "supports": [], "support_versions": {},
                  "selection_policy": {"id": "never_v1"}, "output_root": str(tmp_path / "runs"),
                  "prompt_version": "world_prompt_v1"}
        directory = run(config, dev, tmp_path / "runs")
        metadata, _, _, episodes = load_run(directory)
        assert episodes["world:1"]["success"]
        assert metadata["prompt_versions"] == {"mockworld": "world_prompt_v1"}
    finally:
        BENCHMARK_REGISTRY.pop("mockworld")
        BENCHMARK_DRIVERS.pop("mockworld")
        MANIFEST_VALIDATORS.pop("mockworld")
        MODEL_REGISTRY.pop("world_model")
