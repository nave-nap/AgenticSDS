"""CPU-only behavioral tests for ADaPT's recursive TextCraft controller."""

import copy
import json
import re

import pytest

from agent_support.adapt import (ADaPTController, PlanParseError, QwenExecutor, QwenPlanner,
                                 _goal_item, _inventory_count, _next_line, _parse_adapt_action, parse_plan)
from agent_support import adapt_development
from agent_support.analysis import compare_trajectory, load_run
from agent_support.common import digest_text, validate_config
from agent_support.runner import run


BASE = json.loads(open("configs/textcraft_adapt_dynamic_replanning.json").read())
INITIAL = "Crafting commands:\ncraft 1 plank using 1 wood\n\nGoal: craft 1 plank."


def generation(text):
    return {"text": text, "input_tokens": 10, "output_tokens": 4, "inference_seconds": 0.1}


class ScriptedRole:
    def __init__(self, *outputs):
        self.outputs = iter(outputs)
        self.calls = []

    def plan(self, commands, task):
        self.calls.append({"task": task, "commands": commands})
        return generation(next(self.outputs))

    def next_action(self, commands, task, inventory_hint, history):
        self.calls.append({"task": task, "inventory_hint": inventory_hint,
                           "history": list(history)})
        return generation(next(self.outputs))


class CraftEpisode:
    def __init__(self, task=None, wood=0):
        self.task = task or {"task_id": "textcraft:seed:1", "initial_observation": INITIAL}
        self.initial_observation = self.task["initial_observation"]
        self.observation = self.initial_observation
        self.task_prompt = self.initial_observation
        self.inventory = {"wood": wood}
        self.actions = []
        self.closed = False

    def reset(self):
        self.observation = self.initial_observation

    def step(self, action):
        self.actions.append(action)
        success = False
        if action == "inventory":
            observation = "Inventory: " + " ".join(
                f"[{item}] ({count})" for item, count in self.inventory.items() if count)
        elif match := re.fullmatch(r"get (\d+) wood", action):
            self.inventory["wood"] = self.inventory.get("wood", 0) + int(match.group(1))
            observation = "Got wood"
        elif action == "craft 1 plank using 1 wood" and self.inventory.get("wood", 0):
            self.inventory["wood"] -= 1
            self.inventory["plank"] = self.inventory.get("plank", 0) + 1
            observation = "Crafted plank"
            success = True
        else:
            observation = "Could not craft"
        self.observation = observation
        return {"observation": observation, "reward": float(success), "done": success,
                "success": success, "failed_action": observation.startswith("Could not")}

    def close(self):
        self.closed = True


def config(**changes):
    value = copy.deepcopy(BASE)
    value.update(changes)
    return value


def run_controller(executor_outputs, planner_outputs=(), *, settings=None, wood=0):
    episode = CraftEpisode(wood=wood)
    planner = ScriptedRole(*planner_outputs)
    executor = ScriptedRole(*executor_outputs)
    result = ADaPTController(episode, settings or config(), planner, executor).run(episode.task)
    return result, episode, planner, executor


def test_plan_parser_nested_and_or_and_malformed():
    assert _next_line("> inventory:") == "inventory"
    assert _next_line("> get 1 wood.") == "get 1 wood"
    assert _parse_adapt_action("craft honey bottle using 1 honey block, 4 glass bottle") == (
        "craft honey bottle using 1 honey block, 4 glass bottle")
    assert _parse_adapt_action("craft honey bottle using glass bottle") is None
    plan = parse_plan("Step 1: fetch 1 wood\nStep 2: craft 1 plank using 1 wood\n"
                      "Step 3: fetch 1 stone\nExecution Order: (Step 1 AND (Step 2 OR Step 3))")
    assert plan.logic == "AND"
    assert plan.steps[1].logic == "OR"
    for raw in ("Step 1: fetch 1 wood", "Step 1: a\nExecution Order: Step 2",
                "Step 1: a\nStep 1: b\nExecution Order: Step 1",
                "Step 1: a\nStep 2: b\nExecution Order: Step 1 OR Step 1"):
        with pytest.raises(PlanParseError):
            parse_plan(raw)


def test_planner_and_executor_use_separate_prompts_and_token_caps():
    class CaptureModel:
        def __init__(self):
            self.calls = []
        def generate(self, messages, max_new_tokens=None):
            self.calls.append((messages, max_new_tokens))
            return generation("think: task failed!")
    model = CaptureModel()
    QwenPlanner(model, 800).plan("craft 1 plank using 1 wood", "craft 1 plank")
    QwenExecutor(model, 150).next_action("craft 1 plank using 1 wood", "craft 1 plank", "", [])
    assert [call[1] for call in model.calls] == [800, 150]
    assert "Execution Order" in model.calls[0][0][1]["content"]
    assert "think: Task Completed!" in model.calls[1][0][1]["content"]


def test_direct_success_never_calls_planner():
    result, episode, planner, executor = run_controller(
        ["craft 1 plank using 1 wood"], wood=1)
    assert result["success"] and result["direct_execution_succeeded"]
    assert result["decompositions"] == result["planner_calls"] == 0
    assert result["environment_actions"] == result["model_calls"] == 1
    assert episode.closed and episode.actions == ["craft 1 plank using 1 wood"]
    assert not planner.calls and len(executor.calls) == 1


def test_upstream_optional_craft_count_reaches_textcraft_episode():
    class OptionalCountEpisode(CraftEpisode):
        def step(self, action):
            assert action == "craft plank using 1 wood"
            self.actions.append(action)
            return {"observation": "Crafted 1 minecraft:plank", "reward": 1.0,
                    "done": True, "success": True, "failed_action": False}

    episode = OptionalCountEpisode(wood=1)
    planner = ScriptedRole()
    result = ADaPTController(episode, config(), planner,
                             ScriptedRole("craft plank using 1 wood")).run(episode.task)
    assert result["success"] and result["parser_errors"] == 0
    assert episode.actions == ["craft plank using 1 wood"]
    assert not planner.calls


def test_failed_direct_execution_decomposes_and_preserves_inventory():
    plan = "Step 1: fetch 1 wood\nStep 2: craft 1 plank using 1 wood\nExecution Order: (Step 1 AND Step 2)"
    result, episode, planner, executor = run_controller(
        ["think: task failed!", "get 1 wood", "think: Task Completed!",
         "craft 1 plank using 1 wood"], [plan])
    assert result["success"] and not result["direct_execution_succeeded"]
    assert result["decomposition_trigger"] == "executor_reported_failure"
    assert result["decompositions"] == result["planner_calls"] == 1
    assert episode.actions == ["get 1 wood", "inventory", "craft 1 plank using 1 wood"]
    assert "[wood] (1)" in executor.calls[-1]["inventory_hint"]
    assert result["model_calls"] == 5 and result["input_tokens"] == 50
    assert result["support_calls"] == 1 and result["retrieval_seconds"] == 0
    assert result["decomposition_tree"]["children"][0]["success"]


def test_nested_recursion_uses_same_episode_and_depth():
    outer = "Step 1: fetch 1 plank\nExecution Order: Step 1"
    inner = "Step 1: fetch 1 wood\nStep 2: craft 1 plank using 1 wood\nExecution Order: (Step 1 AND Step 2)"
    result, episode, planner, _ = run_controller(
        ["think: task failed!", "think: task failed!", "get 1 wood",
         "think: Task Completed!", "craft 1 plank using 1 wood"], [outer, inner])
    assert result["success"] and result["max_depth_reached"] == 3
    assert result["decompositions"] == 2 and len(planner.calls) == 2
    assert episode.actions[-1] == "craft 1 plank using 1 wood"


def test_or_tries_second_branch_and_and_stops_after_failed_branch():
    or_plan = "Step 1: fetch 1 stone\nStep 2: craft 1 plank using 1 wood\nExecution Order: (Step 1 OR Step 2)"
    result, episode, _, _ = run_controller(
        ["think: task failed!", "think: task failed!", "craft 1 plank using 1 wood"],
        [or_plan], settings=config(adapt_max_depth=2), wood=1)
    assert result["success"] and len(result["decomposition_tree"]["children"]) == 2
    assert episode.actions == ["craft 1 plank using 1 wood"]
    and_plan = or_plan.replace(" OR ", " AND ")
    result, episode, _, _ = run_controller(
        ["think: task failed!", "think: task failed!"], [and_plan],
        settings=config(adapt_max_depth=2), wood=1)
    assert not result["success"] and len(result["decomposition_tree"]["children"]) == 1
    assert not episode.actions


def test_depth_limit_and_malformed_plan_keep_failure_and_raw_output():
    plan = "Step 1: fetch 1 wood\nExecution Order: Step 1"
    result, _, planner, _ = run_controller(
        ["think: task failed!", "think: task failed!"], [plan],
        settings=config(adapt_max_depth=2))
    assert not result["success"] and len(planner.calls) == 1
    assert result["decomposition_tree"]["children"][0]["reason"] == "max_depth"
    raw = "Step 1: fetch 1 wood\nExecution Order: Step 9"
    result, _, _, _ = run_controller(["think: task failed!"], [raw])
    assert not result["success"] and result["plan_parse_errors"] == 1
    assert result["decomposition_tree"]["raw_plan"] == raw


def test_false_completion_claim_does_not_satisfy_subtask():
    plan = "Step 1: fetch 1 wood\nExecution Order: Step 1"
    result, episode, _, _ = run_controller(
        ["think: task failed!", "think: Task Completed!"], [plan],
        settings=config(adapt_max_depth=2))
    assert not result["success"] and result["false_completion_claims"] == 1
    assert episode.actions == ["inventory"]
    assert result["trace"][-1]["verified_completion"] is False
    assert _goal_item("craft 4 oak planks using 1 oak log") == (4, "oak planks")
    assert _inventory_count("Inventory: [oak_log] (2)", "oak logs") == 2


def test_global_action_and_model_budgets_stop_recursion():
    plan = "Step 1: fetch 1 wood\nExecution Order: Step 1"
    result, episode, planner, _ = run_controller(
        ["get 1 wood", "get 1 wood"], [plan],
        settings=config(max_actions=1, adapt_max_environment_actions=1))
    assert not result["success"] and result["status"] == "max_actions"
    assert episode.actions == ["get 1 wood"] and not planner.calls
    result, _, planner, _ = run_controller(
        ["think: task failed!"], [plan],
        settings=config(max_decisions=1, adapt_max_model_calls=1))
    assert result["status"] == "max_decisions" and not planner.calls


def test_config_rejects_memory_and_wrong_budget():
    validate_config(config(), "textcraft")
    with pytest.raises(ValueError):
        validate_config(config(trajectory_corpus="x"), "textcraft")
    with pytest.raises(ValueError):
        validate_config(config(max_actions=10), "textcraft")
    with pytest.raises(ValueError):
        validate_config(config(benchmark="alfworld"), "alfworld")


def test_runner_integration_and_paired_development_comparison(tmp_path):
    task = {"task_id": "textcraft:seed:1", "seed": 1, "initial_observation": INITIAL,
            "observation_sha256": digest_text(INITIAL)}
    manifest = {"schema_version": 1, "benchmark": "textcraft", "partition": "development", "tasks": [task]}
    class Model:
        def __init__(self, outputs):
            self.outputs = iter(outputs)
        def generate(self, messages, max_new_tokens=None):
            return generation(next(self.outputs))
    factory = lambda benchmark, task, config: CraftEpisode(task)
    adapt = run(config(), manifest, tmp_path, model=Model(["think: task failed!",
        "Step 1: fetch 1 wood\nStep 2: craft 1 plank using 1 wood\nExecution Order: (Step 1 AND Step 2)",
        "get 1 wood", "think: Task Completed!", "craft 1 plank using 1 wood"]), episode_factory=factory)
    baseline_config = json.loads(open("configs/textcraft_adapt_control_baseline.json").read())
    baseline = run(baseline_config, manifest, tmp_path, model=Model([
        "Action: get 1 wood", "Action: craft 1 plank using 1 wood"]), episode_factory=factory)
    meta, _, _, episodes = load_run(adapt)
    assert meta["status"] == "complete" and next(iter(episodes.values()))["success"]
    report = compare_trajectory(baseline, adapt)
    assert report["paired"]["same"] == 1
    assert report["right"]["model_calls"] == 5


def test_development_protocol_freezes_revision_manifest_and_configs(tmp_path, monkeypatch):
    source = json.loads(open("manifests/textcraft/development.json").read())
    folder = tmp_path / "manifests" / "textcraft"
    folder.mkdir(parents=True)
    (folder / "development.json").write_text(json.dumps(source))
    monkeypatch.setattr(adapt_development, "cached_model_revision", lambda _: "a" * 40)
    monkeypatch.setattr(adapt_development, "_clean_commit", lambda: "b" * 40)
    first = adapt_development.prepare(tmp_path)
    assert first["model_revision"] == "a" * 40
    assert adapt_development.preflight(tmp_path) == first
    assert adapt_development.prepare(tmp_path) == first
    changed = tmp_path / "adapt_development" / "configs" / "baseline.json"
    config_json = json.loads(changed.read_text())
    config_json["max_actions"] = 1
    changed.write_text(json.dumps(config_json))
    with pytest.raises(ValueError):
        adapt_development.preflight(tmp_path)


def test_development_protocol_rejects_manifest_change(tmp_path, monkeypatch):
    source = json.loads(open("manifests/textcraft/development.json").read())
    folder = tmp_path / "manifests" / "textcraft"
    folder.mkdir(parents=True)
    source["tasks"][0]["seed"] += 1
    (folder / "development.json").write_text(json.dumps(source))
    monkeypatch.setattr(adapt_development, "cached_model_revision", lambda _: "a" * 40)
    with pytest.raises(ValueError):
        adapt_development.prepare(tmp_path)
