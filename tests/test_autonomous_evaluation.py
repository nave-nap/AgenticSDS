"""CPU validation for frozen evaluation controls and comparison arithmetic."""

import json
from pathlib import Path

import pytest

from agent_support.autonomous_evaluation import STAGE_SIZES, locked_manifest, preflight, resolved_config
from agent_support.autonomous_evaluation_compare import SHARED_AGENT_KEYS, compare_cohort
from agent_support.autonomous_support import SupportContext, make_support_controller
from agent_support.common import canonical_digest, write_json_atomic
from agent_support.runner import run_episode
from agent_support.support_sequence_analysis import analyze_episodes, decision_sequences


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "manifests/textcraft/development.json").read_text())


def context(task_id="textcraft:seed:130", results=()):
    return SupportContext("textcraft", task_id, "Goal: craft beehive.", "Could not craft beehive",
                          (), (), 1, "get 1 spruce logs", "get 1 spruce logs", results=results)


def test_random_selector_is_seeded_and_stops_after_one_result():
    config = resolved_config("random_support")
    config["autonomous_support"]["random_selection"]["probability"] = 1.0
    first = make_support_controller(config)
    second = make_support_controller(config)
    a = first.decide(context())
    b = second.decide(context())
    assert a.decision.as_dict() == b.decision.as_dict()
    assert a.decision.kind == "request"
    assert a.decision.request.operation in config["autonomous_support"]["operations"]
    assert first.decide(context(results=(object(),))).decision.kind == "execute"
    config["autonomous_support"]["random_selection"]["probability"] = 0.0
    assert make_support_controller(config).decide(context()).decision.kind == "execute"


def test_fixed_reasoning_has_two_predeclared_calls_then_execute():
    controller = make_support_controller(resolved_config("fixed_reasoning_2"))
    first = controller.decide(context())
    second = controller.decide(context(results=(object(),)))
    third = controller.decide(context(results=(object(), object())))
    assert [item.decision.kind for item in (first, second, third)] == ["request", "request", "execute"]
    assert first.decision.request.operation == second.decision.request.operation == "reason_over_artifacts_v1"
    assert first.decision.request.arguments == second.decision.request.arguments


def test_fixed_and_random_controls_run_through_opt_in_episode_loop():
    class TaskModel:
        def generate(self, messages):
            return {"text": "Action: get 1 spruce logs", "input_tokens": 10,
                    "output_tokens": 5, "inference_seconds": 0.01}
    class ControllerModel:
        max_new_tokens = 1024
        def __init__(self):
            self.calls = 0
        def count_messages_tokens(self, messages):
            return 10
        def generate(self, messages, max_new_tokens=None):
            self.calls += 1
            return {"text": json.dumps({"output_type": "candidate_action",
                                         "content": "Action: get 1 oak logs"}),
                    "input_tokens": 10, "output_tokens": 20,
                    "inference_seconds": 0.01, "generation_status": "complete"}
    class Episode:
        def __init__(self, item):
            self.item = item
            self.actions = []
            self.executed = []
        def reset(self):
            self.initial_observation = self.observation = self.task_prompt = self.item["initial_observation"]
        def step(self, action):
            self.executed.append(action)
            return {"observation": "Got resource", "done": True,
                    "success": True, "failed_action": False}
        def close(self):
            pass
    item = MANIFEST["tasks"][0]
    fixed = resolved_config("fixed_reasoning_2")
    fixed["max_actions"] = fixed["max_decisions"] = 1
    episode = Episode(item)
    model = ControllerModel()
    result = run_episode("textcraft", item, fixed, TaskModel(),
                         episode_factory=lambda *_: episode, controller_model=model)
    assert result["support_calls"] == 2 and result["support_model_calls"] == 2
    assert episode.executed == ["get 1 oak logs"] and model.calls == 2
    random_config = resolved_config("random_support")
    random_config["max_actions"] = random_config["max_decisions"] = 1
    random_config["autonomous_support"]["operations"] = ["inspect_history_v1"]
    random_config["autonomous_support"]["random_selection"]["operations"] = ["inspect_history_v1"]
    random_config["autonomous_support"]["random_selection"]["probability"] = 1.0
    random_config["autonomous_support"].pop("memory")
    episode = Episode(item)
    result = run_episode("textcraft", item, random_config, TaskModel(),
                         episode_factory=lambda *_: episode)
    assert result["support_calls"] == 1 and result["controller_model_calls"] == 0
    assert episode.executed == ["get 1 spruce logs"]


def _episode(task_id, success, operations=()):
    events = [{"request": {"operation": name, "arguments": {}, "objective": "fixture"},
               "result": {"status": "ok", "information": {"observation": "fixture"},
                          "metadata": {}, "cost": {"model_calls": 1 if name == "reason_over_artifacts_v1" else 0,
                                                      "input_tokens": 10, "output_tokens": 5}},
               "controller_cost": {"model_calls": 1, "input_tokens": 4, "output_tokens": 3},
               "controller_elapsed_seconds": 0.1, "operation_elapsed_seconds": 0.2}
              for name in operations]
    trace = [{"decision": 1, "proposed_action": "get 1 spruce logs",
              "action": "get 1 oak logs" if operations else "get 1 spruce logs",
              "outcome": {"failed_action": not success},
              "support_trace": {"events": events, "termination_reason": "execute",
                                "controller_decisions": len(operations) + 1}}]
    return {"task_id": task_id, "success": success, "status": "success" if success else "max_actions",
            "environment_actions": 1, "failed_actions": int(not success),
            "model_decisions": 1, "model_calls": 1 + len(operations),
            "input_tokens": 20 + 10 * len(operations),
            "output_tokens": 7 + 5 * len(operations), "inference_seconds": 0.3,
            "parser_errors": 0, "support_calls": len(operations),
            "support_model_calls": sum(name == "reason_over_artifacts_v1" for name in operations),
            "actions_changed_by_support": int(bool(operations)), "trace": trace}


def test_sequence_analysis_counts_no_support_composition_and_outcome():
    episodes = {"a": _episode("a", True, ("inspect_history_v1", "reason_over_artifacts_v1")),
                "b": _episode("b", False)}
    report = analyze_episodes(episodes)
    assert report["no_support_decisions"] == 1
    assert report["multi_support_decisions"] == 1
    assert report["capability_counts"] == {"inspect_history_v1": 1, "reason_over_artifacts_v1": 1}
    assert report["by_task_outcome"]["success"]["composed_decisions"] == 1
    assert decision_sequences(episodes["a"])[0]["attempted_sequence"] == [
        "inspect_history_v1", "reason_over_artifacts_v1"]


def test_sequence_analysis_separates_explicit_no_support_from_fallback():
    episode = _episode("a", False)
    first = episode["trace"][0]["support_trace"]
    first["events"] = [{"controller_decision": {"kind": "execute", "action": None}}]
    assert analyze_episodes({"a": episode})["controller_chose_no_support_decisions"] == 1
    first["events"] = []
    first["termination_reason"] = "model_call_budget"
    report = analyze_episodes({"a": episode})
    assert report["no_support_decisions"] == 1
    assert report["controller_chose_no_support_decisions"] == 0


def test_legacy_m2_support_events_are_counted_without_inventing_wall_time():
    episode = _episode("a", True)
    record = episode["trace"][0]
    record.pop("support_trace")
    record["support_events"] = [{"mechanism": "m2_trajectory_memory", "retrieved": [],
                                  "retrieval_seconds": 0.2, "input_tokens": 30,
                                  "output_tokens": 10, "inference_seconds": 0.4}]
    row = decision_sequences(episode)[0]
    assert row["sequence"] == ["m2_trajectory_memory"]
    assert row["support_model_calls"] == 1
    assert row["support_wall_seconds"] is None


def _write_run(root, arm, tasks, success_flags):
    config = resolved_config(arm)
    snapshot = {**MANIFEST, "tasks": tasks}
    directory = root / arm
    (directory / "episodes").mkdir(parents=True)
    write_json_atomic(directory / "config.json", config)
    write_json_atomic(directory / "manifest.json", snapshot)
    recipe = (ROOT / "agent_support/textcraft_recipe_order.json").read_bytes()
    (directory / "textcraft_recipe_order.json").write_bytes(recipe)
    import hashlib
    metadata = {"status": "complete", "run_id": arm, "condition": config["condition"],
                "benchmark": "textcraft", "manifest_sha256": canonical_digest(MANIFEST),
                "selected_manifest_sha256": canonical_digest(snapshot),
                "selected_task_count": len(tasks), "limit": len(tasks),
                "config_sha256": canonical_digest(config),
                "textcraft_recipe_order_sha256": hashlib.sha256(recipe).hexdigest(),
                "git_commit": "fixture-commit", "python_version": "3.12",
                "package_versions": {"textcraft": "0.0.3"}}
    if arm == "qwen_m2_200":
        metadata.update(trajectory_corpus_sha256="corpus", trajectory_index_sha256="index")
    if arm in {"random_support", "autonomous"}:
        metadata.update(autonomous_memory_corpus_sha256="corpus",
                        autonomous_memory_index_sha256="index")
    write_json_atomic(directory / "run.json", metadata)
    for index, (task, success) in enumerate(zip(tasks, success_flags)):
        ops = ("inspect_history_v1", "reason_over_artifacts_v1") if arm == "autonomous" else ()
        write_json_atomic(directory / "episodes" / f"{index:04d}.json",
                          _episode(task["task_id"], success, ops))
    return directory


def test_comparator_pairs_tasks_and_rejects_incompatible_agent(tmp_path):
    tasks = MANIFEST["tasks"][:2]
    names = ("qwen_standalone", "qwen_reconsideration", "qwen_m2_200",
             "random_support", "autonomous", "fixed_reasoning_2")
    paths = {name: _write_run(tmp_path, name, tasks,
                              [True, False] if name != "autonomous" else [True, True])
             for name in names}
    report = compare_cohort(paths, require_full=False)
    assert report["aggregates"]["autonomous"]["successes"] == 2
    pair = next(item for item in report["paired_comparisons"]
                if item["left"] == "qwen_standalone" and item["right"] == "autonomous")
    assert pair["success_rate_difference"] == 0.5
    assert pair["helped_task_ids"] == [tasks[1]["task_id"]]
    assert pair["mcnemar_exact_two_sided_p"] == 1.0
    assert report["sequence_summaries"]["autonomous"]["multi_support_decisions"] == 2
    metadata_path = paths["autonomous"] / "run.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["autonomous_memory_index_sha256"] = "different"
    write_json_atomic(metadata_path, metadata)
    with pytest.raises(ValueError, match="M2-200 corpus or index differs"):
        compare_cohort(paths, require_full=False)
    metadata["autonomous_memory_index_sha256"] = "index"
    write_json_atomic(metadata_path, metadata)
    config_path = paths["autonomous"] / "config.json"
    config = json.loads(config_path.read_text())
    config["model_revision"] = "different"
    write_json_atomic(config_path, config)
    metadata = json.loads(metadata_path.read_text())
    metadata["config_sha256"] = canonical_digest(config)
    write_json_atomic(metadata_path, metadata)
    with pytest.raises(ValueError, match="Incompatible primary cohort"):
        compare_cohort(paths, require_full=False)


def test_cpu_preflight_uses_frozen_manifest_and_memory():
    assert preflight("qwen_standalone")["task_count"] == 20
    assert preflight("autonomous")["memory_checked"] is True


def test_frozen_m2_artifact_directory_override_is_explicit_and_shared(monkeypatch):
    root = Path.home() / "agent_support_data/m2_200"
    monkeypatch.setenv("AGENT_SUPPORT_M2_200_ARTIFACT_DIR", str(root))
    for arm in ("autonomous", "random_support"):
        memory = resolved_config(arm)["autonomous_support"]["memory"]
        assert memory["corpus_path"] == str(root / "corpus_200.json")
        assert memory["index_path"] == str(root / "index_200.json")
    m2 = resolved_config("qwen_m2_200")
    assert m2["trajectory_corpus"] == str(root / "corpus_200.json")
    assert m2["trajectory_index"] == str(root / "index_200.json")
    assert resolved_config("autonomous")["autonomous_support"]["memory"]["corpus_sha256"] == (
        "96a06b5e6436c20af36cd15e20241e9eb53ee37a9b9aff43a962266c9f884556")
    assert "trajectory_corpus" not in resolved_config("qwen_standalone")
    monkeypatch.setenv("AGENT_SUPPORT_M2_200_ARTIFACT_DIR", "relative/m2_200")
    with pytest.raises(ValueError, match="must be absolute"):
        resolved_config("autonomous")


def test_one_task_autonomous_and_standalone_smokes_are_operationally_matched():
    assert STAGE_SIZES["smoke"] == 1
    assert locked_manifest()["tasks"][0] == MANIFEST["tasks"][0]
    autonomous = resolved_config("autonomous")
    standalone = resolved_config("qwen_standalone")
    assert {key: autonomous.get(key) for key in SHARED_AGENT_KEYS} == {
        key: standalone.get(key) for key in SHARED_AGENT_KEYS}
    assert autonomous["autonomous_support"]["controller"] == "prompted_gpt_oss_v1"
    assert autonomous["autonomous_support"]["model"]["model"]["model_id"] == "openai/gpt-oss-20b"


def test_v2_reliability_smoke_is_versioned_and_keeps_frozen_task_and_limits():
    assert STAGE_SIZES["smoke"] == 1
    old = resolved_config("autonomous")
    new = resolved_config("autonomous_v2")
    assert new["autonomous_support"]["controller"] == "prompted_gpt_oss_v2"
    assert old["autonomous_support"]["controller"] == "prompted_gpt_oss_v1"
    assert {key: value for key, value in new.items() if key != "autonomous_support"} == {
        key: value for key, value in old.items() if key != "autonomous_support"}
    assert {key: value for key, value in new["autonomous_support"].items() if key != "controller"} == {
        key: value for key, value in old["autonomous_support"].items() if key != "controller"}
