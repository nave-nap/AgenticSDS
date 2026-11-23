"""CPU-only policy routing, trace attribution, and matched-stage checks."""
from __future__ import annotations

import json

import pytest

from agent_support import recovery_policy_eval as evaluation
from agent_support.common import read_json
from agent_support.recovery_accuracy_policy import AccuracyFirstPolicy, StateAwarePolicy
from agent_support.recovery_policy_registry import (POLICIES, make_policy,
                                                    policy_configuration, policy_digest)
from agent_support.recovery_policy_telemetry import decision_telemetry


def test_registry_routes_three_distinct_policies_and_preserves_frozen_config():
    assert POLICIES == ("original_v4", "state_aware_v1", "accuracy_first_v1")
    assert make_policy("original_v4") is None
    assert isinstance(make_policy("state_aware_v1"), StateAwarePolicy)
    assert isinstance(make_policy("accuracy_first_v1"), AccuracyFirstPolicy)
    assert policy_digest("state_aware_v1") != policy_digest("accuracy_first_v1")
    assert policy_configuration("original_v4")["selector"] == "unmodified_v4_controller"
    for policy_id in POLICIES:
        pre = evaluation.preflight(policy_id)
        assert pre["selector_matches_id"] is True
        assert pre["base_config_sha256"] == evaluation.v4_preflight(
            "autonomous_recovery_v4")["config_sha256"]
        assert pre["manifest_sha256"] == evaluation.MANIFEST_SHA256
        assert all(pre["v4_preflight"]["protocol_checks"].values())
    with pytest.raises(ValueError, match="Unknown recovery policy"):
        make_policy("autonomous_recovery_v4")


def test_telemetry_separates_policy_intervention_from_parser_fallback():
    record = {"decision": 2, "input_observation": "Could not craft",
        "proposed_action": "craft 1 plank using 1 log", "action": "get 1 log",
        "controller_fallback_reason": "controller_parse_error",
        "controller_fallback_detail": "bad JSON",
        "support_trace": {"events": [
            {"state": {"history": [{"action": "craft 1 plank using 1 log",
                "observation": "Could not craft"}]},
             "recovery_state_before": {"last_progress": {"failed_action": True}},
             "controller_messages": [{"role": "user", "content": json.dumps({
                 "policy_guidance": {"stalled": True}})}],
             "policy_intervention": {"policy_id": "accuracy_first_v1",
                                     "request": {"operation": "inspect_candidate_v1"}},
             "request": {"operation": "inspect_candidate_v1",
                         "arguments": {"action": "get 1 log"}},
             "result": {"status": "ok", "information": {"admissible": True},
                        "proposed_action": "get 1 log", "metadata": {}}},
            {"parse_error": "bad JSON"}]},
        "outcome": {"success": False, "failed_action": False},
        "progress_evidence": {"observed_feedback": "Got 1 log"}}
    telemetry = decision_telemetry(record, "accuracy_first_v1")
    assert telemetry["policy_id"] == "accuracy_first_v1"
    assert telemetry["failure_signals"]["last_progress"]["failed_action"] is True
    assert telemetry["failure_signals"]["policy_guidance"]["stalled"] is True
    assert telemetry["supports"][0]["selected_by_policy"] is True
    assert telemetry["candidate_evaluations"][0]["action"] == "get 1 log"
    assert {item["source"] for item in telemetry["candidate_actions"]} == {
        "qwen", "inspect_candidate_v1"}
    assert telemetry["controller_fallback_reason"] == "controller_parse_error"
    assert telemetry["infrastructure_failure_signal"] is False
    assert telemetry["final_selected_action"] == "get 1 log"


def test_fatal_worker_and_backend_fallback_are_infrastructure_signals():
    fatal = decision_telemetry({"decision": 1, "action": None,
        "termination_reason": "controller_cuda_oom",
        "support_trace": {"events": [{"error": "CUDA OOM"}]}}, "original_v4")
    assert fatal["fatal_infrastructure_failure"] is True
    assert fatal["infrastructure_failure_signal"] is True
    backend = decision_telemetry({"decision": 1, "action": "inventory",
        "controller_fallback_reason": "controller_backend_error",
        "support_trace": {"events": []}}, "original_v4")
    assert backend["fatal_infrastructure_failure"] is False
    assert backend["infrastructure_failure_signal"] is True


class FakeWorker:
    def __init__(self, *_):
        self.episodes = []

    def metadata(self):
        return {"gpu_name": "NVIDIA L40S", "model_id": "fake"}

    def begin_episode(self, task_id, seed):
        self.episodes.append((task_id, seed))

    def close(self):
        pass


def fake_episode(task, config, qwen, gpt, *, memory, checkpoint, policy):
    trace = [{"decision": 1, "input_observation": task["initial_observation"],
        "proposed_action": "inventory", "action": "inventory",
        "support_trace": {"events": [{"state": {"history": []},
                                      "controller_decision": {"kind": "execute",
                                                              "action": "inventory"}}]},
        "outcome": {"observation": "Inventory: empty", "done": False,
                    "success": False, "failed_action": False}}]
    checkpoint({"task_id": task["task_id"], "trace": trace, "status": "running"})
    return {"task_id": task["task_id"], "arm": config["arm"],
            "initial_observation": task["initial_observation"],
            "success": False, "status": "max_decisions", "environment_actions": 1,
            "failed_actions": 0, "model_decisions": 1, "support_calls": 0,
            "trace": trace}


def test_stage_and_batch_route_policies_and_lock_metadata(monkeypatch, tmp_path):
    monkeypatch.setattr(evaluation, "require_clean_commit", lambda: "test-commit")
    plan = evaluation._plan(require_runtime=False)
    calls = []

    def recording_episode(*args, **kwargs):
        policy = kwargs["policy"]
        calls.append("original_v4" if policy is None else policy.policy_id)
        return fake_episode(*args, **kwargs)

    def fake_run(root, policy_id, indices, selected_plan):
        return evaluation._run_tasks(root, policy_id, indices, selected_plan,
            qwen_factory=lambda _config: object(), gpt_factory=FakeWorker,
            memory_factory=lambda *_: None, episode_runner=recording_episode)

    for stage in evaluation.STAGES:
        for policy_id in POLICIES:
            fake_run(tmp_path / stage / policy_id, policy_id,
                     list(evaluation.STAGES[stage]), plan)
        report = evaluation.stage_report(tmp_path, stage)
        assert report["ready"] is True
        assert report["task_indices"] == list(evaluation.STAGES[stage])
    assert calls.count("original_v4") == 4
    assert calls.count("state_aware_v1") == 4
    assert calls.count("accuracy_first_v1") == 4
    monkeypatch.setattr(evaluation, "_plan", lambda require_runtime: plan)
    initialized = evaluation.initialize(tmp_path)
    assert initialized["implementation_commit"] == "test-commit"
    first = evaluation.run_batch(tmp_path, "accuracy_first_v1", 0, 1, runner=fake_run)
    second = evaluation.run_batch(tmp_path, "accuracy_first_v1", 0, 1, runner=fake_run)
    assert (first["completed"], second["completed"]) == (1, 2)
    assert evaluation.status(tmp_path, "accuracy_first_v1", 0)["missing_indices"] == [2]
    runs = sorted((tmp_path / "runs" / "accuracy_first_v1" / "batch_00").glob("textcraft_*"))
    first_run = next(path for path in runs if (path / "checkpoints" / "0000.json").exists())
    run_meta = read_json(first_run / "run.json")
    assert run_meta["implementation_commit"] == "test-commit"
    assert run_meta["policy_sha256"] == policy_digest("accuracy_first_v1")
    assert run_meta["model_settings"] == plan["model_settings"]
    assert read_json(first_run / "policy.json") == policy_configuration("accuracy_first_v1")
    snapshot = read_json(first_run / "checkpoints" / "0000.json")
    assert snapshot["decision_telemetry"][0]["policy_id"] == "accuracy_first_v1"
    with pytest.raises(ValueError, match="Three policy runs do not|complete"):
        evaluation.compare(tmp_path)


def test_paired_comparison_scores_success_without_cost_penalty(monkeypatch, tmp_path):
    tasks = evaluation.manifest()["tasks"]
    plan = {"implementation_commit": "fixed", "policy_sha256": {
        policy: policy_digest(policy) for policy in POLICIES}}
    monkeypatch.setattr(evaluation, "checked_plan", lambda *_: plan)
    def completed(_root, policy_id, _plan):
        successes = {"original_v4": 10, "state_aware_v1": 12,
                     "accuracy_first_v1": 14}[policy_id]
        return {task["task_id"]: {
            "task_id": task["task_id"], "success": index < successes,
            "status": "success" if index < successes else "max_actions",
            "failed_actions": 1, "environment_actions": 2,
            "model_decisions": 2, "support_calls": 100 if policy_id == "accuracy_first_v1" else 1,
            "input_tokens": 100000 if policy_id == "accuracy_first_v1" else 10,
            "output_tokens": 100000 if policy_id == "accuracy_first_v1" else 10,
            "inference_seconds": 1000 if policy_id == "accuracy_first_v1" else 1,
            "infrastructure_invalid": False, "decision_telemetry": []}
            for index, task in enumerate(tasks)}
    monkeypatch.setattr(evaluation, "completed", completed)
    report = evaluation.compare(tmp_path)
    assert report["exploratory_development_gate_met"] is True
    assert report["policy_reports"]["accuracy_first_v1"]["success_rate"] == 14 / 30
    assert report["paired"]["accuracy_first_v1_minus_original_v4"]["paired_wins"] == 4
    assert report["paired"]["accuracy_first_v1_minus_state_aware_v1"]["paired_wins"] == 2
    assert report["policy_reports"]["accuracy_first_v1"]["totals"]["input_tokens"] > 0


def test_fatal_cuda_episode_stops_batch_and_blocks_resume(monkeypatch, tmp_path):
    monkeypatch.setattr(evaluation, "require_clean_commit", lambda: "test-commit")
    plan = evaluation._plan(require_runtime=False)
    executed = []

    def fatal_episode(*args, **kwargs):
        result = fake_episode(*args, **kwargs)
        executed.append(result["task_id"])
        result["status"] = "controller_cuda_oom"
        result["trace"][-1]["termination_reason"] = "controller_cuda_oom"
        result["trace"][-1]["action"] = None
        return result

    directory = tmp_path / "runs" / "accuracy_first_v1" / "batch_00"
    run_dir = evaluation._run_tasks(directory, "accuracy_first_v1", [0, 1, 2], plan,
        qwen_factory=lambda _config: object(), gpt_factory=FakeWorker,
        memory_factory=lambda *_: None, episode_runner=fatal_episode)
    assert len(executed) == 1
    assert read_json(run_dir / "run.json")["status"] == "stopped_worker_failure"
    episode = read_json(next((run_dir / "episodes").glob("*.json")))
    assert episode["infrastructure_invalid"] is True
    monkeypatch.setattr(evaluation, "checked_plan", lambda *_: plan)
    with pytest.raises(ValueError, match="Prior stopped_worker_failure"):
        evaluation.run_batch(tmp_path, "accuracy_first_v1", 0)


def test_backend_fallback_stops_new_evaluator_without_changing_v4(monkeypatch, tmp_path):
    monkeypatch.setattr(evaluation, "require_clean_commit", lambda: "test-commit")
    plan = evaluation._plan(require_runtime=False)
    executed = []

    def backend_episode(*args, **kwargs):
        result = fake_episode(*args, **kwargs)
        executed.append(result["task_id"])
        result["trace"][-1]["controller_fallback_reason"] = "controller_backend_error"
        return result

    run_dir = evaluation._run_tasks(
        tmp_path / "runs" / "original_v4" / "batch_00", "original_v4", [0, 1, 2], plan,
        qwen_factory=lambda _config: object(), gpt_factory=FakeWorker,
        memory_factory=lambda *_: None, episode_runner=backend_episode)
    assert len(executed) == 1
    assert read_json(run_dir / "run.json")["status"] == "stopped_infrastructure_failure"
    episode = read_json(next((run_dir / "episodes").glob("*.json")))
    assert episode["infrastructure_invalid"] is True
    assert episode["decision_telemetry"][0]["controller_fallback_reason"] == "controller_backend_error"
