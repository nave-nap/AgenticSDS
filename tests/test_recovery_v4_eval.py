from __future__ import annotations

from agent_support import recovery_v4_eval as evaluation
from agent_support.common import canonical_digest


def test_repaired_preflight_checks_protocol_without_gpu():
    result = evaluation.preflight("autonomous_recovery_v4")
    assert result["recovery_integration_version"] == "autonomous_recovery_v4_protocol_v2"
    assert all(result["protocol_checks"].values())
    assert result["task_count"] == 30


def test_support_assessment_total_is_v4_only(monkeypatch, tmp_path):
    tasks = [{"task_id": "one", "observation_sha256": "digest", "seed": 51}]
    monkeypatch.setattr(evaluation, "manifest", lambda: {"tasks": tasks})
    monkeypatch.setattr(evaluation, "checked_plan", lambda *_: {"git_commit": "fixed"})
    monkeypatch.setattr(evaluation, "completed", lambda *_: {
        "one": {"task_id": "one", "success": False, "environment_actions": 1,
                "support_assessments": 2, "support_useful_judgments": 1, "trace": []}})
    v4 = evaluation.aggregate(tmp_path, "autonomous_recovery_v4")
    v3 = evaluation.aggregate(tmp_path, "expanded_v3_v1")
    assert v4["totals"]["support_assessments"] == 2
    assert "support_assessments" not in v3["totals"]


def test_resume_selects_only_missing_tasks(monkeypatch, tmp_path):
    arm = "autonomous_recovery_v4"
    tasks = evaluation.manifest()["tasks"]
    plan = {"git_commit": "fixed", "manifest_sha256": evaluation.MANIFEST_SHA256,
            "recipe_order_sha256": evaluation.RECIPE_SHA256,
            "config_sha256": {arm: canonical_digest(evaluation.config_for(arm))},
            "python_version": "fixed", "package_versions": {}}
    monkeypatch.setattr(evaluation, "checked_plan", lambda *_: plan)
    monkeypatch.setattr(evaluation, "package_versions", lambda: {})
    monkeypatch.setattr(evaluation.platform, "python_version", lambda: "fixed")

    class Worker:
        def __init__(self, *_):
            pass
        def metadata(self):
            return {"gpu_name": "NVIDIA L40S"}
        def begin_episode(self, *_):
            pass
        def close(self):
            pass

    executed = []
    def episode(task, config, qwen, gpt, **kwargs):
        executed.append(task["task_id"])
        kwargs["checkpoint"]({"task_id": task["task_id"], "status": "running"})
        return {"task_id": task["task_id"], "arm": arm,
                "initial_observation": task["initial_observation"],
                "success": False, "status": "max_decisions", "environment_actions": 0,
                "trace": []}

    original = evaluation._run_tasks
    def fake_run(*args, **kwargs):
        return original(*args, qwen_factory=lambda _config: object(),
                        gpt_factory=Worker, memory_factory=lambda *_: None,
                        episode_runner=episode, **kwargs)
    monkeypatch.setattr(evaluation, "_run_tasks", fake_run)
    first = evaluation.run_batch(tmp_path, arm, 0, 1)
    second = evaluation.run_batch(tmp_path, arm, 0, 1)
    assert first["completed"] == 1 and second["completed"] == 2
    assert executed == [tasks[0]["task_id"], tasks[1]["task_id"]]
    assert evaluation.status(tmp_path, arm, 0)["missing_indices"] == [2]
