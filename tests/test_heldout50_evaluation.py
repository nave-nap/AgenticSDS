"""CPU tests for the held-out50 lock, batch recovery, and historical reuse."""

import json
import sys

import pytest

from agent_support import heldout50_evaluation as evaluation
from agent_support.common import canonical_digest, write_json_atomic


def test_selection_and_frozen_memory_preflight():
    parent, frozen = evaluation.manifests()
    assert frozen["tasks"] == [task for task in parent["tasks"] if task["seed"] != 51][:50]
    assert len({task["task_id"] for task in frozen["tasks"]}) == 50
    assert all(task["seed"] != 51 for task in frozen["tasks"])
    result = evaluation.preflight("autonomous_v3")
    assert result["manifest_sha256"] == evaluation.FROZEN_SHA256
    assert result["memory_checked"] is True
    assert evaluation.preflight("qwen_standalone")["memory_checked"] is False


def test_batch_boundaries():
    assert evaluation.batch_indices(0, 10) == list(range(10))
    assert evaluation.batch_indices(4, 10) == list(range(40, 50))
    assert evaluation.batch_indices(9, 5) == list(range(45, 50))
    for batch, size in ((5, 10), (-1, 5), (0, 11), (0, 0)):
        with pytest.raises(ValueError):
            evaluation.batch_indices(batch, size)
    with pytest.raises(ValueError, match="max_tasks_per_attempt"):
        evaluation.run_batch("/unused", "qwen_standalone", 0, 0)


def test_runner_selects_noncontiguous_tasks_without_changing_parent_hash(tmp_path):
    from agent_support.runner import run

    class Model:
        def generate(self, messages):
            return {"text": "Action: get 1 oak logs", "input_tokens": 5,
                    "output_tokens": 4, "inference_seconds": 0.01}

    class Episode:
        def __init__(self, task):
            self.initial_observation = task["initial_observation"]
            self.observation = self.initial_observation
            self.task_prompt = self.initial_observation.split("Goal: ")[-1]

        def reset(self):
            return self.observation

        def step(self, action):
            return {"observation": "Done", "done": True, "success": True,
                    "failed_action": False}

        def close(self):
            pass

    parent, _ = evaluation.manifests()
    config = evaluation.resolved_config("qwen_standalone")
    for indices in ([0, 0], [50, 1], [-1], [], [100]):
        with pytest.raises(ValueError, match="task_indices"):
            run(config, parent, tmp_path, task_indices=indices)
    path = run(config, parent, tmp_path, task_indices=[10, 12], model=Model(),
               episode_factory=lambda benchmark, task, settings: Episode(task))
    metadata = json.loads((path / "run.json").read_text())
    selected = json.loads((path / "manifest.json").read_text())
    assert metadata["manifest_sha256"] == evaluation.PARENT_SHA256
    assert metadata["task_indices"] == [10, 12]
    assert selected["tasks"] == [parent["tasks"][10], parent["tasks"][12]]
    assert len(list((path / "episodes").glob("*.json"))) == 2


def _plan():
    return {"schema_version": 1, "protocol": "textcraft_heldout50_v1",
            "batch_size": 5, "manifest_sha256": evaluation.FROZEN_SHA256,
            "parent_manifest_sha256": evaluation.PARENT_SHA256,
            "recipe_order_sha256": evaluation.RECIPE_SHA256,
            "git_commit": "test-commit", "python_version": sys.version.split()[0],
            "package_versions": {"textcraft": "0.0.3"},
            "model_revisions": {
                "qwen_task_agent": evaluation.resolved_config("qwen_standalone")["model_revision"],
                "gpt_oss_controller": evaluation.resolved_config("autonomous_v3")
                ["autonomous_support"]["model"]["model"]["model_revision"]},
            "config_sha256": {arm: canonical_digest(evaluation.resolved_config(arm))
                              for arm in evaluation.ARMS},
            "historical_sources": {}}


def _write_run(path, indices, plan):
    parent, _ = evaluation.manifests()
    config = evaluation.resolved_config("qwen_standalone")
    selected = dict(parent, tasks=[parent["tasks"][i] for i in indices])
    path.mkdir(parents=True)
    (path / "episodes").mkdir()
    write_json_atomic(path / "run.json", {
        "condition": config["condition"], "manifest_sha256": evaluation.PARENT_SHA256,
        "config_sha256": plan["config_sha256"]["qwen_standalone"],
        "selected_manifest_sha256": canonical_digest(selected),
        "git_commit": "test-commit", "package_versions": plan["package_versions"],
        "python_version": plan["python_version"],
        "textcraft_recipe_order_sha256": evaluation.RECIPE_SHA256,
        "task_indices": indices, "status": "failed", "completed_task_count": len(indices),
    })
    write_json_atomic(path / "config.json", config)
    write_json_atomic(path / "manifest.json", selected)
    for i in indices:
        task = parent["tasks"][i]
        write_json_atomic(path / "episodes" / f"{i:04d}.json", {
            "task_id": task["task_id"], "benchmark": "textcraft",
            "initial_observation": task["initial_observation"],
            "success": True, "trace": [], "environment_actions": 1,
        })


def test_resume_skips_saved_episode_and_aggregates_partial(monkeypatch, tmp_path):
    plan = _plan()
    write_json_atomic(tmp_path / "plan.json", plan)
    monkeypatch.setattr(evaluation, "git_commit", lambda: "test-commit")
    monkeypatch.setattr(evaluation, "package_versions", lambda: plan["package_versions"])
    monkeypatch.setattr(evaluation, "require_clean_commit", lambda: "test-commit")
    monkeypatch.setattr(evaluation, "preflight", lambda arm: {
        "arm": arm, "package_versions": plan["package_versions"]})
    batch = tmp_path / "runs/qwen_standalone/batch_00"
    _write_run(batch / "textcraft_partial", [0], plan)
    captured = {}

    def fake_run(config, manifest, destination, task_indices):
        captured["indices"] = task_indices
        path = destination / "textcraft_retry"
        _write_run(path, task_indices, plan)
        return path

    monkeypatch.setattr(evaluation, "run", fake_run)
    result = evaluation.run_batch(tmp_path, "qwen_standalone", 0)
    assert captured["indices"] == [1, 2, 3, 4]
    assert result["missing_indices"] == []
    assert evaluation.run_batch(tmp_path, "qwen_standalone", 0)["reason"] == "already_complete"
    aggregate = evaluation.aggregate(tmp_path, "qwen_standalone")
    assert aggregate["tasks_completed"] == 5
    assert len(aggregate["missing_task_ids"]) == 45
    _write_run(batch / "textcraft_duplicate", [0], plan)
    with pytest.raises(ValueError, match="Duplicate"):
        evaluation.status(tmp_path, "qwen_standalone", 0)


def test_historical_qwen_coverage_skips_only_verified_tasks(monkeypatch, tmp_path):
    plan = _plan()
    plan["historical_sources"] = {"qwen_standalone": {"path": "/read-only/old-run",
                                                      "commit": "reviewed-commit"}}
    parent, _ = evaluation.manifests()
    task = parent["tasks"][0]
    episode = {"task_id": task["task_id"], "initial_observation": task["initial_observation"],
               "success": True, "trace": []}
    monkeypatch.setattr(evaluation, "historical_run", lambda *args, **kwargs: {
        "episodes": {task["task_id"]: episode}, "path": "/read-only/old-run",
        "commit": "reviewed-commit"})
    monkeypatch.setattr(evaluation, "checked_plan", lambda root: plan)
    result = evaluation.status(tmp_path, "qwen_standalone", 0)
    assert result["missing_indices"] == [1, 2, 3, 4]


def test_short_attempt_keeps_same_batch_mapping(monkeypatch, tmp_path):
    plan = _plan()
    write_json_atomic(tmp_path / "plan.json", plan)
    monkeypatch.setattr(evaluation, "git_commit", lambda: "test-commit")
    monkeypatch.setattr(evaluation, "package_versions", lambda: plan["package_versions"])
    monkeypatch.setattr(evaluation, "require_clean_commit", lambda: "test-commit")
    monkeypatch.setattr(evaluation, "preflight", lambda arm: {
        "package_versions": plan["package_versions"]})
    counter = [0]

    def fake_run(config, manifest, destination, task_indices):
        counter[0] += 1
        path = destination / f"textcraft_attempt_{counter[0]}"
        _write_run(path, task_indices, plan)
        return path

    monkeypatch.setattr(evaluation, "run", fake_run)
    assert evaluation.run_batch(tmp_path, "qwen_standalone", 0, 2)["missing_indices"] == [2, 3, 4]
    assert evaluation.run_batch(tmp_path, "qwen_standalone", 0, 2)["missing_indices"] == [4]
    assert evaluation.run_batch(tmp_path, "qwen_standalone", 0, 2)["missing_indices"] == []


def test_cli_preflight_resolves_new_manifest(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["heldout50_evaluation", "preflight",
                                       "--arm", "autonomous_v3"])
    evaluation.main()
    assert json.loads(capsys.readouterr().out)["manifest_sha256"] == evaluation.FROZEN_SHA256


def test_plan_freezes_batch_size_models_and_source_without_overwrite(monkeypatch, tmp_path):
    monkeypatch.setattr(evaluation, "require_clean_commit", lambda: "test-commit")
    monkeypatch.setattr(evaluation, "git_commit", lambda: "test-commit")
    monkeypatch.setattr(evaluation, "package_versions", lambda: {"textcraft": "0.0.3"})
    monkeypatch.setattr(evaluation, "preflight", lambda arm: {
        "config_sha256": canonical_digest(evaluation.resolved_config(arm))})
    monkeypatch.setattr(evaluation, "historical_run", lambda *args, **kwargs: {
        "kind": kwargs["kind"], "path": "/archived/m2", "commit": "old-commit",
        "config_sha256": "frozen", "coverage": 50, "episodes_sha256": "episode-digest",
        "episodes": {}})
    plan = evaluation.initialize(tmp_path, 5, legacy_m2="/archived/m2",
                                 reviewed_m2_commit="old-commit")
    assert plan["batch_size"] == 5
    assert plan["historical_sources"]["qwen_legacy_m2"]["coverage"] == 50
    assert "episodes" not in plan["historical_sources"]["qwen_legacy_m2"]
    assert plan["model_revisions"]["qwen_task_agent"]
    assert plan["model_revisions"]["gpt_oss_controller"]
    assert evaluation.checked_plan(tmp_path) == plan
    with pytest.raises(FileExistsError):
        evaluation.initialize(tmp_path, 5)
