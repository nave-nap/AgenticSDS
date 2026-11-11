"""Experiment 1 accounts for selected, blocked, and executed support separately."""

import csv
import hashlib
import io
import random
from pathlib import Path

import pytest

from agent_support import experiment1_report as report
from agent_support.common import canonical_digest, read_json, write_json_atomic


def decision(reason, selected=False):
    return {"invoke": selected, "support_ids": [report.SUPPORT] if selected else [],
            "reason": reason}


def step(reason, *, selected=False, failed=False, blocked=None, event=False):
    return {"decision": 1, "action": "inventory", "outcome": {"failed_action": failed},
            "support_policy": {"before_decision": decision("ineligible_phase_or_proposal"),
                               "after_decision": decision(reason, selected),
                               "blocked": blocked or []},
            "support_events": ([{"mechanism": "reconsideration_only_v1"}] if event else [])}


def test_failure_smoke_reconciles_pre_call_blocks(monkeypatch, tmp_path):
    first = {"task_id": "task-1", "support_calls": 0,
             "trace": [step("failure_threshold_not_met", failed=True),
                       step("failure_threshold_met", selected=True,
                            blocked=[{"reason": "support_call_budget"}])]}
    second = {"task_id": "task-2", "support_calls": 1,
              "trace": [step("failure_threshold_not_met", failed=True),
                        step("failure_threshold_met", selected=True, event=True)]}
    third = {"task_id": "task-3", "support_calls": 0,
             "trace": [step("failure_threshold_not_met") ]}
    episodes = {item["task_id"]: {**item, "model_decisions": len(item["trace"]),
                                 "model_calls": 2 if item is not third else 1}
                for item in (first, second, third)}
    config = {"selection_policy": {"id": "failure_triggered_v1", "support_id": report.SUPPORT,
                                    "threshold": 1, "cooldown": 0}}
    manifest = {"partition": "development", "tasks": [{"task_id": key} for key in episodes]}
    monkeypatch.setattr(report, "load_run", lambda directory: (
        {"schema_version": 2, "limit": 3, "run_id": "smoke"}, config, manifest, episodes))
    write_json_atomic(tmp_path / "summary.json", {"support_calls": 1, "model_calls": 5})
    result = report.audit_failure_smoke(tmp_path, {"selections": 2, "actual_invocations": 1,
                                                     "not_met": 3, "model_calls": 5})
    assert result["blocked"] == {"support_call_budget": 1}
    assert result["totals"]["actual_invocations"] == 1
    episodes["task-1"]["trace"][1]["support_policy"]["blocked"] = []
    with pytest.raises(ValueError, match="accounting mismatch"):
        report.audit_failure_smoke(tmp_path, {})


def test_report_keeps_paired_success_and_compute_differences(monkeypatch, tmp_path):
    task_ids = [f"task-{index}" for index in range(20)]
    runs = {}
    for name in report.CONDITIONS:
        policy = ({"id": "never_v1"} if name == "never" else
                  {"id": "always_v1", "support_id": report.SUPPORT} if name == "always" else
                  {"id": "random_v1", "support_id": report.SUPPORT,
                   "probability": 0.25, "seed": 20261110} if name == "random" else
                  {"id": "failure_triggered_v1", "support_id": report.SUPPORT,
                   "threshold": 1, "cooldown": 0})
        episodes = {}
        for index, task_id in enumerate(task_ids):
            seed = f"20261110\0{task_id}\0random_v1".encode()
            random_selected = random.Random(int.from_bytes(hashlib.sha256(seed).digest(), "big")).random() < 0.25
            selected = name == "always" or (name == "random" and random_selected)
            reason = ("never" if name == "never" else "always" if name == "always" else
                      "random_draw_selected" if selected and name == "random" else
                      "random_draw_skipped" if name == "random" else "failure_threshold_not_met")
            trace_step = step(reason, selected=selected, event=selected)
            if name == "never":
                trace_step["support_policy"]["before_decision"]["reason"] = "never"
            episodes[task_id] = {"task_id": task_id,
                                 "success": index < (11 if name == "always" else 10),
                                 "support_calls": int(selected), "model_calls": 1 + int(selected),
                                 "input_tokens": 10, "output_tokens": 2 + int(selected),
                                 "failed_actions": 0, "actions_changed_by_support": 0,
                                 "inference_seconds": 0.1 + 0.1 * int(selected),
                                 "trace": [trace_step]}
        runs[name] = ({"git_commit": "same", "model_revision": "revision", "manifest_sha256": "hash",
                       "run_id": name, "config_sha256": name},
                      {"selection_policy": policy}, {}, episodes)
    monkeypatch.setattr(report, "checked_runs", lambda paths: (runs, task_ids))
    monkeypatch.setattr(report, "BOOTSTRAP_REPLICATES", 100)
    result, markdown, task_csv = report.build_report(
        {name: Path(tmp_path / name) for name in report.CONDITIONS})
    assert result["conditions"]["always"]["successes"] == 11
    assert result["conditions"]["always"]["actual_invocations"] == 20
    assert result["comparisons_vs_never"]["always"]["helped"] == 1
    assert result["comparisons_vs_never"]["always"]["success_difference_pp"] == 5
    assert result["comparisons_vs_never"]["always"]["metric_differences"]["model_calls"]["total"] == 20
    assert len(list(csv.DictReader(io.StringIO(task_csv)))) == 80
    assert "limited power" in markdown
    random_episode = runs["random"][3][task_ids[0]]
    random_episode["trace"][0]["support_policy"]["after_decision"]["invoke"] = not random_episode["trace"][0]["support_policy"]["after_decision"]["invoke"]
    with pytest.raises(ValueError, match="Policy decision disagrees"):
        report.trace_counts(random_episode, policy_spec=runs["random"][1]["selection_policy"])


def test_report_validates_complete_pinned_runs_and_provenance(monkeypatch, tmp_path):
    monkeypatch.setattr(report, "REPO", tmp_path)
    monkeypatch.setattr(report, "BOOTSTRAP_REPLICATES", 100)
    configs = tmp_path / "configs"
    manifest_dir = tmp_path / "manifests/textcraft"
    configs.mkdir()
    manifest_dir.mkdir(parents=True)
    manifest = {"schema_version": 1, "benchmark": "textcraft", "partition": "development",
                "tasks": [{"task_id": f"task-{index}", "seed": index,
                           "observation_sha256": f"hash-{index}"} for index in range(20)]}
    manifest_path = manifest_dir / "development.json"
    write_json_atomic(manifest_path, manifest)
    root = tmp_path / "experiment1_textcraft_development"
    root.mkdir()
    paths = {}
    for name in report.CONDITIONS:
        config = {"model_revision": "revision", "support_versions": {report.SUPPORT: "v1"},
                  "output_root": "separate", "max_actions": 30, "max_decisions": 30,
                  "max_model_calls": 40, "max_support_calls": 10, "max_support_tokens": 20000,
                  "selection_policy": ({"id": "never_v1"} if name == "never" else
                                       {"id": "always_v1", "support_id": report.SUPPORT} if name == "always" else
                                       {"id": "random_v1", "support_id": report.SUPPORT,
                                        "probability": 0.25, "seed": 20261110} if name == "random" else
                                       {"id": "failure_triggered_v1", "support_id": report.SUPPORT,
                                        "threshold": 1, "cooldown": 0})}
        config_path = configs / f"experiment1_textcraft_dev_{name}.json"
        write_json_atomic(config_path, config)
        directory = root / f"textcraft_{name}"
        (directory / "episodes").mkdir(parents=True)
        write_json_atomic(directory / "config.json", config)
        write_json_atomic(directory / "manifest.json", manifest)
        metadata = {"status": "complete", "schema_version": 2, "partition": "development",
                    "limit": None, "selected_task_count": 20,
                    "selected_manifest_sha256": canonical_digest(manifest),
                    "manifest_sha256": canonical_digest(manifest),
                    "config_sha256": canonical_digest(config), "git_dirty": False,
                    "python_hash_seed": "0", "effective_output_root": str(root),
                    "config_file_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                    "manifest_file_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                    "model_revision": "revision", "model_metadata": {"model_revision": "revision"},
                    "policy_parameters": config["selection_policy"],
                    "support_versions": config["support_versions"],
                    "budget_limits": {key: config[key] for key in (
                        "max_actions", "max_decisions", "max_model_calls", "max_support_calls",
                        "max_support_tokens")},
                    "git_commit": "same-commit", "run_id": name}
        write_json_atomic(directory / "run.json", metadata)
        checkpoint = {}
        for index, task in enumerate(manifest["tasks"]):
            task_id = task["task_id"]
            seed = f"20261110\0{task_id}\0random_v1".encode()
            chosen = (name == "always" or (name == "random" and
                      random.Random(int.from_bytes(hashlib.sha256(seed).digest(), "big")).random() < 0.25))
            reason = ("never" if name == "never" else "always" if name == "always" else
                      "random_draw_selected" if chosen and name == "random" else
                      "random_draw_skipped" if name == "random" else "failure_threshold_not_met")
            trace_step = step(reason, selected=chosen, event=chosen)
            if name == "never":
                trace_step["support_policy"]["before_decision"]["reason"] = "never"
            episode = {"task_id": task_id, "task_sha256": canonical_digest(task),
                       "config_sha256": canonical_digest(config), "success": index < 10,
                       "support_calls": int(chosen), "model_calls": 1 + int(chosen),
                       "input_tokens": 10, "output_tokens": 2,
                       "failed_actions": 0, "actions_changed_by_support": 0,
                       "inference_seconds": 0.1, "trace": [trace_step]}
            filename = f"{index:04d}.json"
            write_json_atomic(directory / "episodes" / filename, episode)
            checkpoint[filename] = canonical_digest(episode)
        write_json_atomic(directory / "checkpoint.json", {"episodes": checkpoint})
        paths[name] = directory
    result, _, task_csv = report.build_report(paths)
    assert result["task_count"] == 20
    assert len(list(csv.DictReader(io.StringIO(task_csv)))) == 80
    changed = read_json(paths["always"] / "run.json")
    changed["git_commit"] = "different-commit"
    write_json_atomic(paths["always"] / "run.json", changed)
    with pytest.raises(ValueError, match="provenance differs in git_commit"):
        report.build_report(paths)

