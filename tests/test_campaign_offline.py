from __future__ import annotations

import json

import pytest

from agent_support import campaign_offline as campaign
from agent_support.common import canonical_digest


def test_manifest_integrity_is_static_only():
    report = campaign.integrity()
    assert report["task_count"] == 30
    assert report["scientific_status"] == "static_integrity_only"
    assert report["manifest_sha256"] == campaign.MANIFEST_SHA256


def test_policy_source_is_frozen_but_requires_three_gpu_port():
    report = campaign.policy_source()
    assert tuple(report["policy_ids"]) == campaign.POLICIES
    assert report["frozen_config_manifest_tree_matches"]
    assert report["archived_stage_requests_one_gpu"]
    assert report["archived_stage_requests_flex_attention"]


def test_trajectory_flags_backend_contamination_and_support_usage(tmp_path):
    path = tmp_path / "runs" / "sample" / "episodes" / "0000.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"task_id": "textcraft:seed:51", "success": False,
        "status": "max_actions", "trace": [{"action": "get clay", "input_observation": "x",
        "outcome": {"failed_action": True}, "controller_fallback_reason": "controller_backend_error",
        "support_trace": {"events": [{"request": {"operation": "inspect_state_v1"},
                                       "result": {"status": "ok"}}]}}]}))
    report = campaign.trajectory(tmp_path)
    assert report["infrastructure_invalid_count"] == 1
    assert report["support_operations"] == {"inspect_state_v1": 1}


def test_exact_paired_test_and_report_gate(tmp_path):
    assert campaign.mcnemar_exact(0, 0) == 1
    assert campaign.mcnemar_exact(5, 0) == 0.0625
    tasks = campaign.manifest()["tasks"]
    paths = {}
    for policy in campaign.POLICIES:
        path = tmp_path / f"{policy}.json"
        rows = [{"task_id": task["task_id"], "seed": task["seed"],
                 "observation_sha256": task["observation_sha256"],
                 "success": policy != "original_v4" and index == 0,
                 "status": "success" if policy != "original_v4" and index == 0 else "max_actions",
                 "infrastructure_invalid": False, "environment_actions": 3}
                for index, task in enumerate(tasks)]
        path.write_text(json.dumps({"policy_id": policy, "primary_score_ready": True,
            "complete": True, "infrastructure_invalid_task_ids": [], "tasks_completed": 30,
            "successes": int(policy != "original_v4"),
            "manifest_sha256": campaign.MANIFEST_SHA256, "plan_sha256": "plan",
            "protocol": "matched", "tasks": rows}))
        paths[policy] = path
    result = campaign.compare_reports(paths)
    assert result["comparisons"]["state_aware_v1_minus_original_v4"]["wins"] == 1
    bad = json.loads(paths["original_v4"].read_text())
    bad["primary_score_ready"] = False
    paths["original_v4"].write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="Incomplete"):
        campaign.compare_reports(paths)


def test_allocation_cost_counts_preempted_gpu_time(tmp_path):
    path = tmp_path / "sacct.psv"
    path.write_text("171188|COMPLETED|0:0|300|cpu=8,mem=96G,gres/gpu=3|1G\n"
                    "171189|PREEMPTED|0:0|600|cpu=2,mem=16G|2G\n")
    report = campaign.allocation_cost(path, gpu_hour_rate=2, cpu_hour_rate=0.1)
    assert report["gpu_hours"] == 0.25
    assert report["estimated_dollars"] == 0.5 + round((8 * 300 + 2 * 600) / 3600 * 0.1, 4)
