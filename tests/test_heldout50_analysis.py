"""Paired held-out analysis uses only shared task identities and digests."""

import pytest

from agent_support.heldout50_analysis import compare, paired_interval
from agent_support.heldout50_evaluation import FROZEN_SHA256


def _aggregate(arm, entries):
    return {"arm": arm, "protocol": "textcraft_heldout50_v1",
            "plan_sha256": "shared-plan", "manifest_sha256": FROZEN_SHA256,
            "tasks": [{"task_id": task_id, "observation_sha256": digest,
                       "success": success, "environment_actions": 10,
                       "failed_actions": 2, "model_decisions": 10,
                       "model_calls": 10, "input_tokens": 100,
                       "output_tokens": 10, "inference_seconds": 5.0}
                      for task_id, digest, success in entries]}


def test_comparison_reports_pairing_uncertainty_and_cost():
    autonomous = _aggregate("autonomous_v3", [("a", "hash-a", True),
                                                ("b", "hash-b", False),
                                                ("c", "hash-c", True)])
    baseline = _aggregate("qwen_standalone", [("a", "hash-a", False),
                                                ("b", "hash-b", True)])
    result = compare(autonomous, baseline)
    assert result["paired_tasks"] == 2
    assert result["autonomous_only_successes"] == 1
    assert result["baseline_only_successes"] == 1
    assert result["paired_success_difference"] == 0
    assert result["missing_baseline_task_ids"] == ["c"]
    assert result["paired_task_bootstrap_95"] is not None
    assert result["autonomous"]["mean_model_calls"] == 10
    assert paired_interval([1]) is None


def test_m2_is_labeled_secondary_and_digest_mismatch_fails():
    autonomous = _aggregate("autonomous_v3", [("a", "hash-a", True)])
    baseline = _aggregate("qwen_legacy_m2", [("a", "hash-a", False)])
    assert compare(autonomous, baseline)["comparison_type"].startswith("secondary")
    baseline["tasks"][0]["observation_sha256"] = "other"
    with pytest.raises(ValueError, match="observation"):
        compare(autonomous, baseline)
    baseline["tasks"][0]["observation_sha256"] = "hash-a"
    baseline["plan_sha256"] = "different-plan"
    with pytest.raises(ValueError, match="protocol"):
        compare(autonomous, baseline)
