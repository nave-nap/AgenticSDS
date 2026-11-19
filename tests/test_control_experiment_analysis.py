import pytest

from agent_support import control_experiment_analysis as analysis
from agent_support.heldout50_evaluation import FROZEN_SHA256, manifests


def _aggregate(arm, successes):
    _, frozen = manifests()
    rows = [{"task_id": task["task_id"], "observation_sha256": task["observation_sha256"],
             "success": index in successes, "environment_actions": 1,
             "failed_actions": 0, "model_calls": 1, "input_tokens": 10,
             "output_tokens": 2, "inference_seconds": 1.0}
            for index, task in enumerate(frozen["tasks"])]
    return {"arm": arm, "manifest_sha256": FROZEN_SHA256,
            "complete": True, "tasks_completed": 50, "tasks": rows}


def test_four_arm_paired_comparison():
    aggregates = {
        "qwen_standalone": _aggregate("qwen_standalone", {0, 1, 2}),
        "gptoss_standalone_v1": _aggregate("gptoss_standalone_v1", {0, 3, 4}),
        "gptoss_review_only_v1": _aggregate("gptoss_review_only_v1", {0, 1, 4, 5}),
        "autonomous_v3": _aggregate("autonomous_v3", {0, 1, 4, 5, 6}),
    }
    report = analysis.compare(aggregates, 2.0)
    assert report["arms"]["autonomous_v3"]["success_rate"] == 0.1
    assert report["comparisons"]["autonomous_v3_minus_gptoss_standalone_v1"]["left_only_successes"] == 3
    assert report["comparisons"]["gptoss_review_only_v1_minus_qwen_standalone"]["right_only_successes"] == 1
    assert len(report["comparisons"]) == 6
    aggregates["qwen_standalone"]["tasks"][0]["observation_sha256"] = "wrong"
    with pytest.raises(ValueError):
        analysis.compare(aggregates)
