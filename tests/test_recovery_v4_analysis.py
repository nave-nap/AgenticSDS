from agent_support import recovery_v4_analysis as analysis
from agent_support.recovery_v4 import ARMS
from agent_support.recovery_v4_eval import manifest


def test_paired_analysis_reports_success_and_compute(monkeypatch):
    tasks = manifest()["tasks"]
    episodes = {}
    reports = {}
    for arm in ARMS:
        selected = {0, 1} if arm == ARMS[0] else {0, 2, 3}
        episodes[arm] = {task["task_id"]: {
            "task_id": task["task_id"], "success": index in selected,
            "environment_actions": 31 if index in selected else 2,
            "inference_seconds": 2.0, "strategy_changes": int(arm == ARMS[1]),
            "replanning_events": 0, "trace": []}
            for index, task in enumerate(tasks)}
        reports[arm] = {"complete": True, "manifest_sha256": analysis.MANIFEST_SHA256,
                        "successes": len(selected),
                        "success_at_checkpoints": {str(x): 0 for x in (30, 60, 90, 120, 150)},
                        "totals": {key: (2 if arm == ARMS[0] else 3) for key in analysis.COSTS}}
    monkeypatch.setattr(analysis, "checked_plan", lambda _root: {})
    monkeypatch.setattr(analysis, "aggregate", lambda _root, arm: reports[arm])
    monkeypatch.setattr(analysis, "completed", lambda _root, arm, _plan: episodes[arm])
    result = analysis.analyze("unused")
    assert result["paired"]["v4_only_successes"] == 2
    assert result["paired"]["v3_only_successes"] == 1
    assert result["paired"]["success_rate_difference_v4_minus_v3"] == 1 / 30 or result["paired"]["success_rate_difference_v4_minus_v3"] == 0.0333
    assert result["paired"]["compute_differences_v4_minus_v3"]["model_calls"] == 1
