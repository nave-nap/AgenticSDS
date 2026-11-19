"""Exploratory paired accuracy, computation, and recovery-trajectory report."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from .common import write_json_atomic
from .heldout50_analysis import paired_interval, wilson
from .recovery_v4 import ARMS
from .recovery_v4_eval import MANIFEST_SHA256, aggregate, checked_plan, completed, manifest
from .support_sequence_analysis import analyze_episodes

COSTS = ("environment_actions", "failed_actions", "model_decisions", "model_calls",
         "qwen_model_calls", "gptoss_model_calls", "embedding_model_calls",
         "controller_model_calls",
         "support_model_calls", "input_tokens", "output_tokens", "inference_seconds",
         "elapsed_seconds", "gptoss_inference_seconds", "embedding_inference_seconds",
         "support_calls", "controller_errors", "strategy_changes",
         "replanning_events", "progress_assessments")


def analyze(root: str | Path) -> dict:
    plan = checked_plan(root)
    reports = {arm: aggregate(root, arm) for arm in ARMS}
    if any(not report["complete"] or report["manifest_sha256"] != MANIFEST_SHA256
           for report in reports.values()):
        raise ValueError("Both 30-task arms must be complete")
    episodes = {arm: completed(root, arm, plan) for arm in ARMS}
    tasks = manifest()["tasks"]
    paired = []
    for task in tasks:
        task_id = task["task_id"]
        a, b = episodes[ARMS[0]][task_id], episodes[ARMS[1]][task_id]
        paired.append({"task_id": task_id,
            "expanded_v3_success": bool(a["success"]), "recovery_v4_success": bool(b["success"]),
            "success_difference_v4_minus_v3": int(bool(b["success"])) - int(bool(a["success"])),
            "expanded_v3_actions": a["environment_actions"],
            "recovery_v4_actions": b["environment_actions"],
            "expanded_v3_inference_seconds": a["inference_seconds"],
            "recovery_v4_inference_seconds": b["inference_seconds"],
            "recovery_v4_strategy_changes": b["strategy_changes"],
            "recovery_v4_replanning_events": b["replanning_events"]})
    differences = [row["success_difference_v4_minus_v3"] for row in paired]
    trajectories = {}
    for arm in ARMS:
        items = episodes[arm]
        sequences = analyze_episodes(items)
        extended = [task_id for task_id, episode in items.items()
                    if episode["success"] and episode["environment_actions"] > 30]
        repeated_failure = []
        unproductive_tail = []
        useful_then_changed = 0
        for task_id, episode in items.items():
            executed = [record for record in episode["trace"] if record.get("outcome")]
            failures = [(record["action"], record["output_observation"])
                        for record in executed if record["outcome"].get("failed_action")]
            if len(failures) != len(set(failures)):
                repeated_failure.append(task_id)
            if not episode["success"] and len(executed) >= 30 and not any(
                    not record["outcome"].get("failed_action") for record in executed[-30:]):
                unproductive_tail.append(task_id)
            for record in episode["trace"]:
                events = record.get("support_trace", {}).get("events", [])
                if any(event.get("support_assessment", {}).get("useful")
                       for event in events if event.get("support_assessment")) and (
                        record.get("action") != record.get("proposed_action")):
                    useful_then_changed += 1
        trajectories[arm] = {
            "successful_after_action_30": extended,
            "tasks_with_repeated_identical_failures": repeated_failure,
            "failed_tasks_with_last_30_actions_all_failed": unproductive_tail,
            "useful_support_judgment_and_action_change_same_decision": useful_then_changed,
            "support_sequences": sequences}
    summaries = {}
    for arm in ARMS:
        report = reports[arm]
        successes = report["successes"]
        summaries[arm] = {"successes": successes, "success_rate": successes / 30,
                          "wilson_95": wilson(successes, 30),
                          "success_at_action_checkpoints": report["success_at_checkpoints"],
                          "totals": report["totals"]}
    return {"schema_version": 1, "protocol": "textcraft_autonomous_recovery30_v1",
            "manifest_sha256": MANIFEST_SHA256,
            "interpretation": "Exploratory, task-paired associations. V4 has a different controller protocol and additional visible-evidence supports; differences do not isolate a causal support-selection effect. Model weights do not update across tasks.",
            "arms": summaries,
            "paired": {"tasks": 30,
                       "success_rate_difference_v4_minus_v3": round(statistics.mean(differences), 4),
                       "paired_bootstrap_95": paired_interval(differences),
                       "v4_only_successes": differences.count(1),
                       "v3_only_successes": differences.count(-1),
                       "both_successes": sum(row["expanded_v3_success"] and row["recovery_v4_success"] for row in paired),
                       "compute_differences_v4_minus_v3": {
                           key: round(reports[ARMS[1]]["totals"][key] - reports[ARMS[0]]["totals"][key], 3)
                           for key in COSTS},
                       "task_rows": paired},
            "trajectories": trajectories}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = analyze(args.root)
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    write_json_atomic(args.output, report)
    print(json.dumps({"paired": {key: value for key, value in report["paired"].items()
                                 if key != "task_rows"},
                      "arms": report["arms"]}, indent=2))


if __name__ == "__main__":
    main()
