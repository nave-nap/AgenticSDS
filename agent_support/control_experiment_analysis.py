"""Exploratory four-arm paired report for frozen TextCraft held-out50."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from .common import read_json, write_json_atomic
from .heldout50_analysis import paired_interval, wilson
from .heldout50_evaluation import FROZEN_SHA256, manifests

ARMS = ("qwen_standalone", "gptoss_standalone_v1", "gptoss_review_only_v1", "autonomous_v3")
PAIRS = (("autonomous_v3", "gptoss_standalone_v1"),
         ("autonomous_v3", "gptoss_review_only_v1"),
         ("gptoss_review_only_v1", "qwen_standalone"),
         ("autonomous_v3", "qwen_standalone"),
         ("gptoss_standalone_v1", "qwen_standalone"),
         ("gptoss_review_only_v1", "gptoss_standalone_v1"))
COST_KEYS = ("environment_actions", "failed_actions", "model_calls", "input_tokens",
             "output_tokens", "inference_seconds", "controller_model_calls",
             "controller_input_tokens", "controller_output_tokens",
             "controller_inference_seconds", "proposal_model_calls", "proposal_input_tokens",
             "proposal_output_tokens", "proposal_inference_seconds", "qwen_model_calls",
             "qwen_input_tokens", "qwen_output_tokens", "qwen_inference_seconds",
             "gptoss_model_calls", "gptoss_input_tokens", "gptoss_output_tokens",
             "gptoss_inference_seconds", "parser_errors", "controller_parse_errors",
             "controller_generation_failures", "gptoss_generation_failures",
             "gptoss_backend_errors", "review_parse_errors", "action_revisions",
             "qwen_generation_failures", "review_fallbacks", "actions_changed_by_support",
             "support_calls", "support_model_calls", "support_input_tokens",
             "support_output_tokens", "support_inference_seconds", "support_parse_errors",
             "invalid_support_requests", "external_support_operations")


def _rows(aggregate: dict, arm: str) -> dict:
    if (aggregate.get("arm") != arm or aggregate.get("manifest_sha256") != FROZEN_SHA256 or
            aggregate.get("complete") is not True or aggregate.get("tasks_completed") != 50):
        raise ValueError(f"Incomplete or incompatible {arm} aggregate")
    _, frozen = manifests()
    expected = {task["task_id"]: task["observation_sha256"] for task in frozen["tasks"]}
    rows = {row["task_id"]: row for row in aggregate["tasks"]}
    if len(rows) != 50 or {key: row.get("observation_sha256") for key, row in rows.items()} != expected:
        raise ValueError(f"Task identities or observations differ for {arm}")
    return rows


def compare(aggregates: dict[str, dict], gpu_hour_price: float | None = None) -> dict:
    if set(aggregates) != set(ARMS):
        raise ValueError("All four aggregates are required")
    if gpu_hour_price is not None and gpu_hour_price < 0:
        raise ValueError("GPU-hour price must be nonnegative")
    rows = {arm: _rows(aggregates[arm], arm) for arm in ARMS}
    summaries = {}
    for arm in ARMS:
        values = rows[arm].values()
        successes = sum(bool(row["success"]) for row in values)
        totals = {key: round(sum(row.get(key) or 0 for row in values), 3) for key in COST_KEYS}
        if arm == "qwen_standalone":
            totals["qwen_model_calls"] = totals["model_calls"] or sum(row.get("model_decisions") or 0 for row in values)
            totals["qwen_input_tokens"] = totals["input_tokens"]
            totals["qwen_output_tokens"] = totals["output_tokens"]
            totals["qwen_inference_seconds"] = totals["inference_seconds"]
        elif arm == "gptoss_review_only_v1":
            totals["controller_model_calls"] = totals["gptoss_model_calls"]
            totals["controller_input_tokens"] = totals["gptoss_input_tokens"]
            totals["controller_output_tokens"] = totals["gptoss_output_tokens"]
            totals["controller_inference_seconds"] = totals["gptoss_inference_seconds"]
            totals["controller_parse_errors"] = totals["review_parse_errors"]
            totals["controller_generation_failures"] = totals["gptoss_generation_failures"]
        elif arm == "autonomous_v3":
            totals["qwen_model_calls"] = totals["proposal_model_calls"]
            totals["qwen_input_tokens"] = totals["proposal_input_tokens"]
            totals["qwen_output_tokens"] = totals["proposal_output_tokens"]
            totals["qwen_inference_seconds"] = totals["proposal_inference_seconds"]
            totals["gptoss_model_calls"] = totals["controller_model_calls"] + totals["support_model_calls"]
            totals["gptoss_input_tokens"] = totals["controller_input_tokens"] + totals["support_input_tokens"]
            totals["gptoss_output_tokens"] = totals["controller_output_tokens"] + totals["support_output_tokens"]
            totals["gptoss_inference_seconds"] = round(totals["controller_inference_seconds"] + totals["support_inference_seconds"], 3)
            totals["action_revisions"] = totals["actions_changed_by_support"]
            totals["external_support_operations"] = totals["support_calls"]
        summaries[arm] = {"successes": successes, "task_count": 50,
                          "success_rate": successes / 50, "wilson_95": wilson(successes, 50),
                          "totals": totals,
                          "inference_gpu_hours_lower_bound": round(totals["inference_seconds"] / 3600, 4),
                          "inference_cost_lower_bound": (round(totals["inference_seconds"] / 3600 * gpu_hour_price, 2)
                                                         if gpu_hour_price is not None else None)}
    paired = {}
    for left, right in PAIRS:
        details = []
        for task_id in rows[left]:
            a, b = rows[left][task_id], rows[right][task_id]
            details.append({"task_id": task_id, "left_success": bool(a["success"]),
                            "right_success": bool(b["success"]),
                            "difference": int(bool(a["success"])) - int(bool(b["success"]))})
        differences = [item["difference"] for item in details]
        paired[f"{left}_minus_{right}"] = {
            "paired_tasks": 50, "success_rate_difference": round(statistics.mean(differences), 4),
            "paired_bootstrap_95": paired_interval(differences),
            "left_only_successes": differences.count(1),
            "right_only_successes": differences.count(-1),
            "both_succeeded": sum(item["left_success"] and item["right_success"] for item in details),
            "both_failed": sum(not item["left_success"] and not item["right_success"] for item in details),
            "compute_differences": {key: round(summaries[left]["totals"][key] -
                                               summaries[right]["totals"][key], 3)
                                    for key in ("environment_actions", "failed_actions", "model_calls",
                                                "input_tokens", "output_tokens", "inference_seconds")},
            "tasks": details}
    return {"schema_version": 1, "interpretation": "Exploratory follow-up on a previously observed held-out set; fresh held-out tasks are required for final validation. Differences are paired associations and do not identify a causal support-selection effect.",
            "manifest_sha256": FROZEN_SHA256, "gpu_hour_price": gpu_hour_price,
            "hardware_cost_note": "Inference GPU hours and priced values exclude model loading, environment time, queue time, and idle allocation; use Slurm accounting for actual billed GPU hours.",
            "arms": summaries, "comparisons": paired}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ARMS:
        parser.add_argument("--" + arm.replace("_", "-"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu-hour-price", type=float)
    args = parser.parse_args()
    aggregates = {arm: read_json(getattr(args, arm)) for arm in ARMS}
    report = compare(aggregates, args.gpu_hour_price)
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    write_json_atomic(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "comparisons"}, indent=2))


if __name__ == "__main__":
    main()
