"""Task-paired held-out50 comparison, including labeled legacy Qwen M2 context."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path

from .common import canonical_digest, read_json, write_json_atomic
from .heldout50_evaluation import (FROZEN_SHA256, checked_plan, historical_run,
                                   manifests)


def legacy_m2_aggregate(root: str | Path) -> dict:
    plan = checked_plan(root)
    source = plan["historical_sources"].get("qwen_legacy_m2")
    if source is None:
        raise ValueError("No historical Qwen legacy M2 run was locked")
    episodes = historical_run(source["path"], kind="qwen_legacy_m2",
                              reviewed_commit=source["commit"],
                              runtime=plan["package_versions"])["episodes"]
    _, frozen = manifests()
    rows = []
    for task in frozen["tasks"]:
        e = episodes.get(task["task_id"])
        if e is None:
            continue
        rows.append({"task_id": task["task_id"], "seed": task["seed"],
                     "observation_sha256": task["observation_sha256"],
                     **{key: e.get(key) for key in
                        ("success", "status", "environment_actions", "failed_actions",
                         "model_decisions", "model_calls", "input_tokens", "output_tokens",
                         "inference_seconds", "support_calls")}})
    return {"arm": "qwen_legacy_m2", "protocol": plan["protocol"],
            "plan_sha256": canonical_digest(plan),
            "manifest_sha256": FROZEN_SHA256, "source_run": source,
            "comparability": "secondary_partial_older_memory_corpus",
            "tasks_completed": len(rows), "complete": len(rows) == 50,
            "successes": sum(bool(row["success"]) for row in rows),
            "tasks": rows}


def wilson(successes: int, n: int) -> list[float] | None:
    if not n:
        return None
    z = 1.959963984540054
    p = successes / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0.0, center - half), 4), round(min(1.0, center + half), 4)]


def paired_interval(differences: list[int], draws: int = 10000) -> list[float] | None:
    n = len(differences)
    if n < 2:
        return None
    rng = random.Random(20261118)
    samples = sorted(sum(differences[rng.randrange(n)] for _ in range(n)) / n
                     for _ in range(draws))
    return [round(samples[int(draws * 0.025)], 4),
            round(samples[int(draws * 0.975) - 1], 4)]


def compare(autonomous: dict, baseline: dict) -> dict:
    if (autonomous.get("arm") != "autonomous_v3" or
            baseline.get("arm") not in {"qwen_standalone", "qwen_legacy_m2"} or
            autonomous.get("protocol") != "textcraft_heldout50_v1" or
            baseline.get("protocol") != "textcraft_heldout50_v1" or
            not isinstance(autonomous.get("plan_sha256"), str) or
            not autonomous["plan_sha256"] or
            autonomous.get("plan_sha256") != baseline.get("plan_sha256") or
            (baseline.get("arm") == "qwen_standalone" and
             autonomous.get("git_commit") != baseline.get("git_commit")) or
            autonomous.get("manifest_sha256") != FROZEN_SHA256 or
            baseline.get("manifest_sha256") != FROZEN_SHA256):
        raise ValueError("Comparison arm, manifest, or protocol differs")
    left = {row["task_id"]: row for row in autonomous["tasks"]}
    right = {row["task_id"]: row for row in baseline["tasks"]}
    if len(left) != len(autonomous["tasks"]) or len(right) != len(baseline["tasks"]):
        raise ValueError("Duplicate task IDs in aggregate")
    paired = []
    for task_id in sorted(left.keys() & right.keys()):
        a, b = left[task_id], right[task_id]
        if a["observation_sha256"] != b["observation_sha256"]:
            raise ValueError(f"Task observation differs: {task_id}")
        paired.append({"task_id": task_id, "autonomous_success": bool(a["success"]),
                       "baseline_success": bool(b["success"]),
                       "success_difference": int(bool(a["success"])) - int(bool(b["success"])),
                       **{f"{prefix}_{key}": row.get(key)
                          for prefix, row in (("autonomous", a), ("baseline", b))
                          for key in ("environment_actions", "failed_actions", "model_decisions",
                                      "model_calls", "input_tokens", "output_tokens",
                                      "inference_seconds", "controller_model_calls",
                                      "controller_inference_seconds", "support_calls",
                                      "actions_changed_by_support", "controller_parse_errors",
                                      "controller_generation_failures", "invalid_support_requests")}})
    def summary(prefix: str) -> dict:
        successes = sum(row[f"{prefix}_success"] for row in paired)
        result = {"successes": successes, "success_rate": round(successes / len(paired), 4)
                  if paired else None, "wilson_95": wilson(successes, len(paired))}
        for key in ("environment_actions", "failed_actions", "model_decisions", "model_calls",
                    "input_tokens", "output_tokens", "inference_seconds",
                    "controller_model_calls", "controller_inference_seconds", "support_calls",
                    "actions_changed_by_support", "controller_parse_errors",
                    "controller_generation_failures", "invalid_support_requests"):
            values = [row[f"{prefix}_{key}"] for row in paired if row[f"{prefix}_{key}"] is not None]
            result[f"mean_{key}"] = round(statistics.mean(values), 3) if values else None
            result[f"median_{key}"] = round(statistics.median(values), 3) if values else None
        return result
    differences = [row["success_difference"] for row in paired]
    cost_keys = ("environment_actions", "failed_actions", "model_decisions",
                 "model_calls", "input_tokens", "output_tokens", "inference_seconds")
    mean_deltas = {}
    for key in cost_keys:
        values = [row[f"autonomous_{key}"] - row[f"baseline_{key}"]
                  for row in paired if row[f"autonomous_{key}"] is not None
                  and row[f"baseline_{key}"] is not None]
        mean_deltas[key] = round(statistics.mean(values), 3) if values else None
    return {"autonomous_arm": autonomous["arm"], "baseline_arm": baseline["arm"],
            "comparison_type": ("primary" if baseline["arm"] == "qwen_standalone"
                                else "secondary_partial_older_memory_corpus"),
            "paired_tasks": len(paired), "autonomous_coverage": len(left),
            "baseline_coverage": len(right),
            "missing_baseline_task_ids": sorted(left.keys() - right.keys()),
            "missing_autonomous_task_ids": sorted(right.keys() - left.keys()),
            "autonomous": summary("autonomous"), "baseline": summary("baseline"),
            "mean_autonomous_minus_baseline": mean_deltas,
            "paired_success_difference": round(statistics.mean(differences), 4)
            if differences else None,
            "paired_task_bootstrap_95": paired_interval(differences),
            "autonomous_only_successes": sum(d == 1 for d in differences),
            "baseline_only_successes": sum(d == -1 for d in differences),
            "paired_tasks_detail": paired,
            "interpretation": "Task-paired association. GPT-OSS adds computation; support-call effects require controlled replay."}


def save_comparison(report: dict, output: str | Path) -> None:
    destination = Path(output)
    if destination.exists() or destination.with_suffix(".csv").exists():
        raise FileExistsError("Comparison output already exists")
    write_json_atomic(destination, report)
    rows = report["paired_tasks_detail"]
    with destination.with_suffix(".csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["task_id"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("legacy-m2")
    p.add_argument("--root", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("compare")
    p.add_argument("--autonomous", required=True)
    p.add_argument("--baseline", required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "legacy-m2":
        result = legacy_m2_aggregate(args.root)
        if Path(args.output).exists():
            raise FileExistsError(args.output)
        write_json_atomic(args.output, result)
    else:
        result = compare(read_json(args.autonomous), read_json(args.baseline))
        save_comparison(result, args.output)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"tasks", "paired_tasks_detail"}}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
