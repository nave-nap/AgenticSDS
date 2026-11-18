"""Describe complete per-decision support sequences without causal claims."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from .analysis import load_run
from .common import write_json_atomic


def decision_sequences(episode: dict) -> list[dict]:
    rows = []
    for record in episode.get("trace", []):
        support = record.get("support_trace", {})
        events = support.get("events", [])
        if not events and record.get("support_events"):
            events = [{"request": {"operation": item["mechanism"], "arguments": {},
                                   "objective": None},
                       "result": {"status": "ok" if not item.get("fallback_reason") else "empty",
                                  "information": {"retrieved": item.get("retrieved", []),
                                                  "action_changed": item.get("action_changed", False)},
                                  "metadata": {"legacy_event": True,
                                               "fallback_reason": item.get("fallback_reason"),
                                               "parse_error": item.get("parse_error", False)},
                                  "cost": {"model_calls": int("inference_seconds" in item),
                                           "input_tokens": item.get("input_tokens", 0),
                                           "output_tokens": item.get("output_tokens", 0)}},
                       "operation_elapsed_seconds": item.get("retrieval_seconds", 0)}
                      for item in record["support_events"]]
        steps = [{"request": event["request"], "status": event.get("result", {}).get("status"),
                  "blocked": event.get("blocked"), "error": event.get("error"),
                  "result_information": event.get("result", {}).get("information"),
                  "result_metadata": event.get("result", {}).get("metadata"),
                  "cost": event.get("result", {}).get("cost")}
                 for event in events if event.get("request")]
        sequence = [event["request"]["operation"] for event in events
                    if event.get("request") and event.get("result")]
        rows.append({
            "task_id": episode["task_id"], "task_success": bool(episode["success"]),
            "decision": record["decision"], "sequence": sequence,
            "attempted_sequence": [step["request"]["operation"] for step in steps],
            "support_steps": steps,
            "combination": sorted(set(sequence)), "length": len(sequence),
            "original_action": record.get("proposed_action"),
            "final_action": record.get("action"),
            "action_replaced": (record.get("action") is not None and
                                record.get("proposed_action") is not None and
                                record["action"] != record["proposed_action"]),
            "termination_reason": support.get("termination_reason"),
            "controller_chose_no_support": (not sequence and
                support.get("termination_reason") == "execute" and
                any(event.get("controller_decision", {}).get("kind") == "execute"
                    for event in events)),
            "controller_decisions": support.get("controller_decisions", 0),
            "controller_model_calls": sum(event.get("controller_cost", {}).get("model_calls", 0)
                                          for event in events),
            "controller_tokens": sum(event.get("controller_cost", {}).get("input_tokens", 0) +
                                      event.get("controller_cost", {}).get("output_tokens", 0)
                                      for event in events),
            "controller_wall_seconds": round(sum(event.get("controller_elapsed_seconds", 0)
                                                  for event in events), 3),
            "support_wall_seconds": (None if record.get("support_events") else
                                     round(sum(event.get("operation_elapsed_seconds", 0)
                                               for event in events), 3)),
            "support_model_calls": sum(event.get("result", {}).get("cost", {}).get("model_calls", 0)
                                       for event in events),
            "support_tokens": sum(event.get("result", {}).get("cost", {}).get("input_tokens", 0) +
                                  event.get("result", {}).get("cost", {}).get("output_tokens", 0)
                                  for event in events),
            "generation_failures": sum(
                bool(event.get("controller_metadata", {}).get("backend_error")) or
                event.get("controller_metadata", {}).get("generation_status", "complete") != "complete" or
                bool(event.get("result", {}).get("metadata", {}).get("backend_error")) or
                event.get("result", {}).get("metadata", {}).get("generation_status", "complete") != "complete"
                for event in events),
            "controller_parse_errors": sum(bool(event.get("controller_metadata", {}).get("parse_error"))
                                           for event in events),
            "support_parse_errors": sum(bool(event.get("result", {}).get("metadata", {}).get("parse_error")) or
                                        event.get("result", {}).get("status") == "invalid_action"
                                        for event in events),
            "budget_exhausted": support.get("termination_reason", "").endswith("budget"),
            "parser_error": record.get("error") == "parse_error",
            "environment_failed_action": bool(record.get("outcome", {}).get("failed_action", False)),
        })
    return rows


def analyze_episodes(episodes: dict[str, dict]) -> dict:
    decisions = [row for episode in episodes.values() for row in decision_sequences(episode)]
    sequence_counts = Counter(tuple(row["sequence"]) for row in decisions)
    outcome_sequences = {label: Counter(tuple(row["sequence"]) for row in decisions
                                        if row["task_success"] is success)
                         for label, success in (("success", True), ("failure", False))}
    combination_counts = Counter(tuple(row["combination"]) for row in decisions)
    capability_counts = Counter(operation for row in decisions for operation in row["sequence"])
    length_counts = Counter(row["length"] for row in decisions)
    by_outcome = defaultdict(lambda: {"tasks": 0, "decisions": 0,
                                      "support_operations": 0, "composed_decisions": 0,
                                      "action_replacements": 0})
    for episode in episodes.values():
        group = by_outcome["success" if episode["success"] else "failure"]
        group["tasks"] += 1
        rows = decision_sequences(episode)
        group["decisions"] += len(rows)
        group["support_operations"] += sum(row["length"] for row in rows)
        group["composed_decisions"] += sum(row["length"] > 1 for row in rows)
        group["action_replacements"] += sum(row["action_replaced"] for row in rows)
    by_support_count = defaultdict(lambda: {"tasks": 0, "successes": 0})
    for episode in episodes.values():
        count = sum(row["length"] for row in decision_sequences(episode))
        group = by_support_count[str(count)]
        group["tasks"] += 1
        group["successes"] += bool(episode["success"])
    return {
        "tasks": len(episodes), "decisions": len(decisions),
        "no_support_decisions": length_counts[0],
        "no_support_fraction": length_counts[0] / len(decisions) if decisions else None,
        "controller_chose_no_support_decisions": sum(row["controller_chose_no_support"]
                                                     for row in decisions),
        "multi_support_decisions": sum(row["length"] > 1 for row in decisions),
        "action_replacements": sum(row["action_replaced"] for row in decisions),
        "capability_counts": dict(sorted(capability_counts.items())),
        "sequence_lengths": {str(key): value for key, value in sorted(length_counts.items())},
        "sequences": [{"operations": list(key), "count": value}
                      for key, value in sorted(sequence_counts.items(), key=lambda pair: (-pair[1], pair[0]))],
        "sequences_by_task_outcome": {
            label: [{"operations": list(key), "count": value}
                    for key, value in sorted(counter.items(), key=lambda pair: (-pair[1], pair[0]))]
            for label, counter in outcome_sequences.items()},
        "combinations": [{"operations": list(key), "count": value}
                         for key, value in sorted(combination_counts.items(), key=lambda pair: (-pair[1], pair[0]))],
        "by_task_outcome": dict(by_outcome),
        "by_support_count": dict(sorted(by_support_count.items(), key=lambda item: int(item[0]))),
        "controller_model_calls": sum(row["controller_model_calls"] for row in decisions),
        "controller_tokens": sum(row["controller_tokens"] for row in decisions),
        "controller_wall_seconds": round(sum(row["controller_wall_seconds"] for row in decisions), 3),
        "support_wall_seconds": (None if any(row["support_wall_seconds"] is None for row in decisions)
                                 else round(sum(row["support_wall_seconds"] for row in decisions), 3)),
        "support_model_calls": sum(row["support_model_calls"] for row in decisions),
        "support_tokens": sum(row["support_tokens"] for row in decisions),
        "budget_exhausted_decisions": sum(row["budget_exhausted"] for row in decisions),
        "generation_failures": sum(row["generation_failures"] for row in decisions),
        "controller_parse_errors": sum(row["controller_parse_errors"] for row in decisions),
        "support_parse_errors": sum(row["support_parse_errors"] for row in decisions),
        "parser_errors": sum(row["parser_error"] for row in decisions),
        "environment_failed_actions": sum(row["environment_failed_action"] for row in decisions),
        "decision_rows": decisions,
        "interpretation": "Associations are descriptive; task difficulty and action history confound support use.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    metadata, _, _, episodes = load_run(args.run)
    report = {"run_id": metadata["run_id"], "condition": metadata["condition"],
              **analyze_episodes(episodes)}
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(f"Support-sequence report already exists: {destination}")
    write_json_atomic(destination, report)
    print(json.dumps({"output": str(destination), "tasks": report["tasks"],
                      "decisions": report["decisions"]}, indent=2))


if __name__ == "__main__":
    main()
