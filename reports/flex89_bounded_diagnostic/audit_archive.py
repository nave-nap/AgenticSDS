"""Read-only audit of the first 22 archived GPT-OSS controller requests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent_support.common import canonical_digest


def audit(archive: Path, plan: dict) -> dict:
    run = next((archive / "runs" / "autonomous_recovery_v4" / "batch_03").glob("textcraft_*"))
    episodes = {int(path.name[:4]): json.loads(path.read_text())
                for path in (run / "episodes").glob("*.json")}
    if set(episodes) != {9, 10}:
        raise ValueError("Expected the two archived episodes")
    telemetry_path = next((run / "diagnostics").glob("gpt_oss_cuda_memory_*.jsonl"))
    telemetry = [json.loads(line) for line in telemetry_path.open()]
    starts = {item["request_id"]: item for item in telemetry
              if item["event"] == "generation_start"}
    ends = {item["request_id"]: item for item in telemetry
            if item["event"] == "generation_end"}
    rows = []
    for episode_index, planned_episode in zip((9, 10), plan["episodes"], strict=True):
        episode = episodes[episode_index]
        if episode["task_id"] != planned_episode["task_id"]:
            raise ValueError("Plan episode order differs from the archive")
        covered_trace = episode["trace"] if episode_index == 9 else episode["trace"][:6]
        for record, planned_event in zip(covered_trace, planned_episode["events"], strict=True):
            if record["decision"] != planned_event["decision"] or record["qwen_messages"] != planned_event["qwen_messages"]:
                raise ValueError("Plan decision or Qwen input differs from the archive")
            archived_events = record["support_trace"]["events"]
            archived_indices = [index for index, event in enumerate(archived_events)
                                if "controller_messages" in event]
            planned_indices = [request["event_index"] for request in planned_event["controller_requests"]]
            if archived_indices != planned_indices:
                raise ValueError("Plan omits or reorders an archived controller input")
            for request in planned_event["controller_requests"]:
                request_id = len(rows) + 1
                event_index = request["event_index"]
                event = archived_events[event_index]
                generation = event.get("controller_generation") or {}
                cost = event.get("controller_cost") or {}
                end = ends[request_id]
                count = (generation.get("input_tokens") or cost.get("input_tokens") or
                         end.get("input_tokens"))
                if (request["request_id"] != request_id or
                        request["decision"] != record["decision"] or
                        event["controller_messages"] != request["messages"] or
                        canonical_digest(request["messages"]) != request["messages_sha256"] or
                        request["expected_input_tokens"] != count or
                        end["input_tokens"] != count or
                        starts[request_id]["max_output_tokens"] != request["max_new_tokens"]):
                    raise ValueError(f"Archived request {request_id} differs from replay plan or telemetry")
                rows.append({"request_id": request_id, "task_id": episode["task_id"],
                             "decision": record["decision"], "event_index": event_index,
                             "input_tokens": count, "messages_sha256": request["messages_sha256"],
                             "historical_input_ids_sha256": None,
                             "max_new_tokens": request["max_new_tokens"],
                             "output_tokens": generation.get("output_tokens"),
                             "status": end["status"], "error_type": end.get("error_type")})
    if (len(rows) != 22 or [row["request_id"] for row in rows] != list(range(1, 23)) or
            any(row["status"] != "ok" for row in rows[:21]) or
            rows[21]["status"] != "error" or rows[21]["error_type"] != "AcceleratorError"):
        raise ValueError("Expected 21 successful requests and the first error at request 22")
    return {"schema_version": 1, "source_run_commit": plan["source_run_commit"],
            "historical_input_ids_available": False,
            "token_id_verification": "Requires the pinned worker tokenizer on Locust",
            "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.archive, json.loads(args.plan.read_text()))
    payload = json.dumps(report, indent=2) + "\n"
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.write_text(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
