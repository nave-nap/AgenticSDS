"""CPU-only, fail-closed audit of Slurm 164919's GPT-OSS request sequence.

Writes a replay plan outside the benchmark root. No model or environment loads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from agent_support.common import canonical_digest, write_json_atomic
from agent_support.recovery_v4_eval import config_for

SOURCE_COMMIT = "701c25d4a76903743148210b3f3e780f246d7131"
SOURCE_POLICY = "state_aware_v1"
SOURCE_TASK = "textcraft:seed:51"
EXPECTED_REQUESTS = 10
SOURCE_RUN_SHA256 = "213a0f8c98f57662a1c4dc53a9e06214426e699e056c773179d65f811994825b"
SOURCE_EPISODE_SHA256 = "19f2c623fe496a7a421533544bed86c4732f021d0fe72f703633eda5be10065c"
SOURCE_WORKER_LOG_SHA256 = "da08f515674ae1da9c4463848ea5665c71b4ed81d72b8f22a15c47aad5154051"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(run_dir: Path) -> tuple[dict, dict]:
    run_path = run_dir / "run.json"
    meta = _read(run_path)
    config = _read(run_dir / "config.json")
    frozen = config_for("autonomous_recovery_v4")
    episodes = list((run_dir / "episodes").glob("*.json"))
    if (config != frozen or len(episodes) != 1 or
            _sha(run_path) != SOURCE_RUN_SHA256 or
            _sha(episodes[0]) != SOURCE_EPISODE_SHA256 or
            meta.get("implementation_commit") != SOURCE_COMMIT or
            meta.get("policy_id") != SOURCE_POLICY or
            meta.get("status") != "stopped_worker_failure" or
            meta.get("stop_reason") != "controller_cuda_fatal" or
            meta.get("attention_implementation") != "flex_attention" or
            meta.get("gptoss_metadata", {}).get("attention_implementation") != "flex_attention" or
            meta.get("gptoss_metadata", {}).get("gpu_name") != "NVIDIA L40S"):
        raise ValueError("Source run does not match the frozen seed-51 fatal diagnostic")
    episode = _read(episodes[0])
    if (episode.get("task_id") != SOURCE_TASK or
            episode.get("status") != "controller_cuda_fatal" or
            episode.get("environment_actions") != 1 or
            not episode.get("infrastructure_invalid")):
        raise ValueError("Source episode identity or fatal status differs")
    worker_log = run_dir / "workers" / "gpt_oss_controller_worker.log"
    worker_text = worker_log.read_text(encoding="utf-8")
    if (_sha(worker_log) != SOURCE_WORKER_LOG_SHA256 or
            f"GPT-OSS request {EXPECTED_REQUESTS}\n" not in worker_text or
            "torch.AcceleratorError: CUDA error: an illegal memory access was encountered"
            not in worker_text):
        raise ValueError("Source worker log lacks the expected fatal traceback")

    decisions = []
    rows = []
    for record in episode["trace"]:
        requests = []
        for event_index, event in enumerate(record["support_trace"]["events"]):
            # The shared GPT-OSS worker also serves reason_over_artifacts. This
            # particular run has none; reject any future archive that does.
            if event.get("request", {}).get("operation") == "reason_over_artifacts_v1":
                raise ValueError("GPT-OSS support request needs a separate message reconstruction")
            messages = event.get("controller_messages")
            if messages is None:
                continue
            generation = event.get("controller_generation") or {}
            cost = event.get("controller_cost") or {}
            count = generation.get("input_tokens", cost.get("input_tokens"))
            remaining = event.get("state", {}).get("remaining_budget", {})
            if (not isinstance(messages, list) or not messages or
                    type(count) is not int or count <= 0 or
                    type(remaining.get("additional_tokens")) is not int):
                raise ValueError("Controller request lacks exact messages or input count")
            limit = min(config["autonomous_support"]["model"]["model"]["max_new_tokens"],
                        remaining["additional_tokens"] - count)
            if limit <= 0:
                raise ValueError("Controller output ceiling is invalid")
            row = {"request_id": len(rows) + 1, "decision": record["decision"],
                   "event_index": event_index, "messages": messages,
                   "messages_sha256": canonical_digest(messages),
                   "input_tokens": count, "output_tokens": generation.get("output_tokens"),
                   "max_new_tokens": limit,
                   "peak_allocated_bytes": generation.get("peak_gpu_memory_allocated_bytes"),
                   "peak_reserved_bytes": generation.get("peak_gpu_memory_reserved_bytes"),
                   "status": "error" if event.get("error") else "complete"}
            rows.append(row)
            requests.append(row)
        decisions.append({"decision": record["decision"],
                          "qwen_messages": record["qwen_messages"],
                          "controller_requests": requests})
    if (len(rows) != EXPECTED_REQUESTS or
            any(item["status"] != "complete" for item in rows[:-1]) or
            rows[-1]["status"] != "error" or
            rows[-1]["input_tokens"] != 5451 or
            episode.get("controller_model_calls") != EXPECTED_REQUESTS):
        raise ValueError("Controller sequence does not end at the expected request 10")
    last_event = episode["trace"][-1]["support_trace"]["events"][-1]
    if ("ControllerWorkerFatalCUDA" not in last_event.get("error", "") or
            "GPT-OSS request 10\n" not in last_event.get("worker_traceback", "") or
            last_event["worker_traceback"] not in worker_text):
        raise ValueError("The archived trace does not attribute request 10 to fatal CUDA")
    diagnostics = list((run_dir / "diagnostics").glob("gpt_oss_cuda_memory_*.jsonl"))
    plan = {"schema_version": 1, "source_run_commit": SOURCE_COMMIT,
            "source_run_sha256": _sha(run_path), "source_episode_sha256": _sha(episodes[0]),
            "source_worker_log_sha256": _sha(worker_log),
            "source_policy": SOURCE_POLICY, "source_task": SOURCE_TASK,
            "config_sha256": canonical_digest(config), "seed": config["seed"],
            "worker_metadata": meta["gptoss_metadata"],
            "evaluator_package_versions": meta["package_versions"],
            "decisions": decisions, "total_controller_requests": len(rows)}
    summary = {"schema_version": 1, "source_run_sha256": plan["source_run_sha256"],
               "source_episode_sha256": plan["source_episode_sha256"],
               "source_worker_log_sha256": plan["source_worker_log_sha256"],
               "source_commit": SOURCE_COMMIT, "policy": SOURCE_POLICY,
               "task": SOURCE_TASK, "active_attention": "flex_attention",
               "gpu": meta["gptoss_metadata"]["gpu_name"],
               "controller_requests_reconstructed": len(rows),
               "request_message_hashes": [row["messages_sha256"] for row in rows],
               "failing_request": EXPECTED_REQUESTS,
               "failing_input_tokens": rows[-1]["input_tokens"],
               "failing_output_tokens": None,
               "failing_max_output_tokens": rows[-1]["max_new_tokens"],
               "failing_memory_snapshot_available": bool(diagnostics),
               "last_completed_peak_allocated_bytes": rows[-2]["peak_allocated_bytes"],
               "last_completed_peak_reserved_bytes": rows[-2]["peak_reserved_bytes"],
               "worker_traceback": last_event["worker_traceback"]}
    return plan, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    plan, summary = audit(args.run_dir)
    args.output_dir.mkdir(parents=True)
    write_json_atomic(args.output_dir / "replay_plan.json", plan)
    write_json_atomic(args.output_dir / "audit.json", summary)
    print(json.dumps({key: value for key, value in summary.items()
                      if key != "worker_traceback"}, indent=2))


if __name__ == "__main__":
    main()
