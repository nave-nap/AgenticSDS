"""Replay at most ten archived GPT-OSS requests in a fresh L40S worker.

Diagnostic only: no TextCraft environment, retries, or policy evaluation.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

from agent_support.common import canonical_digest, write_json_atomic
from agent_support.recovery_v4_eval import config_for
from reports.gptoss_seed51_cuda.audit import (EXPECTED_REQUESTS, SOURCE_COMMIT,
    SOURCE_EPISODE_SHA256, SOURCE_POLICY, SOURCE_RUN_SHA256, SOURCE_TASK,
    SOURCE_WORKER_LOG_SHA256)


def validate_plan(plan: dict) -> list[dict]:
    config = config_for("autonomous_recovery_v4")
    if (plan.get("schema_version") != 1 or plan.get("source_run_commit") != SOURCE_COMMIT or
            plan.get("source_run_sha256") != SOURCE_RUN_SHA256 or
            plan.get("source_episode_sha256") != SOURCE_EPISODE_SHA256 or
            plan.get("source_worker_log_sha256") != SOURCE_WORKER_LOG_SHA256 or
            plan.get("source_policy") != SOURCE_POLICY or
            plan.get("source_task") != SOURCE_TASK or
            plan.get("config_sha256") != canonical_digest(config) or
            plan.get("seed") != config["seed"] or
            plan.get("total_controller_requests") != EXPECTED_REQUESTS):
        raise ValueError("Replay plan differs from the audited frozen run")
    rows = []
    if [item.get("decision") for item in plan.get("decisions", [])] != [1, 2] or [
            len(item.get("controller_requests", [])) for item in plan["decisions"]] != [3, 7]:
        raise ValueError("Archived decision boundaries differ")
    for decision in plan.get("decisions", []):
        if not decision.get("qwen_messages"):
            raise ValueError("Archived Qwen input is missing")
        for row in decision.get("controller_requests", []):
            if (row.get("decision") != decision.get("decision") or
                    not isinstance(row.get("messages"), list) or
                    canonical_digest(row["messages"]) != row.get("messages_sha256") or
                    type(row.get("input_tokens")) is not int or row["input_tokens"] <= 0 or
                    type(row.get("max_new_tokens")) is not int or
                    not 1 <= row["max_new_tokens"] <= 1024):
                raise ValueError("Archived controller request is incomplete or changed")
            rows.append(row)
    if (len(rows) != EXPECTED_REQUESTS or
            [row["request_id"] for row in rows] != list(range(1, EXPECTED_REQUESTS + 1)) or
            any(row["status"] != "complete" for row in rows[:-1]) or
            rows[-1]["status"] != "error" or rows[-1]["input_tokens"] != 5451):
        raise ValueError("Archived request order or failure location differs")
    return rows


def verify_worker(actual: dict, archived: dict, backend: str) -> None:
    keys = ("model_id", "model_revision", "backend_version", "torch_version",
            "torch_cuda_version", "triton_version", "kernels_version",
            "quantization", "quantization_method", "decoding", "reasoning_effort",
            "prompt_date", "chat_template_sha256", "gpu_name",
            "gpu_compute_capability", "gpu_total_memory_bytes")
    differences = [key for key in keys if actual.get(key) != archived.get(key)]
    active = actual.get("attention_implementation")
    if (differences or actual.get("gpu_name") != "NVIDIA L40S" or
            (backend == "flex_attention" and active != "flex_attention") or
            (backend == "default" and active == "flex_attention") or not active):
        raise RuntimeError(f"Worker identity differs: {differences}; active attention={active}")


def _attach_memory(report: dict, directory: Path) -> None:
    paths = list(directory.glob("gpt_oss_cuda_memory_*.jsonl"))
    if len(paths) != 1:
        report["memory_telemetry_status"] = f"expected one GPT-OSS JSONL; found {len(paths)}"
        return
    events = [json.loads(line) for line in paths[0].read_text().splitlines()]
    starts = {item["request_id"]: item for item in events
              if item.get("event") == "generation_start"}
    ends = {item["request_id"]: item for item in events
            if item.get("event") == "generation_end"}
    for row in report["generations"]:
        request_id = row["request_id"]
        row["memory_before"] = starts.get(request_id, {}).get("memory")
        row["memory_after"] = ends.get(request_id, {}).get("memory")
        row["observed_input_tokens"] = ends.get(request_id, {}).get("input_tokens")
        row["observed_max_output_tokens"] = starts.get(request_id, {}).get("max_output_tokens")
    report["memory_telemetry_status"] = "recorded"


def run(plan: dict, backend: str, output_dir: Path, *, worker_factory=None,
        qwen_factory=None, version_lookup=importlib.metadata.version) -> dict:
    validate_plan(plan)
    if backend not in {"flex_attention", "default"}:
        raise ValueError("Unknown attention selection")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    required = ("CUDA_LAUNCH_BLOCKING", "AGENT_SUPPORT_GPTOSS_WORKER_TRACEBACKS",
                "AGENT_SUPPORT_GPU_MEMORY_DIAGNOSTICS")
    if (any(os.environ.get(name) != "1" for name in required) or
            os.environ.get("AGENT_SUPPORT_GPTOSS_CUDA_CACHE_RELEASE", "0") != "0"):
        raise ValueError("Require blocking launches, traceback, telemetry, and no cache release")
    os.environ["AGENT_SUPPORT_GPTOSS_FLEX_ATTENTION"] = (
        "1" if backend == "flex_attention" else "0")
    config = config_for("autonomous_recovery_v4")
    if worker_factory is None:
        from agent_support.controller_backend import GptOssControllerWorker
        worker_factory = GptOssControllerWorker
    if qwen_factory is None:
        from agent_support.model import QwenModel
        qwen_factory = QwenModel

    output_dir.mkdir(parents=True)
    report = {"schema_version": 1, "source_run_commit": SOURCE_COMMIT,
              "source_policy": SOURCE_POLICY, "source_task": SOURCE_TASK,
              "backend_requested": backend, "status": "running", "generations": [],
              "first_failure": None, "completed_generations": 0,
              "cuda_launch_blocking": True}
    worker = None
    try:
        for package, expected in plan["evaluator_package_versions"].items():
            if version_lookup(package) != expected:
                raise RuntimeError(f"Evaluator {package} version differs from archived run")
        worker = worker_factory(config["autonomous_support"]["model"],
                                output_dir / "workers")
        report["worker_metadata"] = worker.metadata()
        verify_worker(worker.metadata(), plan["worker_metadata"], backend)
        qwen = qwen_factory(config)
        enable_diagnostics = getattr(qwen, "enable_gpu_memory_diagnostics", None)
        if callable(enable_diagnostics):
            enable_diagnostics(output_dir / "diagnostics")
        report["qwen_device_map"] = {str(key): str(value) for key, value in
            getattr(getattr(qwen, "model", None), "hf_device_map", {}).items()}
        if (report["qwen_device_map"] and not set(report["qwen_device_map"].values())
                <= {"0", "cuda:0"}):
            raise RuntimeError("Qwen is not colocated on the single L40S")
        worker.begin_episode(SOURCE_TASK, plan["seed"])
        for decision in plan["decisions"]:
            qwen.generate(decision["qwen_messages"])
            for archived in decision["controller_requests"]:
                row = {key: archived[key] for key in
                       ("request_id", "decision", "event_index", "messages_sha256",
                        "input_tokens", "max_new_tokens")}
                row["archived_output_tokens"] = archived["output_tokens"]
                report["generations"].append(row)
                started = time.perf_counter()
                try:
                    counted = worker.count_messages_tokens(archived["messages"])
                    row["worker_counted_input_tokens"] = counted
                    if counted != archived["input_tokens"]:
                        raise RuntimeError(f"Input count differs at request {archived['request_id']}")
                    generated = worker.generate(archived["messages"],
                                                max_new_tokens=archived["max_new_tokens"])
                except Exception as exc:
                    row.update(status="error", elapsed_seconds=time.perf_counter() - started,
                               output_tokens=None, error_type=getattr(exc, "error_type", type(exc).__name__),
                               error_detail=getattr(exc, "detail", str(exc)),
                               worker_failure_kind=getattr(exc, "failure_kind", None),
                               worker_request_id=getattr(exc, "request_id", None),
                               failed_phase=getattr(exc, "failed_phase", None),
                               worker_traceback=getattr(exc, "worker_traceback", None))
                    report["first_failure"] = row.copy()
                    report["status"] = ("stopped_on_fatal_cuda" if
                        getattr(exc, "failure_kind", None) == "cuda_fatal" else
                        "stopped_on_error")
                    return report  # A fatal worker is never retried in-process.
                row.update(status="complete" if generated.get("generation_status") == "complete"
                           else "incomplete", elapsed_seconds=time.perf_counter() - started,
                           output_tokens=generated.get("output_tokens"),
                           generation_status=generated.get("generation_status"))
                if row["status"] != "complete":
                    report["first_failure"] = row.copy()
                    report["status"] = "stopped_on_incomplete_generation"
                    return report
                report["completed_generations"] += 1
        report["status"] = "complete"
        return report
    except Exception as exc:
        report["first_failure"] = {"error_type": type(exc).__name__, "error_detail": str(exc)}
        report["status"] = "preflight_or_replay_error"
        return report
    finally:
        if worker is not None:
            try:
                worker.close()
            except Exception as exc:
                report["worker_close_error"] = repr(exc)
        _attach_memory(report, output_dir / "diagnostics")
        failure = report.get("first_failure")
        if isinstance(failure, dict) and "request_id" in failure:
            match = next((row for row in report["generations"]
                          if row["request_id"] == failure["request_id"]), None)
            if match is not None:
                report["first_failure"] = match.copy()
        write_json_atomic(output_dir / "replay_result.json", report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--backend", choices=("flex_attention", "default"), required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    requests = validate_plan(plan)
    if args.check_only:
        print(json.dumps({"status": "ready", "requests": len(requests),
                          "last_request_input_tokens": requests[-1]["input_tokens"]}))
        return
    report = run(plan, args.backend, args.output_dir)
    print(json.dumps({"status": report["status"], "completed_generations":
                      report["completed_generations"], "first_failure":
                      report["first_failure"]}, default=str))
    if report["status"] != "complete":
        sys.exit(20)


if __name__ == "__main__":
    main()
