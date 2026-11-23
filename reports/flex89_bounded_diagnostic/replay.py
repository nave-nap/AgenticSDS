"""Bounded, opt-in replay of archived GPT-OSS controller inputs on Locust.

This is a diagnostic driver, not an evaluation runner. It never advances a
TextCraft environment and stops on the first generation error.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import time
from pathlib import Path

from agent_support.common import canonical_digest, write_json_atomic
from agent_support.recovery_v4_eval import config_for


def validate_plan(plan: dict, config: dict) -> list[dict]:
    if (plan.get("schema_version") != 1 or plan.get("total_controller_requests") != 22 or
            plan.get("config_sha256") != canonical_digest(config) or
            plan.get("seed") != config["seed"]):
        raise ValueError("Archived replay plan differs from the frozen configuration")
    episodes = plan.get("episodes")
    if (not isinstance(episodes, list) or len(episodes) != 2 or
            [item.get("task_id") for item in episodes] !=
            ["textcraft:seed:151", "textcraft:seed:89"]):
        raise ValueError("Expected seed 151 followed by seed 89")
    requests = []
    for episode in episodes:
        for event in episode["events"]:
            if not event.get("qwen_messages"):
                raise ValueError("Missing archived Qwen input")
            for request in event["controller_requests"]:
                requests.append(request)
                if (not request.get("messages") or
                        type(request.get("expected_input_tokens")) is not int or
                        type(request.get("max_new_tokens")) is not int or
                        request["max_new_tokens"] > 1024 or
                        request["max_new_tokens"] < 1 or
                        canonical_digest(request["messages"]) != request["messages_sha256"]):
                    raise ValueError("Invalid archived controller input or output ceiling")
    if ([item["request_id"] for item in requests] != list(range(1, 23)) or
            requests[-1]["expected_input_tokens"] != 3724 or
            requests[-1]["max_new_tokens"] != 1024):
        raise ValueError("Archived request sequence or first failing request differs")
    return requests


def verify_worker(metadata: dict, expected: dict, backend: str) -> None:
    keys = ("backend_version", "torch_version", "torch_cuda_version", "gpu_name",
            "model_id", "model_revision", "quantization", "decoding",
            "chat_template_sha256", "reasoning_effort", "prompt_date",
            "quantization_method", "representation_type", "weight_layout_type",
            "scale_layout_type", "verified_expert_modules", "verified_packed_projections",
            "weight_storage_dtype", "scale_storage_dtype", "backing_storage_device")
    differences = [key for key in keys if metadata.get(key) != expected.get(key)]
    selected = metadata.get("attention_implementation", "eager")
    if differences or selected != backend:
        raise RuntimeError(f"Worker configuration differs: {differences}; attention={selected}")


def validate_token_manifest(manifest: dict, plan: dict, requests: list[dict]) -> dict[int, dict]:
    if (manifest.get("schema_version") != 1 or
            manifest.get("model_revision") != plan["worker_metadata"]["model_revision"] or
            manifest.get("chat_template_sha256") != plan["worker_metadata"]["chat_template_sha256"]):
        raise ValueError("Pinned tokenizer identity differs from the archived worker")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != 22:
        raise ValueError("Token manifest must contain all 22 requests")
    for request, row in zip(requests, rows, strict=True):
        digest = row.get("input_ids_sha256")
        if (row.get("request_id") != request["request_id"] or
                row.get("input_tokens") != request["expected_input_tokens"] or
                row.get("messages_sha256") != request["messages_sha256"] or
                not isinstance(digest, str) or len(digest) != 64 or
                any(char not in "0123456789abcdef" for char in digest)):
            raise ValueError(f"Invalid token manifest row for request {request['request_id']}")
    return {row["request_id"]: row for row in rows}


def run(plan: dict, config: dict, backend: str, output_dir: Path,
        *, token_manifest: dict, mode: str = "sequence",
        worker_factory=None, qwen_factory=None) -> dict:
    requests = validate_plan(plan, config)
    token_rows = validate_token_manifest(token_manifest, plan, requests)
    if mode not in {"sequence", "isolated22"}:
        raise ValueError("Unknown replay mode")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    required = ("CUDA_LAUNCH_BLOCKING", "AGENT_SUPPORT_GPTOSS_WORKER_TRACEBACKS",
                "AGENT_SUPPORT_GPU_MEMORY_DIAGNOSTICS", "AGENT_SUPPORT_GPTOSS_PHASE_TRACE")
    if (any(os.environ.get(name) != "1" for name in required) or
            os.environ.get("AGENT_SUPPORT_GPTOSS_CUDA_CACHE_RELEASE", "0") != "0"):
        raise ValueError("Require blocking launches, worker tracebacks, telemetry, and no cache release")
    os.environ["AGENT_SUPPORT_GPTOSS_FLEX_ATTENTION"] = "1" if backend == "flex_attention" else "0"
    from agent_support.controller_backend import GptOssControllerWorker
    from agent_support.gpt_oss_attention_probe import _require_single_cuda_qwen
    from agent_support.model import QwenModel
    worker_factory = worker_factory or GptOssControllerWorker
    qwen_factory = qwen_factory or QwenModel

    output_dir.mkdir(parents=True)
    report = {"mode": mode, "backend_requested": backend, "status": "running",
              "source_run_commit": plan["source_run_commit"],
              "config_sha256": plan["config_sha256"], "generations": [],
              "evaluator_package_versions": plan["evaluator_package_versions"],
              "successful_generations": 0, "first_failure": None}
    worker = None
    try:
        for package, expected in plan["evaluator_package_versions"].items():
            actual = importlib.metadata.version(package)
            if actual != expected:
                raise RuntimeError(f"Evaluator {package} {actual} differs from archived {expected}")
        worker = worker_factory(config["autonomous_support"]["model"],
                                output_dir / "workers")
        report["worker_metadata"] = worker.metadata()
        verify_worker(worker.metadata(), plan["worker_metadata"], backend)
        qwen = qwen_factory(config)
        qwen.enable_gpu_memory_diagnostics(output_dir / "diagnostics")
        report["qwen_device_map"] = _require_single_cuda_qwen(qwen)
        if report["qwen_device_map"] != plan["qwen_device_map"]:
            raise RuntimeError("Qwen placement differs from archived run")
        for episode in plan["episodes"]:
            if mode == "isolated22" and episode["task_id"] != "textcraft:seed:89":
                continue
            worker.begin_episode(episode["task_id"], plan["seed"])
            for event in episode["events"]:
                if mode == "isolated22" and event["decision"] != 6:
                    continue
                qwen.generate(event["qwen_messages"])
                for request in event["controller_requests"]:
                    if mode == "isolated22" and request["request_id"] != 22:
                        continue
                    if len(report["generations"]) >= (22 if mode == "sequence" else 1):
                        raise RuntimeError("Diagnostic generation cap reached")
                    row = {"request_id": request["request_id"],
                           "worker_request_id": request["request_id"] if mode == "sequence" else 1,
                           "task_id": episode["task_id"], "decision": request["decision"],
                           "event_index": request["event_index"],
                           "messages_sha256": request["messages_sha256"],
                           "expected_input_tokens": request["expected_input_tokens"],
                           "max_new_tokens": request["max_new_tokens"]}
                    report["generations"].append(row)
                    started = time.perf_counter()
                    stage = "input_probe"
                    try:
                        probe = worker.probe_input(request["messages"])
                        row["input_probe"] = probe
                        if (probe["attention_implementation"] != backend or
                                probe["input_tokens"] != request["expected_input_tokens"] or
                                probe.get("input_ids_sha256") !=
                                token_rows[request["request_id"]]["input_ids_sha256"]):
                            raise RuntimeError(
                                f"Backend, token count, or token hash differs at request {request['request_id']}")
                        stage = "generation"
                        result = worker.generate(request["messages"],
                                                 max_new_tokens=request["max_new_tokens"])
                    except Exception as exc:
                        failure_kind = getattr(exc, "failure_kind", None)
                        row.update(status="error", stage=stage,
                                   latency_seconds=time.perf_counter() - started,
                                   error_type=("OutOfMemoryError" if failure_kind == "cuda_oom"
                                               else getattr(exc, "error_type", type(exc).__name__)),
                                   worker_failure_kind=failure_kind,
                                   error_detail=getattr(exc, "detail", str(exc)),
                                   worker_traceback=getattr(exc, "worker_traceback", None),
                                   failed_phase=getattr(exc, "failed_phase", None),
                                   reported_worker_request_id=getattr(exc, "request_id", None))
                        report["first_failure"] = row
                        report["status"] = "stopped_on_error"
                        return report
                    complete = result["generation_status"] == "complete"
                    row.update(status="complete" if complete else "incomplete",
                               latency_seconds=time.perf_counter() - started,
                               input_tokens=result["input_tokens"],
                               output_tokens=result["output_tokens"],
                               generation_status=result["generation_status"])
                    if complete:
                        report["successful_generations"] += 1
                    else:
                        report["first_failure"] = row
                        report["status"] = "stopped_on_incomplete_generation"
                        return report
        report["status"] = "complete"
        return report
    except Exception as exc:
        report["status"] = "preflight_or_replay_error"
        report["first_failure"] = {"error_type": type(exc).__name__, "error_detail": str(exc)}
        return report
    finally:
        try:
            if worker is not None:
                worker.close()
        finally:
            write_json_atomic(output_dir / "replay_result.json", report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--token-manifest", type=Path)
    parser.add_argument("--backend", required=True, choices=("eager", "flex_attention"))
    parser.add_argument("--mode", choices=("sequence", "isolated22"), default="sequence")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    config = config_for("autonomous_recovery_v4")
    requests = validate_plan(plan, config)
    if args.check_only:
        print(json.dumps({"status": "ready", "requests": len(requests),
                          "first_failing_request": requests[-1]["request_id"]}))
        return
    if args.token_manifest is None:
        parser.error("--token-manifest is required for a GPU replay")
    manifest = json.loads(args.token_manifest.read_text())
    result = run(plan, config, args.backend, args.output_dir,
                 token_manifest=manifest, mode=args.mode)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
