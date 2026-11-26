"""CPU preflight and one bounded three-GPU GPT-OSS feasibility request."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from .common import canonical_digest, write_json_atomic
from .autonomous_evaluation import require_clean_commit
from .gpt_oss_two_gpu import SHARD_ENV, VISIBLE_ENV
from .gpt_oss_chunked_eager import ENABLE_ENV as CHUNKED_ENV, NAME as CHUNKED_NAME
from .recovery_v4_eval import config_for, manifest, memory_checks


PROTOCOL = "three_gpu_chunked_eager_v2"
TARGET_TOKENS = 8887
MIN_FREE_BYTES = 10 * 1024**3


def preflight() -> dict:
    selected = manifest()
    arms = ("expanded_v3_v1", "autonomous_recovery_v4")
    configs = {arm: config_for(arm) for arm in arms}
    checks = {arm: memory_checks(configs[arm], require_files=True) for arm in arms}
    return {"protocol": PROTOCOL, "manifest_sha256": canonical_digest(selected),
            "config_sha256": {arm: canonical_digest(configs[arm]) for arm in arms},
            "memory": checks, "gptoss_revision": configs[arms[0]]["autonomous_support"]["model"]["model"]["model_revision"],
            "qwen_revision": configs[arms[0]]["model_revision"],
            "shard_devices": "worker CUDA ordinal 0/1; Qwen isolated on parent CUDA ordinal 0",
            "target_input_tokens": TARGET_TOKENS, "attention": CHUNKED_NAME,
            "minimum_required_free_bytes_per_gptoss_gpu": MIN_FREE_BYTES}


def _synthetic_messages(worker, target_tokens: int = TARGET_TOKENS) -> list[dict]:
    prefix = "For the runtime memory feasibility check, inspect this repeated inventory note.\n"
    def messages(n: int, suffix: str = "") -> list[dict]:
        return [{"role": "user", "content": prefix + (" inventory" * n) + suffix}]
    low, high = 0, target_tokens
    while low < high:
        middle = (low + high + 1) // 2
        if worker.count_messages_tokens(messages(middle)) <= target_tokens:
            low = middle
        else:
            high = middle - 1
    base = messages(low)
    count = worker.count_messages_tokens(base)
    if count == target_tokens:
        return base
    for filler in (" x", " z", "!", "\n"):
        for repeats in range(1, 24):
            candidate = messages(low, filler * repeats)
            if worker.count_messages_tokens(candidate) == target_tokens:
                return candidate
    raise RuntimeError(f"Could not construct exactly {target_tokens:,} GPT-OSS input tokens")


def _messages(worker, path: Path | None) -> tuple[list[dict], str]:
    if path is None:
        return _synthetic_messages(worker), "synthetic_exact_length"
    value = json.loads(path.read_text(encoding="utf-8"))
    messages = value.get("messages") if isinstance(value, dict) else value
    if not isinstance(messages, list) or not messages:
        raise ValueError("Archived request must contain a nonempty messages list")
    return messages, "archived_exact_request"


def validate_prefill_execution(records: list[dict], input_tokens: int) -> dict:
    entered = {row["layer"]: row for row in records if row.get("event") == "layer_q_proj_enter"
               and row.get("query_tokens") == input_tokens}
    completed = {row["layer"]: row for row in records if row.get("event") == "layer_complete"
                 and row.get("query_tokens") == input_tokens}
    if set(entered) != set(range(24)) or set(completed) != set(range(24)):
        raise RuntimeError("Long prefill did not execute all 24 decoder layers")
    for index in range(24):
        expected_device = f"cuda:{0 if index < 12 else 1}"
        if entered[index]["execution_device"] != expected_device:
            raise RuntimeError(f"Long prefill layer {index} ran on the wrong GPU")
    chunk_events = [row for row in records if row.get("event") == "attention_chunk_end"
                    and row.get("query_tokens") == input_tokens]
    if not any(row.get("layer") == 0 for row in chunk_events) or not any(
            row.get("layer") == 23 for row in chunk_events):
        raise RuntimeError("Long prefill did not exercise chunked attention across both GPUs")
    return {"prefill_layers_completed": 24,
            "first_layer_device": entered[0]["execution_device"],
            "last_layer_device": entered[23]["execution_device"],
            "recorded_attention_chunk_ends": len(chunk_events)}


def _worker_evidence(output_dir: Path, worker=None, error=None) -> dict:
    path = output_dir / "workers" / "gpt_oss_controller_worker.log"
    process = getattr(worker, "process", None)
    exit_code = (process.poll() if process is not None else
                 getattr(error, "worker_exit_code", None))
    evidence = {"worker_exit_code": exit_code,
                "worker_log_path": str(path), "worker_log_exists": path.exists()}
    if error is not None and getattr(error, "worker_cleanup_error", None):
        evidence["worker_cleanup_error"] = error.worker_cleanup_error
    if error is not None and getattr(error, "worker_poll_error", None):
        evidence["worker_poll_error"] = error.worker_poll_error
    if path.exists():
        try:
            with path.open("rb") as stream:
                stream.seek(0, os.SEEK_END)
                size = stream.tell()
                stream.seek(max(0, size - 8192))
                evidence["worker_log_tail"] = stream.read().decode("utf-8", errors="replace")
                evidence["worker_log_bytes"] = size
        except OSError as exc:
            evidence["worker_log_read_error"] = repr(exc)
    return evidence


def _write_result(output_dir: Path, result: dict) -> None:
    try:
        write_json_atomic(output_dir / "feasibility_result.json", result)
    except Exception as write_error:
        result["result_write_error"] = repr(write_error)
        if result.get("status") != "failed":
            result.update(status="failed", error_type=type(write_error).__name__,
                          error_detail=f"Could not write feasibility result: {write_error}")
        # An exclusive fallback avoids replacing an earlier artifact if atomic writes fail.
        try:
            with (output_dir / "feasibility_result_fallback.json").open("x", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2, sort_keys=True)
                stream.write("\n")
        except Exception as fallback_error:
            print(f"Feasibility result fallback write failed: {fallback_error!r}", file=sys.stderr,
                  flush=True)
        print(json.dumps(result, indent=2, sort_keys=True), file=sys.stderr, flush=True)


def _stage(output_dir: Path, result: dict, name: str) -> None:
    result["startup_stage"] = name
    print(f"Feasibility startup stage: {name}", file=sys.stderr, flush=True)
    row = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "stage": name,
           "pid": os.getpid(), "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
    with (output_dir / "startup_events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row) + "\n")
        stream.flush()
    write_json_atomic(output_dir / "feasibility_result.json", result)


def _check_worker_process(spec: dict, worker_devices: list[str]) -> dict:
    python = os.environ.get(spec["python_env"], "")
    if not python or not Path(python).is_file():
        raise ValueError("Pinned GPT-OSS Python executable is unavailable")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(worker_devices)
    hf_variable = spec.get("hf_home_env")
    if hf_variable:
        environment["HF_HOME"] = os.environ[hf_variable]
    script = ("import json, torch, agent_support.gpt_oss_controller_worker; "
              "print(json.dumps({'count': torch.cuda.device_count(), "
              "'names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}))")
    completed = subprocess.run([python, "-c", script], env=environment,
                               capture_output=True, text=True, timeout=60)
    evidence = {"exit_code": completed.returncode, "stderr_tail": completed.stderr[-8192:],
                "cuda_visible_devices": environment["CUDA_VISIBLE_DEVICES"]}
    if completed.returncode != 0:
        error = RuntimeError(f"GPT-OSS worker import check exited {completed.returncode}: "
                             f"{completed.stderr[-8192:]}")
        error.worker_exit_code = completed.returncode
        raise error
    identity = json.loads(completed.stdout.strip().splitlines()[-1])
    evidence.update(identity)
    if identity.get("count") != 2 or any("L40S" not in name for name in identity.get("names", [])):
        raise RuntimeError(f"GPT-OSS worker process sees incorrect GPU identities: {identity}")
    return evidence


def run(output_dir: Path, messages_file: Path | None, *, startup_only: bool = False) -> dict:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    result = {"status": "running", "request_source": None,
              "protocol": PROTOCOL, "startup_stage": "created"}
    worker = None
    try:
        _stage(output_dir, result, "preflight")
        result.update(preflight())
        _stage(output_dir, result, "clean_commit")
        result["git_commit"] = require_clean_commit()
        _stage(output_dir, result, "gpu_visibility")
        qwen_device = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        worker_devices = [item.strip() for item in os.environ.get(VISIBLE_ENV, "").split(",")]
        if (os.environ.get(SHARD_ENV) != "1" or
                not qwen_device or "," in qwen_device or
                len(worker_devices) != 2 or any(not item for item in worker_devices) or
                len({qwen_device, *worker_devices}) != 3 or
                os.environ.get(CHUNKED_ENV) != "1" or
                os.environ.get("AGENT_SUPPORT_GPU_MEMORY_DIAGNOSTICS") != "1" or
                os.environ.get("AGENT_SUPPORT_GPTOSS_CUDA_CACHE_RELEASE", "0") != "0"):
            raise RuntimeError("Three-GPU sharding and memory telemetry are not configured")
        import torch
        from .controller_backend import GptOssControllerWorker
        from .model import QwenModel

        if torch.cuda.device_count() != 1 or "L40S" not in torch.cuda.get_device_name(0):
            raise RuntimeError("Qwen process must see exactly one L40S")
        result["parent_gpu_identity"] = {"cuda_visible_devices": qwen_device,
                                         "name": torch.cuda.get_device_name(0)}
        result["worker_cuda_visible_devices"] = worker_devices
        config = config_for("autonomous_recovery_v4")
        if startup_only:
            _stage(output_dir, result, "worker_process_check")
            result["worker_process_check"] = _check_worker_process(
                config["autonomous_support"]["model"], worker_devices)
            result["status"] = "startup_valid"
            _stage(output_dir, result, "startup_valid")
            return result
        _stage(output_dir, result, "worker_initialize")
        startup_start = time.monotonic()
        try:
            worker = GptOssControllerWorker(config["autonomous_support"]["model"], output_dir / "workers")
        finally:
            result["gptoss_worker_startup_seconds"] = round(time.monotonic() - startup_start, 3)
        result["gptoss_metadata"] = worker.metadata()
        if worker.metadata().get("attention_implementation") != CHUNKED_NAME:
            raise RuntimeError("GPT-OSS did not load the chunked eager attention backend")
        _stage(output_dir, result, "qwen_initialize")
        torch.cuda.reset_peak_memory_stats(0)
        qwen_minimum_free = torch.cuda.mem_get_info(0)[0]
        qwen_samples = 0
        stop = threading.Event()
        def sample_qwen() -> None:
            nonlocal qwen_minimum_free, qwen_samples
            while not stop.wait(0.02):
                free = torch.cuda.mem_get_info(0)[0]
                qwen_minimum_free = min(qwen_minimum_free, free)
                qwen_samples += 1
        sampler = threading.Thread(target=sample_qwen, daemon=True)
        sampler.start()
        try:
            qwen = QwenModel(config)
        finally:
            stop.set()
            sampler.join(timeout=5)
        qwen_minimum_free = min(qwen_minimum_free, torch.cuda.mem_get_info(0)[0])
        qwen_map = {str(key): str(value) for key, value in qwen.model.hf_device_map.items()}
        if qwen_map != {"": "0"}:
            raise RuntimeError(f"Qwen did not remain entirely on GPU 0: {qwen_map}")
        result["qwen_hf_device_map"] = qwen_map
        result["qwen_gpu_memory"] = {"allocated_bytes": torch.cuda.memory_allocated(0),
                                     "reserved_bytes": torch.cuda.memory_reserved(0),
                                     "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                                     "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
                                     "minimum_global_free_bytes": qwen_minimum_free,
                                     "free_memory_samples": qwen_samples,
                                     "free_bytes": torch.cuda.mem_get_info(0)[0]}
        if qwen_samples < 1:
            raise RuntimeError("Qwen GPU-0 free memory was not sampled during model load")
        _stage(output_dir, result, "request_prepare")
        messages, source = _messages(worker, messages_file)
        result["request_source"] = source
        count = worker.count_messages_tokens(messages)
        if count != TARGET_TOKENS:
            raise RuntimeError(f"Feasibility request has {count} input tokens, expected {TARGET_TOKENS}")
        result["input_tokens"] = count
        result["messages_sha256"] = canonical_digest(messages)
        # One short warmup followed by exactly one long request; no environment advances.
        _stage(output_dir, result, "warmup_generate")
        worker.generate([{"role": "user", "content": "Reply briefly."}], max_new_tokens=32)
        _stage(output_dir, result, "long_generate")
        generation = worker.generate(messages, max_new_tokens=1024)
        result["generation"] = {key: generation[key] for key in
                                ("input_tokens", "output_tokens", "generation_status", "inference_seconds")}
        if (generation["input_tokens"] != TARGET_TOKENS or generation["output_tokens"] < 1 or
                generation["generation_status"] != "complete"):
            raise RuntimeError("Long GPT-OSS generation did not complete successfully")
        _stage(output_dir, result, "validate_generation")
        files = list((output_dir / "diagnostics").glob("gpt_oss_cuda_memory_*.jsonl"))
        if len(files) != 1:
            raise RuntimeError("Expected one GPT-OSS memory telemetry file")
        records = [json.loads(line) for line in files[0].read_text().splitlines()]
        ends = [row for row in records if row.get("event") == "generation_end"]
        starts = [row for row in records if row.get("event") == "generation_start"]
        if len(ends) != 2 or len(starts) != 2 or ends[-1]["status"] != "ok":
            raise RuntimeError("Incomplete two-GPU generation telemetry")
        result["forward_execution"] = validate_prefill_execution(records, TARGET_TOKENS)
        result["gpu_memory"] = {key: {"peak_allocated_bytes": value["peak_allocated_bytes"],
                                      "peak_reserved_bytes": value["peak_reserved_bytes"],
                                      "minimum_global_free_bytes": ends[-1]["minimum_global_free_bytes"][key],
                                      "free_memory_samples": ends[-1]["free_memory_samples"][key],
                                      "pre_generation_allocated_bytes": starts[-1]["memory"]["devices"][key]["allocated_bytes"]}
                                for key, value in ends[-1]["memory"]["devices"].items()}
        if ends[-1]["sampling_errors"] or any(
                row["free_memory_samples"] < 1 or
                row["minimum_global_free_bytes"] < MIN_FREE_BYTES or
                row["peak_allocated_bytes"] <= row["pre_generation_allocated_bytes"]
                for row in result["gpu_memory"].values()):
            raise RuntimeError("Two-GPU generation allocations or memory headroom failed validation")
        result["status"] = "infrastructure_valid"
        _stage(output_dir, result, "infrastructure_valid")
    except BaseException as exc:
        result.update(status="failed", error_type=type(exc).__name__, error_detail=str(exc),
                      traceback=traceback.format_exc())
        try:
            result["worker_evidence"] = _worker_evidence(output_dir, worker, exc)
        except Exception as evidence_error:
            result["worker_evidence_error"] = repr(evidence_error)
        print(result["traceback"], file=sys.stderr, flush=True)
    finally:
        if worker is not None:
            try:
                worker.close()
            except Exception as exc:
                result["close_error"] = repr(exc)
                if result["status"] != "failed":
                    result.update(status="failed", error_type=type(exc).__name__,
                                  error_detail=f"Worker cleanup failed: {exc}",
                                  traceback=traceback.format_exc())
        _write_result(output_dir, result)
    return result


def gate(path: Path) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    expected = preflight()
    if (result.get("status") != "infrastructure_valid" or
            result.get("git_commit") != require_clean_commit() or
            any(result.get(key) != value for key, value in expected.items())):
        raise RuntimeError("Three-GPU feasibility gate is not infrastructure valid for this commit")
    return {"status": "gate_passed", "feasibility_result": str(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    test = sub.add_parser("test")
    test.add_argument("--output-dir", required=True, type=Path)
    test.add_argument("--messages-file", type=Path)
    startup = sub.add_parser("startup-check")
    startup.add_argument("--output-dir", required=True, type=Path)
    check = sub.add_parser("gate")
    check.add_argument("--result", required=True, type=Path)
    args = parser.parse_args()
    result = (preflight() if args.command == "preflight" else
              gate(args.result) if args.command == "gate" else
              run(args.output_dir, args.messages_file if args.command == "test" else None,
                  startup_only=args.command == "startup-check"))
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.command == "test" and result["status"] != "infrastructure_valid":
        raise SystemExit(1)
    if args.command == "startup-check" and result["status"] != "startup_valid":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
