"""Bounded GPT-OSS latency and memory gate before three-GPU feasibility."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

from .autonomous_evaluation import require_clean_commit
from .common import write_json_atomic
from .gpt_oss_chunked_eager import ENABLE_ENV as CHUNKED_ENV, NAME as CHUNKED_NAME
from .gpt_oss_load_diagnostics import ENABLE_ENV as LOAD_DIAGNOSTICS_ENV
from .gpt_oss_two_gpu import SHARD_ENV, VISIBLE_ENV
from .recovery_v4_eval import config_for
from .three_gpu_feasibility import (MIN_FREE_BYTES, TARGET_TOKENS, _messages,
                                    preflight, validate_prefill_execution)


STAGES = (("warmup_first", 8, 600), ("warmup_repeat", 8, 120),
          ("long_prefill", 2, 900))


def validate_stage(records: list[dict], request_id: int, input_tokens: int) -> dict:
    stage = [row for row in records if row.get("request_id") == request_id]
    starts = [row for row in stage if row.get("event") == "generation_start"]
    ends = [row for row in stage if row.get("event") == "generation_end"]
    if len(starts) != 1 or len(ends) != 1 or ends[0].get("status") != "ok":
        raise RuntimeError(f"Request {request_id} lacks successful generation telemetry")
    forward = validate_prefill_execution(stage, input_tokens)
    def timestamp(row):
        return datetime.fromisoformat(row["timestamp_utc"])
    entered = {row["layer"]: row for row in stage if row.get("event") == "layer_q_proj_enter"
               and row.get("query_tokens") == input_tokens}
    completed = {row["layer"]: row for row in stage if row.get("event") == "layer_complete"
                 and row.get("query_tokens") == input_tokens}
    chunk_starts = {}
    chunk_ends = {}
    for row in stage:
        if row.get("query_tokens") != input_tokens:
            continue
        if row.get("event") == "attention_chunk_start":
            chunk_starts.setdefault(row["layer"], row)
        elif row.get("event") == "attention_chunk_end":
            chunk_ends[row["layer"]] = row
    if set(chunk_starts) != set(range(24)) or set(chunk_ends) != set(range(24)):
        raise RuntimeError(f"Request {request_id} lacks attention timing on all 24 layers")
    attention_intervals = {str(index): round((timestamp(chunk_ends[index]) -
                                              timestamp(chunk_starts[index])).total_seconds(), 3)
                           for index in set(chunk_starts) & set(chunk_ends)}
    timing = {"prefill_observed_seconds": round((timestamp(completed[23]) -
                                                  timestamp(entered[0])).total_seconds(), 3),
              "attention_observed_seconds_by_layer": attention_intervals,
              "boundary_gap_observed_seconds": round((timestamp(entered[12]) -
                                                       timestamp(completed[11])).total_seconds(), 3),
              "after_prefill_observed_seconds": round((timestamp(ends[0]) -
                                                       timestamp(completed[23])).total_seconds(), 3)}
    devices = ends[0]["memory"]["devices"]
    if set(devices) != {"cuda:0", "cuda:1"} or ends[0].get("sampling_errors"):
        raise RuntimeError(f"Request {request_id} lacks reliable two-GPU memory telemetry")
    memory = {}
    for device, values in devices.items():
        free = ends[0]["minimum_global_free_bytes"][device]
        samples = ends[0]["free_memory_samples"][device]
        if samples < 1 or free < MIN_FREE_BYTES:
            raise RuntimeError(f"Request {request_id} lacks {MIN_FREE_BYTES} bytes headroom on {device}")
        memory[device] = {"peak_allocated_bytes": values["peak_allocated_bytes"],
                          "peak_reserved_bytes": values["peak_reserved_bytes"],
                          "minimum_global_free_bytes": free, "free_memory_samples": samples}
    return {"forward": forward, "memory": memory, "timing": timing}


def run(output_dir: Path, messages_file: Path | None = None) -> dict:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    result = {"status": "running", "git_commit": require_clean_commit(),
              "preflight": preflight(), "stages": []}
    worker = None
    try:
        qwen_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        worker_visible = [part.strip() for part in os.environ.get(VISIBLE_ENV, "").split(",")]
        if (os.environ.get(SHARD_ENV) != "1" or os.environ.get(CHUNKED_ENV) != "1" or
                os.environ.get(LOAD_DIAGNOSTICS_ENV) != "1" or
                os.environ.get("AGENT_SUPPORT_GPU_MEMORY_DIAGNOSTICS") != "1" or
                not qwen_visible or "," in qwen_visible or len(worker_visible) != 2 or
                len({qwen_visible, *worker_visible}) != 3):
            raise RuntimeError("Three-GPU performance probe environment is incomplete")
        import torch
        from .controller_backend import GptOssControllerWorker
        if torch.cuda.device_count() != 1 or "L40S" not in torch.cuda.get_device_name(0):
            raise RuntimeError("Parent process must see only its allocated L40S")
        config = config_for("autonomous_recovery_v4")
        started = time.monotonic()
        try:
            worker = GptOssControllerWorker(config["autonomous_support"]["model"], output_dir / "workers")
        finally:
            result["worker_startup_seconds"] = round(time.monotonic() - started, 3)
            write_json_atomic(output_dir / "performance_result.json", result)
        metadata = worker.metadata()
        if (metadata.get("attention_implementation") != CHUNKED_NAME or
                metadata.get("verified_expert_modules_by_device") != {"cuda:0": 12, "cuda:1": 12}):
            raise RuntimeError("GPT-OSS backend or packed expert placement differs")
        result["gptoss_metadata"] = {"attention_implementation": metadata["attention_implementation"],
                                     "sharding": metadata.get("sharding"),
                                     "verified_expert_modules_by_device":
                                         metadata["verified_expert_modules_by_device"]}
        messages, source = _messages(worker, messages_file)
        if worker.count_messages_tokens(messages) != TARGET_TOKENS:
            raise RuntimeError("Long probe request does not contain 8,887 input tokens")
        result["request_source"] = source
        warmup = [{"role": "user", "content": "Reply briefly."}]
        frozen_generation_timeout = worker.timeout
        for request_id, (name, max_tokens, max_seconds) in enumerate(STAGES, 1):
            prompt = messages if name == "long_prefill" else warmup
            stage = {"name": name, "request_id": request_id, "max_new_tokens": max_tokens,
                     "maximum_seconds": max_seconds, "status": "running"}
            result["stages"].append(stage)
            write_json_atomic(output_dir / "performance_result.json", result)
            # This probe fails early; the frozen evaluation RPC deadline stays unchanged.
            worker.timeout = min(frozen_generation_timeout, max_seconds + 15)
            started = time.monotonic()
            generation = worker.generate(prompt, max_new_tokens=max_tokens)
            stage.update(status="complete", wall_seconds=round(time.monotonic() - started, 3),
                         inference_seconds=generation["inference_seconds"],
                         input_tokens=generation["input_tokens"],
                         output_tokens=generation["output_tokens"])
            write_json_atomic(output_dir / "performance_result.json", result)
            if generation["output_tokens"] < 2 or stage["wall_seconds"] > max_seconds:
                raise RuntimeError(f"{name} failed bounded decode or latency gate")
            files = list((output_dir / "diagnostics").glob("gpt_oss_cuda_memory_*.jsonl"))
            if len(files) != 1:
                raise RuntimeError("Expected one GPT-OSS CUDA telemetry file")
            records = [json.loads(line) for line in files[0].read_text().splitlines()]
            stage["validation"] = validate_stage(records, request_id, stage["input_tokens"])
            write_json_atomic(output_dir / "performance_result.json", result)
        worker.timeout = frozen_generation_timeout
        result["status"] = "performance_valid"
    except Exception as exc:
        result.update(status="failed", error_type=type(exc).__name__, error_detail=str(exc))
    finally:
        if worker is not None:
            try:
                worker.close()
            except Exception as exc:
                result.update(status="failed", close_error=repr(exc))
        write_json_atomic(output_dir / "performance_result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--messages-file", type=Path)
    args = parser.parse_args()
    result = run(args.output_dir, args.messages_file)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "performance_valid":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
