"""CPU preflight and one bounded three-GPU GPT-OSS feasibility request."""

from __future__ import annotations

import argparse
import json
import os
import threading
from pathlib import Path

from .common import canonical_digest, write_json_atomic
from .autonomous_evaluation import require_clean_commit
from .gpt_oss_two_gpu import SHARD_ENV, VISIBLE_ENV
from .recovery_v4_eval import config_for, manifest, memory_checks


PROTOCOL = "three_gpu_mxfp4_eager_v1"
TARGET_TOKENS = 8887


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
            "target_input_tokens": TARGET_TOKENS, "attention": "eager"}


def _synthetic_messages(worker) -> list[dict]:
    prefix = "For the runtime memory feasibility check, inspect this repeated inventory note.\n"
    def messages(n: int, suffix: str = "") -> list[dict]:
        return [{"role": "user", "content": prefix + (" inventory" * n) + suffix}]
    low, high = 0, TARGET_TOKENS
    while low < high:
        middle = (low + high + 1) // 2
        if worker.count_messages_tokens(messages(middle)) <= TARGET_TOKENS:
            low = middle
        else:
            high = middle - 1
    base = messages(low)
    count = worker.count_messages_tokens(base)
    if count == TARGET_TOKENS:
        return base
    for filler in (" x", " z", "!", "\n"):
        for repeats in range(1, 24):
            candidate = messages(low, filler * repeats)
            if worker.count_messages_tokens(candidate) == TARGET_TOKENS:
                return candidate
    raise RuntimeError("Could not construct exactly 8,887 GPT-OSS input tokens")


def _messages(worker, path: Path | None) -> tuple[list[dict], str]:
    if path is None:
        return _synthetic_messages(worker), "synthetic_exact_length"
    value = json.loads(path.read_text(encoding="utf-8"))
    messages = value.get("messages") if isinstance(value, dict) else value
    if not isinstance(messages, list) or not messages:
        raise ValueError("Archived request must contain a nonempty messages list")
    return messages, "archived_exact_request"


def run(output_dir: Path, messages_file: Path | None) -> dict:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    result = {**preflight(), "status": "running", "request_source": None,
              "git_commit": require_clean_commit()}
    worker = None
    try:
        qwen_device = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        worker_devices = [item.strip() for item in os.environ.get(VISIBLE_ENV, "").split(",")]
        if (os.environ.get(SHARD_ENV) != "1" or
                not qwen_device or "," in qwen_device or
                len(worker_devices) != 2 or any(not item for item in worker_devices) or
                len({qwen_device, *worker_devices}) != 3 or
                os.environ.get("AGENT_SUPPORT_GPU_MEMORY_DIAGNOSTICS") != "1" or
                os.environ.get("AGENT_SUPPORT_GPTOSS_CUDA_CACHE_RELEASE", "0") != "0"):
            raise RuntimeError("Three-GPU sharding and memory telemetry are not configured")
        import torch
        from .controller_backend import GptOssControllerWorker
        from .model import QwenModel

        if torch.cuda.device_count() != 1 or "L40S" not in torch.cuda.get_device_name(0):
            raise RuntimeError("Qwen process must see exactly one L40S")
        config = config_for("autonomous_recovery_v4")
        worker = GptOssControllerWorker(config["autonomous_support"]["model"], output_dir / "workers")
        result["gptoss_metadata"] = worker.metadata()
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
        messages, source = _messages(worker, messages_file)
        result["request_source"] = source
        count = worker.count_messages_tokens(messages)
        if count != TARGET_TOKENS:
            raise RuntimeError(f"Feasibility request has {count} input tokens, expected {TARGET_TOKENS}")
        result["input_tokens"] = count
        result["messages_sha256"] = canonical_digest(messages)
        # One short warmup followed by exactly one long request; no environment advances.
        worker.generate([{"role": "user", "content": "Reply briefly."}], max_new_tokens=32)
        generation = worker.generate(messages, max_new_tokens=1024)
        result["generation"] = {key: generation[key] for key in
                                ("input_tokens", "output_tokens", "generation_status", "inference_seconds")}
        if generation["input_tokens"] != TARGET_TOKENS or generation["output_tokens"] < 1:
            raise RuntimeError("Long GPT-OSS generation did not run")
        files = list((output_dir / "diagnostics").glob("gpt_oss_cuda_memory_*.jsonl"))
        if len(files) != 1:
            raise RuntimeError("Expected one GPT-OSS memory telemetry file")
        records = [json.loads(line) for line in files[0].read_text().splitlines()]
        ends = [row for row in records if row.get("event") == "generation_end"]
        starts = [row for row in records if row.get("event") == "generation_start"]
        if len(ends) != 2 or len(starts) != 2 or ends[-1]["status"] != "ok":
            raise RuntimeError("Incomplete two-GPU generation telemetry")
        result["gpu_memory"] = {key: {"peak_allocated_bytes": value["peak_allocated_bytes"],
                                      "peak_reserved_bytes": value["peak_reserved_bytes"],
                                      "minimum_global_free_bytes": ends[-1]["minimum_global_free_bytes"][key],
                                      "free_memory_samples": ends[-1]["free_memory_samples"][key],
                                      "pre_generation_allocated_bytes": starts[-1]["memory"]["devices"][key]["allocated_bytes"]}
                                for key, value in ends[-1]["memory"]["devices"].items()}
        if ends[-1]["sampling_errors"] or any(
                row["free_memory_samples"] < 1 or
                row["peak_allocated_bytes"] <= row["pre_generation_allocated_bytes"]
                for row in result["gpu_memory"].values()):
            raise RuntimeError("Temporary generation allocations were not verified on both GPT-OSS GPUs")
        result["status"] = "infrastructure_valid"
    except Exception as exc:
        result.update(status="failed", error_type=type(exc).__name__, error_detail=str(exc))
    finally:
        if worker is not None:
            try:
                worker.close()
            except Exception as exc:
                result.update(status="failed", close_error=repr(exc))
        write_json_atomic(output_dir / "feasibility_result.json", result)
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
    check = sub.add_parser("gate")
    check.add_argument("--result", required=True, type=Path)
    args = parser.parse_args()
    result = (preflight() if args.command == "preflight" else
              gate(args.result) if args.command == "gate" else
              run(args.output_dir, args.messages_file))
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.command == "test" and result["status"] != "infrastructure_valid":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
