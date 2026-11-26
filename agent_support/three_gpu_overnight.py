"""One-load, gated overnight GPT-OSS diagnosis; no benchmark tasks run here."""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path

from .autonomous_evaluation import require_clean_commit
from .common import write_json_atomic
from .gpt_oss_chunked_eager import NAME as CHUNKED_NAME
from .gpt_oss_two_gpu import SHARD_ENV, VISIBLE_ENV
from .recovery_v4_eval import config_for
from .three_gpu_feasibility import TARGET_TOKENS, _messages, _synthetic_messages, preflight
from .three_gpu_performance_probe import validate_stage

STAGES = (("warmup_first", 8, 600, 1200),
          ("warmup_repeat", 8, 120, 300),
          ("medium_prefill", 2, 360, 900),
          ("long_prefill", 2, 900, 1200))
MIN_CACHE_MIB_PER_SECOND = 8


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # A preempted append may leave one partial final line.
    return rows


def _telemetry_summary(directory: Path) -> dict:
    result = {}
    for kind, pattern in (("load", "gpt_oss_model_load_*.jsonl"),
                          ("inference", "gpt_oss_inference_*.jsonl")):
        files = sorted(directory.glob(pattern))
        records = [row for path in files for row in _rows(path)]
        samples = [row for row in records if row.get("event") in
                   {"sample", "start", "end", "model_load_start"} and "gpus" in row]
        gpu_util = {"cuda:0": [], "cuda:1": []}
        gpu_memory = {"cuda:0": [], "cuda:1": []}
        cpu = []
        reads = 0
        for row in samples:
            if isinstance(row.get("cpu_percent_one_core"), (int, float)):
                cpu.append(row["cpu_percent_one_core"])
            reads += max(0, (row.get("io_delta") or {}).get("read_bytes", 0))
            gpu = row.get("gpus", {}).get("worker_cuda_ordinals", {})
            for device in gpu_util:
                value = gpu.get(device, {})
                if isinstance(value.get("gpu_utilization_percent"), int):
                    gpu_util[device].append(value["gpu_utilization_percent"])
                if isinstance(value.get("memory_used_mib"), int):
                    gpu_memory[device].append(value["memory_used_mib"])
        result[kind] = {"files": [str(path) for path in files], "samples": len(samples),
                        "from_pretrained_seconds": next((row.get("elapsed_seconds") for row in records
                                                          if row.get("event") == "model_load_end"), None),
                        "progress_stalls": sum(row.get("event") == "progress_stall" for row in records),
                        "weights_completed": max((row.get("completed", 0) for row in records
                                                  if row.get("event") == "weight_progress"), default=0),
                        "read_bytes_delta_sum": reads,
                        "cpu_percent_one_core_mean": round(sum(cpu) / len(cpu), 1) if cpu else None,
                        "gpu_utilization_percent_mean": {key: round(sum(v) / len(v), 1) if v else None
                                                         for key, v in gpu_util.items()},
                        "gpu_memory_used_mib_peak": {key: max(v) if v else None
                                                     for key, v in gpu_memory.items()}}
    return result


def _inference_stage_summary(path: Path) -> dict:
    rows = [row for row in _rows(path) if row.get("event") in {"start", "sample", "end"}]
    utilization = {}
    for index in (0, 1):
        key = f"cuda:{index}"
        values = [row.get("gpus", {}).get("worker_cuda_ordinals", {}).get(key, {}).get(
            "gpu_utilization_percent") for row in rows]
        values = [value for value in values if isinstance(value, int)]
        utilization[key] = round(sum(values) / len(values), 1) if values else None
    return {"file": str(path), "sample_count": len(rows),
            "gpu_utilization_percent_mean": utilization,
            "process_read_bytes_delta_sum": sum(max(0, (row.get("io_delta") or {}).get("read_bytes", 0))
                                                for row in rows)}


def write_report(output_dir: Path, result: dict) -> None:
    telemetry = _telemetry_summary(output_dir / "diagnostics")
    lines = ["# Three-GPU overnight diagnosis", "", f"Status: {result.get('status')}",
             f"Commit: {result.get('git_commit', 'unknown')}",
             f"Failure: {result.get('error_type', 'none')}: {result.get('error_detail', 'none')}", "",
             f"GPT-OSS worker startup: {result.get('gptoss_worker_startup_seconds', 'incomplete')} s",
             f"GPT-OSS from_pretrained: {telemetry['load']['from_pretrained_seconds']} s",
             f"Qwen load: {result.get('qwen_load_seconds', 'incomplete')} s", "",
             "| Stage | Input/output tokens | Wall s | Prefill s | GPU 0/1 util % | GPU 0/1 peak GiB | GPU 0/1 min free GiB | Status |",
             "| --- | --- | ---: | ---: | --- | --- | --- | --- |"]
    if result.get("cache_result"):
        try:
            cache = json.loads(Path(result["cache_result"]).read_text(encoding="utf-8"))
            lines.insert(8, f"CPU cache read: {cache.get('throughput_mib_per_second', '?')} MiB/s "
                       f"over {cache.get('bytes_read', '?')} bytes")
        except (OSError, ValueError):
            lines.insert(8, "CPU cache read: result unavailable")
    if result.get("qwen_gpu_memory"):
        lines.insert(8, f"Qwen GPU-0 peak reserved: "
                     f"{round(result['qwen_gpu_memory']['peak_reserved_bytes'] / 1024**3, 2)} GiB")
    gib = 1024 ** 3
    for stage in result.get("stages", []):
        validation = stage.get("validation", {})
        memory = validation.get("memory", {})
        def pair(field):
            return "/".join(str(round(memory.get(f"cuda:{index}", {}).get(field, 0) / gib, 2))
                            if memory.get(f"cuda:{index}") else "?" for index in (0, 1))
        utilization = stage.get("inference_telemetry", {}).get("gpu_utilization_percent_mean", {})
        util_pair = "/".join(str(utilization.get(f"cuda:{index}", "?")) for index in (0, 1))
        lines.append(f"| {stage['name']} | {stage.get('input_tokens', '?')}/{stage.get('output_tokens', '?')} "
                     f"| {stage.get('wall_seconds', '?')} | "
                     f"{validation.get('timing', {}).get('prefill_observed_seconds', '?')} | "
                     f"{util_pair} | {pair('peak_reserved_bytes')} | {pair('minimum_global_free_bytes')} | "
                     f"{stage.get('status', '?')} |")
    lines.extend(["", "## Process and GPU samples", "",
                  "GPU utilization is the mean of `nvidia-smi` samples; memory is the sampled physical peak.", ""])
    for kind, value in telemetry.items():
        lines.append(f"- {kind}: {value['samples']} samples; process reads {value['read_bytes_delta_sum']} bytes; "
                     f"mean CPU {value['cpu_percent_one_core_mean']}% of one core; "
                     f"GPU utilization {value['gpu_utilization_percent_mean']}; "
                     f"physical GPU memory peak MiB {value['gpu_memory_used_mib_peak']}; "
                     f"weights {value['weights_completed']}, progress stalls {value['progress_stalls']}.")
    lines.extend(["", f"Profiler: {result.get('profiler', {}).get('profile_path', 'not run')}",
                  "", "The JSON result and append-only JSONL files contain detailed timestamps and failures.", ""])
    target = output_dir / "summary.md"
    temporary = output_dir / "summary.md.tmp"
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(target)


def cache_probe(output_dir: Path, hf_home: Path, *,
                sample_limit_bytes: int = 512 * 1024**2,
                per_file_bytes: int = 128 * 1024**2,
                min_sample_bytes: int = 128 * 1024**2,
                min_speed_mib_per_second: int = MIN_CACHE_MIB_PER_SECOND) -> dict:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    result = {"status": "running", "git_commit": require_clean_commit()}
    try:
        revision = config_for("autonomous_recovery_v4")["autonomous_support"]["model"]["model"]["model_revision"]
        snapshot = hf_home / "hub" / "models--openai--gpt-oss-20b" / "snapshots" / revision
        files = sorted(snapshot.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"No pinned GPT-OSS safetensors in {snapshot}")
        measured = 0
        details = []
        for file in files:
            if measured >= sample_limit_bytes:
                break
            size = min(per_file_bytes, sample_limit_bytes - measured)
            start = time.monotonic()
            count = 0
            with file.open("rb") as stream:
                while count < size:
                    block = stream.read(min(4 * 1024**2, size - count))
                    if not block:
                        break
                    count += len(block)
            details.append({"file": file.name, "bytes_read": count,
                            "seconds": round(time.monotonic() - start, 3)})
            measured += count
        elapsed = sum(row["seconds"] for row in details)
        speed = measured / max(elapsed, 0.001) / 1024**2
        result.update(snapshot=str(snapshot), files=details, bytes_read=measured,
                      throughput_mib_per_second=round(speed, 2))
        if measured < min_sample_bytes or speed < min_speed_mib_per_second:
            raise RuntimeError("Pinned checkpoint read throughput or sample size failed the CPU cache gate")
        result["status"] = "cache_valid"
    except Exception as exc:
        result.update(status="failed", error_type=type(exc).__name__, error_detail=str(exc))
    finally:
        write_json_atomic(output_dir / "cache_result.json", result)
        (output_dir / "summary.md").write_text(
            f"# Read-only pinned cache probe\n\nStatus: {result['status']}\n\n"
            f"Throughput: {result.get('throughput_mib_per_second', '?')} MiB/s\n\n"
            f"Failure: {result.get('error_detail', 'none')}\n", encoding="utf-8")
    return result


def run(output_dir: Path, messages_file: Path | None, cache_result: Path | None) -> dict:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    result = {"status": "running", "git_commit": require_clean_commit(),
              "preflight": preflight(), "stages": []}
    worker = None
    previous_handlers = {}
    def interrupted(signum, _frame):
        raise InterruptedError(f"Received signal {signum}")
    try:
        for signum in (signal.SIGTERM, signal.SIGUSR1, signal.SIGALRM):
            previous_handlers[signum] = signal.signal(signum, interrupted)
        if cache_result is not None:
            cache = json.loads(cache_result.read_text(encoding="utf-8"))
            if cache.get("status") != "cache_valid" or cache.get("git_commit") != result["git_commit"]:
                raise RuntimeError("CPU cache dependency is invalid for this commit")
            result["cache_result"] = str(cache_result)
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        other = [part.strip() for part in os.environ.get(VISIBLE_ENV, "").split(",")]
        required = (SHARD_ENV, "AGENT_SUPPORT_GPTOSS_CHUNKED_EAGER",
                    "AGENT_SUPPORT_GPU_MEMORY_DIAGNOSTICS",
                    "AGENT_SUPPORT_GPTOSS_MODEL_LOAD_DIAGNOSTICS",
                    "AGENT_SUPPORT_GPTOSS_INFERENCE_SAMPLING",
                    "AGENT_SUPPORT_GPTOSS_ALLOW_PROFILER")
        if (any(os.environ.get(key) != "1" for key in required) or
                not visible or "," in visible or len(other) != 2 or
                len({visible, *other}) != 3):
            raise RuntimeError("Three-GPU overnight environment is incomplete")
        import torch
        from .controller_backend import GptOssControllerWorker
        from .model import QwenModel
        if torch.cuda.device_count() != 1 or "L40S" not in torch.cuda.get_device_name(0):
            raise RuntimeError("Qwen process must see exactly one L40S")
        result["qwen_gpu_identity"] = {"cuda_visible_devices": visible,
                                       "name": torch.cuda.get_device_name(0)}
        result["gptoss_cuda_visible_devices"] = other
        config = config_for("autonomous_recovery_v4")
        started = time.monotonic()
        try:
            worker = GptOssControllerWorker(config["autonomous_support"]["model"], output_dir / "workers")
        finally:
            result["gptoss_worker_startup_seconds"] = round(time.monotonic() - started, 3)
            write_json_atomic(output_dir / "campaign_result.json", result)
            write_report(output_dir, result)
        metadata = worker.metadata()
        if (metadata.get("attention_implementation") != CHUNKED_NAME or
                metadata.get("verified_expert_modules_by_device") != {"cuda:0": 12, "cuda:1": 12}):
            raise RuntimeError("GPT-OSS attention or packed expert placement differs")
        result["gptoss_metadata"] = metadata
        torch.cuda.reset_peak_memory_stats(0)
        started = time.monotonic()
        try:
            signal.setitimer(signal.ITIMER_REAL, 5400)
            qwen = QwenModel(config)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            result["qwen_load_seconds"] = round(time.monotonic() - started, 3)
            write_json_atomic(output_dir / "campaign_result.json", result)
            write_report(output_dir, result)
        qwen_map = {str(key): str(value) for key, value in qwen.model.hf_device_map.items()}
        if qwen_map != {"": "0"}:
            raise RuntimeError(f"Qwen placement differs: {qwen_map}")
        result["qwen_hf_device_map"] = qwen_map
        result["qwen_gpu_memory"] = {"allocated_bytes": int(torch.cuda.memory_allocated(0)),
                                     "reserved_bytes": int(torch.cuda.memory_reserved(0)),
                                     "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
                                     "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0))}
        warmup = [{"role": "user", "content": "Reply briefly."}]
        medium = None
        long = None
        for request_id, (name, max_tokens, soft, hard) in enumerate(STAGES, 1):
            if name == "medium_prefill":
                medium = _synthetic_messages(worker, 2048)
            if name == "long_prefill":
                long, source = _messages(worker, messages_file)
                if worker.count_messages_tokens(long) != TARGET_TOKENS:
                    raise RuntimeError("Long request is not 8,887 tokens")
                result["request_source"] = source
            prompt = warmup if name.startswith("warmup") else (medium if name == "medium_prefill" else long)
            stage = {"name": name, "request_id": request_id, "max_new_tokens": max_tokens,
                     "soft_seconds": soft, "hard_seconds": hard, "status": "running"}
            result["stages"].append(stage)
            write_json_atomic(output_dir / "campaign_result.json", result)
            write_report(output_dir, result)
            worker.timeout = hard
            started = time.monotonic()
            generation = worker.generate(prompt, max_new_tokens=max_tokens)
            stage.update(wall_seconds=round(time.monotonic() - started, 3),
                         inference_seconds=generation["inference_seconds"],
                         input_tokens=generation["input_tokens"],
                         output_tokens=generation["output_tokens"])
            files = list((output_dir / "diagnostics").glob("gpt_oss_cuda_memory_*.jsonl"))
            if len(files) != 1 or generation["output_tokens"] < 2:
                raise RuntimeError(f"{name} lacks CUDA telemetry or two generated tokens")
            stage["validation"] = validate_stage(_rows(files[0]), request_id, stage["input_tokens"])
            sampler_files = sorted((output_dir / "diagnostics").glob("gpt_oss_inference_*.jsonl"))
            if len(sampler_files) != request_id:
                raise RuntimeError(f"{name} lacks one inference sampler file per completed stage")
            stage["inference_telemetry"] = _inference_stage_summary(sampler_files[-1])
            stage["status"] = "valid" if stage["wall_seconds"] <= soft else "slow"
            write_json_atomic(output_dir / "campaign_result.json", result)
            write_report(output_dir, result)
            if stage["status"] == "slow":
                result["profiler"] = {"status": "running", "trigger": name}
                write_json_atomic(output_dir / "campaign_result.json", result)
                worker.timeout = 600
                try:
                    response = worker.profile_generate(warmup)
                    result["profiler"] = {"status": "complete", "trigger": name,
                                          "profile_path": response["profile_path"],
                                          "inference_seconds": response["generation"]["inference_seconds"]}
                except Exception as exc:
                    result["profiler"] = {"status": "failed", "trigger": name,
                                          "error": repr(exc)}
                raise RuntimeError(f"{name} exceeded {soft}s latency gate")
        result["status"] = "performance_valid"
    except BaseException as exc:
        result.update(status="failed", error_type=type(exc).__name__, error_detail=str(exc))
        if result["stages"] and result["stages"][-1]["status"] == "running":
            result["stages"][-1].update(status="error", error_type=type(exc).__name__,
                                        error_detail=str(exc))
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        if worker is not None:
            try:
                worker.close()
            except Exception as exc:
                result["close_error"] = repr(exc)
        write_json_atomic(output_dir / "campaign_result.json", result)
        write_report(output_dir, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cache = commands.add_parser("cache")
    cache.add_argument("--output-dir", type=Path, required=True)
    cache.add_argument("--hf-home", type=Path, required=True)
    gpu = commands.add_parser("gpu")
    gpu.add_argument("--output-dir", type=Path, required=True)
    gpu.add_argument("--cache-result", type=Path)
    gpu.add_argument("--messages-file", type=Path)
    summary = commands.add_parser("summary")
    summary.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "cache":
        result = cache_probe(args.output_dir, args.hf_home)
        valid = result["status"] == "cache_valid"
    elif args.command == "gpu":
        result = run(args.output_dir, args.messages_file, args.cache_result)
        valid = result["status"] == "performance_valid"
    else:
        result = json.loads((args.output_dir / "campaign_result.json").read_text())
        write_report(args.output_dir, result)
        valid = True
    print(json.dumps(result, indent=2, sort_keys=True))
    if not valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
