"""Read-only analysis of separate GPT-OSS and Qwen CUDA diagnostic JSONL logs.

Usage: python -m agent_support.gpu_memory_analysis RUN_DIR [RUN_DIR ...]
The command prints JSON; it never changes the supplied runs or their telemetry.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

GIB = 1024 ** 3
PREFIXES = {"gpt_oss_cuda_memory_": "gpt_oss", "qwen_cuda_memory_": "qwen"}


def _gib(snapshot: dict, key: str) -> float | None:
    value = snapshot.get(key)
    return round(value / GIB, 4) if type(value) in (int, float) else None


def telemetry_files(paths) -> list[Path]:
    files = set()
    for raw in paths:
        path = Path(raw)
        if path.is_file() and any(path.name.startswith(prefix) for prefix in PREFIXES):
            files.add(path)
        elif path.is_dir():
            files.update(candidate for candidate in path.rglob("*.jsonl")
                         if any(candidate.name.startswith(prefix) for prefix in PREFIXES))
        else:
            raise ValueError(f"No telemetry file or directory: {path}")
    if not files:
        raise ValueError("No GPT-OSS or Qwen CUDA telemetry JSONL files found")
    return sorted(files)


def analyze(paths) -> dict:
    records = []
    model_loads = []
    for path in telemetry_files(paths):
        model = next(label for prefix, label in PREFIXES.items() if path.name.startswith(prefix))
        starts, ends, releases = {}, {}, {}
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            kind, request_id = event.get("event"), event.get("request_id")
            if kind == "generation_start":
                if request_id in starts:
                    raise ValueError(f"{path}: duplicate generation_start for request {request_id}")
                starts[request_id] = event
            elif kind == "generation_end":
                if request_id in ends:
                    raise ValueError(f"{path}: duplicate generation_end for request {request_id}")
                ends[request_id] = event
            elif kind == "cache_release":
                releases.setdefault(request_id, []).append(event)
            elif kind == "model_loaded":
                model_loads.append({"model": model, "file": str(path), "event": event})
        for request_id in sorted(set(starts) | set(ends) | set(releases)):
            start, end = starts.get(request_id, {}), ends.get(request_id, {})
            before, after = start.get("memory") or {}, end.get("memory") or {}
            peak = _gib(after, "peak_allocated_bytes")
            start_allocated = _gib(before, "allocated_bytes")
            release_rows = [{"status": item.get("status"),
                             "reserved_before_gib": _gib(item.get("memory_before") or {}, "reserved_bytes"),
                             "reserved_after_gib": _gib(item.get("memory_after") or {}, "reserved_bytes")}
                            for item in releases.get(request_id, [])]
            records.append({
                "model": model, "file": str(path), "request_id": request_id,
                "previous_request_id": start.get("previous_request_id"),
                "input_tokens": end.get("input_tokens"),
                "max_output_tokens": end.get("max_output_tokens", start.get("max_output_tokens")),
                "status": end.get("status", "missing_generation_end"),
                "error_type": end.get("error_type"),
                "oom_peak_is_lower_bound": end.get("error_type") == "OutOfMemoryError",
                "peak_allocated_gib": peak,
                "peak_minus_start_allocated_gib": (
                    round(peak - start_allocated, 4)
                    if peak is not None and start_allocated is not None else None),
                "start_allocated_gib": start_allocated,
                "end_allocated_gib": _gib(after, "allocated_bytes"),
                "start_reserved_gib": _gib(before, "reserved_bytes"),
                "end_reserved_gib": _gib(after, "reserved_bytes"),
                "start_global_free_gib": _gib(before, "global_free_bytes"),
                "end_global_free_gib": _gib(after, "global_free_bytes"),
                "cache_release": release_rows,
            })
    return {"schema_version": 1, "telemetry_files": sorted({item["file"] for item in records} |
            {item["file"] for item in model_loads}), "model_loads": model_loads,
            "generations": records, "summary": summarize(records)}


def summarize(records: list[dict]) -> dict:
    result = {}
    for model in ("gpt_oss", "qwen"):
        rows = [item for item in records if item["model"] == model]
        successful = [item for item in rows if item["status"] == "ok" and
                      type(item["input_tokens"]) is int and
                      item["peak_minus_start_allocated_gib"] is not None]
        oom = [item for item in rows if item["error_type"] == "OutOfMemoryError"]
        bins = []
        for lower in sorted({(item["input_tokens"] // 1000) * 1000 for item in successful}):
            group = [item for item in successful if lower <= item["input_tokens"] < lower + 1000]
            bins.append({"input_tokens_from": lower, "input_tokens_below": lower + 1000,
                         "count": len(group),
                         "median_peak_minus_start_allocated_gib": round(statistics.median(
                             item["peak_minus_start_allocated_gib"] for item in group), 4)})
        # This is a descriptive fit among *successful* generations only. OOM
        # peaks are censored and must never be used as completed-demand points.
        x = [(item["input_tokens"] / 1000) ** 2 for item in successful]
        y = [item["peak_minus_start_allocated_gib"] for item in successful]
        denominator = sum((value - statistics.mean(x)) ** 2 for value in x) if len(x) >= 2 else 0
        slope = (sum((a - statistics.mean(x)) * (b - statistics.mean(y))
                     for a, b in zip(x, y)) / denominator if denominator else None)
        result[model] = {"generation_count": len(rows), "successful_count": len(successful),
                         "oom_count": len(oom),
                         "oom_requests": [{"file": item["file"], "request_id": item["request_id"],
                                           "input_tokens": item["input_tokens"],
                                           "observed_peak_lower_bound_gib": item["peak_allocated_gib"]}
                                          for item in oom],
                         "successful_context_bins": bins,
                         "successful_peak_excess_slope_gib_per_million_tokens_squared": (
                             round(slope, 4) if slope is not None else None),
                         "fit_note": "Descriptive successful generations only; failed OOM peaks are censored."}
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Telemetry JSONL files or directories")
    args = parser.parse_args(argv)
    print(json.dumps(analyze(args.paths), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
