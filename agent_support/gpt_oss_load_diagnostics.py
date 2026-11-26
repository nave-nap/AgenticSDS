"""Opt-in, best-effort observations of a pinned GPT-OSS checkpoint load.

Sampling runs outside Transformers' loader. It never wraps tensor operations or
changes checkpoint placement. Progress is read from the worker's existing log.
"""

from __future__ import annotations

import json
import os
import re
import resource
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


ENABLE_ENV = "AGENT_SUPPORT_GPTOSS_MODEL_LOAD_DIAGNOSTICS"
WORKER_LOG_ENV = "AGENT_SUPPORT_GPTOSS_WORKER_LOG_PATH"
SAMPLE_SECONDS = 15
STALL_SECONDS = 120
_PROGRESS = re.compile(r"Loading weights:.*?\b(\d+)/(\d+)\s+\[([0-9:]+)")
_SMI_COMMAND = ["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,utilization.gpu,utilization.memory",
                "--format=csv,noheader,nounits"]


def enabled() -> bool:
    return os.environ.get(ENABLE_ENV) == "1"


def _elapsed_seconds(display: str) -> int:
    parts = [int(item) for item in display.split(":")]
    seconds = 0
    for item in parts:
        seconds = seconds * 60 + item
    return seconds


def _proc_io() -> dict[str, int]:
    try:
        return {key: int(value) for line in Path("/proc/self/io").read_text().splitlines()
                for key, value in [line.split(":", 1)]}
    except OSError as exc:
        return {"error": repr(exc)}


def _rss_bytes() -> int | None:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def _process_snapshot() -> dict:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {"io": _proc_io(), "cpu_user_seconds": usage.ru_utime,
            "cpu_system_seconds": usage.ru_stime, "rss_bytes": _rss_bytes(),
            "minor_faults": usage.ru_minflt, "major_faults": usage.ru_majflt}


def _gpu_snapshot() -> dict:
    try:
        result = subprocess.run(_SMI_COMMAND, capture_output=True, text=True,
                                timeout=3, check=True)
        rows = []
        for line in result.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 6:
                raise ValueError(f"Unexpected nvidia-smi row: {line!r}")
            index, uuid, name, memory, gpu_util, memory_util = fields
            def number(value):
                return int(value) if value.isdecimal() else None
            rows.append({"physical_index": number(index), "uuid": uuid, "name": name,
                         "memory_used_mib": number(memory),
                         "gpu_utilization_percent": number(gpu_util),
                         "memory_utilization_percent": number(memory_util)})
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        tokens = [item.strip() for item in visible.split(",")]
        selected = []
        for token in tokens:
            matches = [row for row in rows if token == row["uuid"] or
                       token == str(row["physical_index"])]
            selected.append(matches[0] if len(matches) == 1 else {"requested": token,
                                                                   "mapping_error": len(matches)})
        return {"cuda_visible_devices": visible, "worker_cuda_ordinals":
                {f"cuda:{index}": row for index, row in enumerate(selected)},
                "physical_gpus": rows}
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return {"measurement_error": repr(exc)}


def _torch_gpu_memory(torch_api) -> dict:
    if torch_api is None:
        return {}
    try:
        cuda = torch_api.cuda
        return {f"cuda:{index}": {"allocated_bytes": int(cuda.memory_allocated(index)),
                                    "reserved_bytes": int(cuda.memory_reserved(index))}
                for index in range(min(cuda.device_count(), 2))}
    except Exception as exc:
        return {"measurement_error": repr(exc)}


class ModelLoadDiagnostics:
    def __init__(self, directory: str | Path, worker_log: str | Path | None,
                 sample_seconds: int = SAMPLE_SECONDS, stall_seconds: int = STALL_SECONDS,
                 torch_api=None):
        if sample_seconds <= 0 or stall_seconds <= 0:
            raise ValueError("Model-load sampling intervals must be positive")
        location = Path(directory)
        location.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.path = location / f"gpt_oss_model_load_{stamp}_{os.getpid()}.jsonl"
        self.path.touch(exist_ok=False)
        self.worker_log = Path(worker_log) if worker_log else None
        self.torch = torch_api
        self.sample_seconds = sample_seconds
        self.stall_seconds = stall_seconds
        self.stop_event = threading.Event()
        self.write_lock = threading.Lock()
        self.log_offset = 0
        self.log_tail = ""
        self.last_progress = 0
        self.last_loader_elapsed = 0
        self.last_progress_at = 0.0
        self.last_stall_at = 0.0
        self.previous_snapshot = None
        self.previous_sample_at = None

    def _write(self, event: str, **values) -> None:
        record = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                  "pid": os.getpid(), "event": event, **values}
        with self.write_lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _progress(self, now: float) -> None:
        if self.worker_log is None:
            return
        try:
            with self.worker_log.open("rb") as stream:
                stream.seek(self.log_offset)
                chunk = stream.read(64 * 1024)
                self.log_offset = stream.tell()
            if not chunk:
                return
            data = self.log_tail + chunk.decode("utf-8", errors="replace")
            lines = re.split(r"[\r\n]", data)
            self.log_tail = lines[-1][-512:]
            for line in lines:
                match = _PROGRESS.search(line)
                if match is None:
                    continue
                completed, total = map(int, match.group(1, 2))
                if completed <= self.last_progress:
                    continue
                elapsed = _elapsed_seconds(match.group(3))
                interval = elapsed - self.last_loader_elapsed
                weights = completed - self.last_progress
                self._write("weight_progress", completed=completed, total=total,
                            loader_elapsed_seconds=elapsed,
                            progress_interval_seconds=interval,
                            completed_since_previous=weights,
                            single_weight_interval_seconds=interval if weights == 1 else None)
                self.last_progress = completed
                self.last_loader_elapsed = elapsed
                self.last_progress_at = now
        except OSError as exc:
            self._write("progress_read_error", error=repr(exc))

    def _sample(self, now: float) -> None:
        process = _process_snapshot()
        previous = self.previous_snapshot
        elapsed = (now - self.previous_sample_at if self.previous_sample_at is not None else None)
        cpu_percent = None
        io_delta = None
        if previous is not None and elapsed and elapsed > 0:
            cpu_seconds = (process["cpu_user_seconds"] + process["cpu_system_seconds"] -
                           previous["cpu_user_seconds"] - previous["cpu_system_seconds"])
            cpu_percent = round(100 * cpu_seconds / elapsed, 2)
            if "error" not in process["io"] and "error" not in previous["io"]:
                io_delta = {key: value - previous["io"].get(key, 0)
                            for key, value in process["io"].items()}
        self._write("sample", elapsed_seconds=round(now - self.started_at, 3),
                    sample_interval_seconds=round(elapsed, 3) if elapsed else None,
                    cpu_percent_one_core=cpu_percent, io_delta=io_delta,
                    process=process, gpus=_gpu_snapshot(),
                    process_cuda_memory=_torch_gpu_memory(self.torch))
        self.previous_snapshot = process
        self.previous_sample_at = now

    def _loop(self) -> None:
        while not self.stop_event.wait(self.sample_seconds):
            try:
                now = time.monotonic()
                self._progress(now)
                self._sample(now)
                if now - self.last_progress_at >= self.stall_seconds and now - self.last_stall_at >= self.stall_seconds:
                    self._write("progress_stall", elapsed_seconds=round(now - self.started_at, 3),
                                completed=self.last_progress,
                                seconds_since_progress=round(now - self.last_progress_at, 3))
                    self.last_stall_at = now
            except Exception as exc:
                try:
                    self._write("sampler_error", error=repr(exc))
                except Exception:
                    pass

    def __enter__(self):
        self.started_at = time.monotonic()
        self.last_progress_at = self.started_at
        self.thread = None
        try:
            self._write("model_load_start", elapsed_seconds=0,
                        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                        hf_home=os.environ.get("HF_HOME"), process=_process_snapshot(),
                        gpus=_gpu_snapshot(), process_cuda_memory=_torch_gpu_memory(self.torch))
            self.thread = threading.Thread(target=self._loop, name="gpt-oss-load-diagnostics", daemon=True)
            self.thread.start()
        except Exception as exc:
            print(f"GPT-OSS model-load diagnostics disabled: {exc}", file=sys.stderr)
            self.thread = None
        return self

    def __exit__(self, error_type, error, _traceback):
        if self.thread is None:
            return False
        self.stop_event.set()
        self.thread.join(timeout=5)
        try:
            now = time.monotonic()
            self._progress(now)
            self._sample(now)
            self._write("model_load_end", elapsed_seconds=round(now - self.started_at, 3),
                        status="error" if error_type else "ok",
                        error_type=error_type.__name__ if error_type else None,
                        error_detail=str(error) if error else None,
                        completed=self.last_progress)
        except Exception:
            pass
        return False


def monitor(directory: str | Path | None, torch_api=None):
    """Return a no-op context unless diagnostics were explicitly enabled."""
    from contextlib import nullcontext
    if not enabled() or not directory:
        return nullcontext()
    try:
        return ModelLoadDiagnostics(directory, os.environ.get(WORKER_LOG_ENV), torch_api=torch_api)
    except Exception as exc:
        print(f"GPT-OSS model-load diagnostics disabled: {exc}", file=sys.stderr)
        return nullcontext()
