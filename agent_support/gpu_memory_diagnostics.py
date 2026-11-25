"""Opt-in, best-effort CUDA memory observations outside evaluation results."""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path


ENABLE_ENV = "AGENT_SUPPORT_GPU_MEMORY_DIAGNOSTICS"
DIRECTORY_ENV = "AGENT_SUPPORT_GPU_MEMORY_DIAGNOSTICS_DIR"
CACHE_RELEASE_ENV = "AGENT_SUPPORT_GPTOSS_CUDA_CACHE_RELEASE"


def cache_release_enabled() -> bool:
    return os.environ.get(CACHE_RELEASE_ENV) == "1"


def enabled() -> bool:
    return os.environ.get(ENABLE_ENV) == "1" or cache_release_enabled()


class GpuMemoryDiagnostics:
    def __init__(self, directory: str | Path, label: str, torch_api):
        self.torch = torch_api
        self.previous_request_id = None
        self.path = None
        try:
            location = Path(directory)
            location.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            self.path = location / f"{label}_{stamp}_{os.getpid()}.jsonl"
            with self.path.open("x", encoding="utf-8"):
                pass
        except Exception:
            self.path = None

    def snapshot(self) -> dict:
        try:
            cuda = self.torch.cuda
            free, total = cuda.mem_get_info(0)
            result = {"allocated_bytes": int(cuda.memory_allocated(0)),
                      "reserved_bytes": int(cuda.memory_reserved(0)),
                      "peak_allocated_bytes": int(cuda.max_memory_allocated(0)),
                      "peak_reserved_bytes": int(cuda.max_memory_reserved(0)),
                      "global_free_bytes": int(free), "global_total_bytes": int(total),
                      "inactive_split_bytes": None}
            try:
                result["inactive_split_bytes"] = int(
                    cuda.memory_stats(0)["inactive_split_bytes.all.current"])
            except Exception:
                pass
            return result
        except Exception as exc:
            return {"measurement_error": repr(exc)}

    def write(self, **values) -> None:
        if self.path is None:
            return
        try:
            event = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                     "pid": os.getpid(), **values}
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def start(self, request_id: int, max_output_tokens: int) -> dict:
        before = self.snapshot()
        reset = False
        try:
            self.torch.cuda.reset_peak_memory_stats(0)
            reset = True
        except Exception:
            pass
        self.write(event="generation_start", request_id=request_id,
                   previous_request_id=self.previous_request_id,
                   max_output_tokens=max_output_tokens, memory=before,
                   peak_reset=reset)
        return before

    def finish(self, request_id: int, input_tokens: int | None,
               max_output_tokens: int, error_type: str | None,
               error_detail: str | None = None) -> None:
        self.write(event="generation_end", request_id=request_id,
                   input_tokens=input_tokens, max_output_tokens=max_output_tokens,
                   status="error" if error_type else "ok", error_type=error_type,
                   error_detail=error_detail, memory=self.snapshot())
        self.previous_request_id = request_id


class TwoGpuMemoryDiagnostics(GpuMemoryDiagnostics):
    """Record both workers' CUDA devices and sample free memory during generation."""

    def snapshot(self) -> dict:
        primary = super().snapshot()
        devices = {"cuda:0": primary}
        try:
            cuda = self.torch.cuda
            free, total = cuda.mem_get_info(1)
            devices["cuda:1"] = {
                "allocated_bytes": int(cuda.memory_allocated(1)),
                "reserved_bytes": int(cuda.memory_reserved(1)),
                "peak_allocated_bytes": int(cuda.max_memory_allocated(1)),
                "peak_reserved_bytes": int(cuda.max_memory_reserved(1)),
                "global_free_bytes": int(free), "global_total_bytes": int(total),
                "inactive_split_bytes": int(cuda.memory_stats(1).get("inactive_split_bytes.all.current", 0))}
        except Exception as exc:
            devices["cuda:1"] = {"measurement_error": repr(exc)}
        return {**primary, "devices": devices}

    def _sample(self) -> None:
        while not self._stop.wait(0.02):
            for index in (0, 1):
                try:
                    free = int(self.torch.cuda.mem_get_info(index)[0])
                    key = f"cuda:{index}"
                    self._minimum_free[key] = min(self._minimum_free[key], free)
                    self._samples[key] += 1
                except Exception as exc:
                    self._sampling_errors.append(repr(exc))
                    return

    def start(self, request_id: int, max_output_tokens: int) -> dict:
        before = self.snapshot()
        for index in (0, 1):
            self.torch.cuda.reset_peak_memory_stats(index)
        self._minimum_free = {key: value["global_free_bytes"]
                              for key, value in before["devices"].items()}
        self._samples = {"cuda:0": 0, "cuda:1": 0}
        self._sampling_errors = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        self.write(event="generation_start", request_id=request_id,
                   previous_request_id=self.previous_request_id,
                   max_output_tokens=max_output_tokens, memory=before, peak_reset=True)
        return before

    def finish(self, request_id: int, input_tokens: int | None,
               max_output_tokens: int, error_type: str | None,
               error_detail: str | None = None) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        after = self.snapshot()
        for key, value in after["devices"].items():
            if "global_free_bytes" in value:
                self._minimum_free[key] = min(self._minimum_free[key], value["global_free_bytes"])
        self.write(event="generation_end", request_id=request_id,
                   input_tokens=input_tokens, max_output_tokens=max_output_tokens,
                   status="error" if error_type else "ok", error_type=error_type,
                   error_detail=error_detail, memory=after,
                   minimum_global_free_bytes=self._minimum_free,
                   free_memory_samples=self._samples, sampling_errors=self._sampling_errors)
        self.previous_request_id = request_id
