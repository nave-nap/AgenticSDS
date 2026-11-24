"""Opt-in, best-effort CUDA memory observations outside evaluation results."""

from __future__ import annotations

import json
import os
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
                     "pid": os.getpid(),
                     "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                     "logical_cuda_device": 0, **values}
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
               *, capture_memory: bool = True) -> None:
        self.write(event="generation_end", request_id=request_id,
                   input_tokens=input_tokens, max_output_tokens=max_output_tokens,
                   status="error" if error_type else "ok", error_type=error_type,
                   memory=self.snapshot() if capture_memory else
                          {"skipped_after_generation_error": True})
        self.previous_request_id = request_id
