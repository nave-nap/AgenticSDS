"""Opt-in, low-frequency process and GPU samples during controller inference."""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from .gpt_oss_load_diagnostics import _gpu_snapshot, _process_snapshot, _torch_gpu_memory

ENABLE_ENV = "AGENT_SUPPORT_GPTOSS_INFERENCE_SAMPLING"


class InferenceSampler:
    def __init__(self, directory: str | Path, torch_api, interval_seconds: float = 15):
        if interval_seconds <= 0:
            raise ValueError("Sampling interval must be positive")
        self.directory = Path(directory)
        self.torch = torch_api
        self.interval_seconds = interval_seconds
        self.stop = threading.Event()

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.path = self.directory / f"gpt_oss_inference_{stamp}_{os.getpid()}.jsonl"
        self.path.touch(exist_ok=False)
        self.started = time.monotonic()
        self.previous = None
        self.previous_at = None
        self._sample("start")
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def _sample(self, event: str):
        now = time.monotonic()
        process = _process_snapshot()
        elapsed = now - self.previous_at if self.previous_at is not None else None
        cpu_percent = None
        io_delta = None
        if elapsed and self.previous is not None:
            cpu_seconds = (process["cpu_user_seconds"] + process["cpu_system_seconds"] -
                           self.previous["cpu_user_seconds"] - self.previous["cpu_system_seconds"])
            cpu_percent = round(100 * cpu_seconds / elapsed, 2)
            if "error" not in process["io"] and "error" not in self.previous["io"]:
                io_delta = {key: value - self.previous["io"].get(key, 0)
                            for key, value in process["io"].items()}
        row = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "event": event,
               "elapsed_seconds": round(now - self.started, 3),
               "cpu_percent_one_core": cpu_percent, "io_delta": io_delta,
               "process": process, "gpus": _gpu_snapshot(),
               "process_cuda_memory": _torch_gpu_memory(self.torch)}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        self.previous, self.previous_at = process, now

    def _loop(self):
        while not self.stop.wait(self.interval_seconds):
            try:
                self._sample("sample")
            except Exception as exc:
                with self.path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"event": "sampler_error", "error": repr(exc)}) + "\n")

    def __exit__(self, _type, _error, _traceback):
        self.stop.set()
        self.thread.join(timeout=5)
        try:
            self._sample("end")
        except Exception:
            pass  # A diagnostic write must not replace the inference result.
        return False
