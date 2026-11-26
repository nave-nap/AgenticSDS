"""One persistent controller model shared by decisions and model supports."""

from __future__ import annotations

import json
import os
import re
import select
import subprocess
from pathlib import Path

from .gpu_memory_diagnostics import DIRECTORY_ENV, enabled
from .gpt_oss_two_gpu import VISIBLE_ENV, enabled as two_gpu_enabled

STARTUP_TIMEOUT_ENV = "AGENT_SUPPORT_GPTOSS_STARTUP_TIMEOUT_SECONDS"


class ControllerWorkerOOM(RuntimeError):
    """A worker-reported CUDA out-of-memory failure."""

    failure_kind = "cuda_oom"
    termination_reason = "controller_cuda_oom"

    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(f"GPT-OSS controller worker OutOfMemoryError: {detail}")


class ControllerWorkerUnavailable(RuntimeError):
    """A definitively closed or exited controller worker."""

    failure_kind = "unavailable"
    termination_reason = "controller_worker_unavailable"

    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(f"GPT-OSS controller worker unavailable: {detail}")


def worker_failure_marker(exc: ControllerWorkerOOM | ControllerWorkerUnavailable) -> dict:
    return {"kind": exc.failure_kind, "detail": exc.detail}


class GptOssControllerWorker:
    def __init__(self, spec: dict, log_dir: str | Path):
        variable = spec["python_env"]
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", variable):
            raise ValueError("Invalid GPT-OSS Python environment variable name")
        python = os.environ.get(variable)
        if not python or not Path(python).is_file():
            raise ValueError(f"Set {variable} to the pinned GPT-OSS Python executable")
        environment = os.environ.copy()
        if two_gpu_enabled():
            visible = os.environ.get(VISIBLE_ENV, "")
            devices = [item.strip() for item in visible.split(",")]
            if len(devices) != 2 or any(not item for item in devices) or devices[0] == devices[1]:
                raise ValueError(f"Set {VISIBLE_ENV} to two distinct allocated GPU IDs")
            environment["CUDA_VISIBLE_DEVICES"] = visible
        hf_variable = spec.get("hf_home_env")
        if hf_variable:
            if not re.fullmatch(r"[A-Z][A-Z0-9_]*", hf_variable) or not os.environ.get(hf_variable):
                raise ValueError("GPT-OSS cache environment variable is missing or invalid")
            environment["HF_HOME"] = os.environ[hf_variable]
        root = str(Path(__file__).resolve().parent.parent)
        environment["PYTHONPATH"] = root + os.pathsep + environment.get("PYTHONPATH", "")
        if enabled():
            environment[DIRECTORY_ENV] = str(Path(log_dir).parent / "diagnostics")
        elif two_gpu_enabled():
            raise ValueError("Two-GPU GPT-OSS requires AGENT_SUPPORT_GPU_MEMORY_DIAGNOSTICS=1")
        self.timeout = int(spec.get("timeout_seconds", 1800))
        self.startup_timeout = int(os.environ.get(STARTUP_TIMEOUT_ENV, self.timeout))
        if self.startup_timeout <= 0:
            raise ValueError("GPT-OSS startup timeout must be positive")
        log_path = Path(log_dir) / "gpt_oss_controller_worker.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = log_path.open("a", encoding="utf-8")
        self.process = subprocess.Popen(
            [python, "-u", "-m", "agent_support.gpt_oss_controller_worker"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            text=True, bufsize=1, env=environment)
        self.max_new_tokens = spec["model"]["max_new_tokens"]
        self.closed = False
        try:
            self.model_metadata = self._request({"op": "init", "config": spec["model"]})["metadata"]
            if (self.model_metadata.get("model_type") != "gpt_oss_20b" or
                    self.model_metadata.get("model_id") != spec["model"]["model_id"] or
                    self.model_metadata.get("model_revision") != spec["model"]["model_revision"]):
                raise ValueError("Controller worker model identity or revision differs")
            if two_gpu_enabled() and (self.model_metadata.get("sharding") != "explicit_decoder_layers_12_12" or
                                      self.model_metadata.get("verified_expert_modules_by_device") !=
                                      {"cuda:0": 12, "cuda:1": 12}):
                raise RuntimeError("GPT-OSS did not load packed experts on both requested GPUs")
        except BaseException:
            self.close()
            raise

    def _request(self, payload: dict) -> dict:
        if self.closed:
            raise ControllerWorkerUnavailable("worker is closed")
        if getattr(self, "unusable", False):
            raise ControllerWorkerUnavailable("worker was terminated")
        exit_code = self.process.poll()
        if exit_code is not None:
            raise ControllerWorkerUnavailable(f"worker exited with code {exit_code}")
        stdin, stdout = self.process.stdin, self.process.stdout
        if (stdin is None or stdout is None or stdin.closed or stdout.closed):
            raise ControllerWorkerUnavailable("protocol stream is closed")
        try:
            stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            if isinstance(exc, BrokenPipeError) or stdin.closed or self.process.poll() is not None:
                raise ControllerWorkerUnavailable(f"protocol write failed: {exc}") from exc
            raise
        try:
            timeout = self.startup_timeout if payload["op"] == "init" else self.timeout
            ready, _, _ = select.select([stdout], [], [], timeout)
        except (OSError, ValueError) as exc:
            if stdout.closed or self.process.poll() is not None:
                raise ControllerWorkerUnavailable(f"protocol stream closed: {exc}") from exc
            raise
        if not ready:
            self.process.terminate()
            self.unusable = True
            detail = (f"worker startup timed out after {timeout} seconds and was terminated"
                      if payload["op"] == "init" else "worker timed out and was terminated")
            raise ControllerWorkerUnavailable(detail)
        try:
            line = stdout.readline()
        except (OSError, ValueError) as exc:
            if stdout.closed or self.process.poll() is not None:
                raise ControllerWorkerUnavailable(f"protocol read failed: {exc}") from exc
            raise
        if not line:
            raise ControllerWorkerUnavailable("worker closed its protocol stream")
        result = json.loads(line)
        if not result.get("ok"):
            if result.get("error_type") == "OutOfMemoryError":
                raise ControllerWorkerOOM(result.get("error"))
            raise RuntimeError(f"GPT-OSS controller worker {result.get('error_type')}: {result.get('error')}")
        return result

    def metadata(self) -> dict:
        return self.model_metadata

    def count_messages_tokens(self, messages: list[dict]) -> int:
        return self._request({"op": "count_messages_tokens", "messages": messages})["count"]

    def generate(self, messages: list[dict], max_new_tokens: int | None = None) -> dict:
        return self._request({"op": "generate", "messages": messages,
                              "max_new_tokens": max_new_tokens})["generation"]

    def begin_episode(self, task_id: str, seed: int) -> None:
        import hashlib
        material = f"{seed}\0{task_id}\0gpt_oss_controller".encode()
        derived = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
        self._request({"op": "seed", "seed": derived})

    def close(self) -> None:
        if self.closed:
            return
        try:
            if self.process.poll() is None:
                try:
                    self._request({"op": "shutdown"})
                except Exception:
                    self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        finally:
            self.closed = True
            self.log.close()


def make_controller_model(config: dict, task_model, log_dir: str | Path):
    spec = config["autonomous_support"].get("model")
    if not isinstance(spec, dict):
        raise ValueError("Model-driven controller requires autonomous_support.model")
    if spec.get("type") == "reuse_task_model":
        if not hasattr(task_model, "count_messages_tokens") or not hasattr(task_model, "generate"):
            raise ValueError("Task model lacks counted inference for controller reuse")
        return task_model
    if spec.get("type") == "gpt_oss_20b_worker_v1":
        return GptOssControllerWorker(spec, log_dir)
    raise ValueError("Unknown controller model backend")
