"""One persistent controller model shared by decisions and model supports."""

from __future__ import annotations

import json
import os
import re
import select
import subprocess
from pathlib import Path


class GptOssControllerWorker:
    def __init__(self, spec: dict, log_dir: str | Path):
        variable = spec["python_env"]
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", variable):
            raise ValueError("Invalid GPT-OSS Python environment variable name")
        python = os.environ.get(variable)
        if not python or not Path(python).is_file():
            raise ValueError(f"Set {variable} to the pinned GPT-OSS Python executable")
        environment = os.environ.copy()
        hf_variable = spec.get("hf_home_env")
        if hf_variable:
            if not re.fullmatch(r"[A-Z][A-Z0-9_]*", hf_variable) or not os.environ.get(hf_variable):
                raise ValueError("GPT-OSS cache environment variable is missing or invalid")
            environment["HF_HOME"] = os.environ[hf_variable]
        root = str(Path(__file__).resolve().parent.parent)
        environment["PYTHONPATH"] = root + os.pathsep + environment.get("PYTHONPATH", "")
        log_path = Path(log_dir) / "gpt_oss_controller_worker.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = log_path.open("a", encoding="utf-8")
        self.process = subprocess.Popen(
            [python, "-u", "-m", "agent_support.gpt_oss_controller_worker"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            text=True, bufsize=1, env=environment)
        self.timeout = int(spec.get("timeout_seconds", 1800))
        self.max_new_tokens = spec["model"]["max_new_tokens"]
        self.closed = False
        try:
            self.model_metadata = self._request({"op": "init", "config": spec["model"]})["metadata"]
            if (self.model_metadata.get("model_type") != "gpt_oss_20b" or
                    self.model_metadata.get("model_id") != spec["model"]["model_id"] or
                    self.model_metadata.get("model_revision") != spec["model"]["model_revision"]):
                raise ValueError("Controller worker model identity or revision differs")
        except BaseException:
            self.close()
            raise

    def _request(self, payload: dict) -> dict:
        if self.closed or self.process.poll() is not None:
            raise RuntimeError("GPT-OSS controller worker is unavailable")
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        ready, _, _ = select.select([self.process.stdout], [], [], self.timeout)
        if not ready:
            self.process.terminate()
            raise TimeoutError("GPT-OSS controller worker timed out")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("GPT-OSS controller worker closed its protocol stream")
        result = json.loads(line)
        if not result.get("ok"):
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
