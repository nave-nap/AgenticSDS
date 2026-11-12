"""Route generation to two persistent models in their original Python environments."""

from __future__ import annotations

import json
import hashlib
import os
import select
import subprocess
from pathlib import Path

from .model import register_model


class WorkerClient:
    def __init__(self, python: str, config: dict, log_path: Path, *, timeout: int = 1800,
                 hf_home: str | None = None):
        self.log = log_path.open("a", encoding="utf-8")
        env = os.environ.copy()
        if hf_home:
            env["HF_HOME"] = hf_home
        self.process = subprocess.Popen(
            [python, "-u", "-m", "agent_support.model_worker"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            text=True, bufsize=1, env=env)
        self.timeout = timeout
        self.broken = False
        try:
            initialized = self.request({"op": "init", "config": config})
            self.metadata = initialized["metadata"]
            self.initial_cuda_memory = initialized["cuda_memory"]
        except BaseException:
            try:
                self.close()
            except Exception:
                pass
            raise

    def request(self, payload: dict) -> dict:
        process = self.process
        if self.broken:
            raise RuntimeError(f"Model worker is unavailable after timeout; see {self.log.name}")
        if process.poll() is not None:
            raise RuntimeError(f"Model worker exited with code {process.returncode}; see {self.log.name}")
        assert process.stdin is not None and process.stdout is not None
        process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        process.stdin.flush()
        ready, _, _ = select.select([process.stdout], [], [], self.timeout)
        if not ready:
            self.broken = True
            raise TimeoutError(f"Model worker timed out after {self.timeout}s; see {self.log.name}")
        line = process.stdout.readline()
        if not line:
            raise RuntimeError(f"Model worker closed its protocol stream; see {self.log.name}")
        response = json.loads(line)
        if not response.get("ok"):
            raise RuntimeError(f"Model worker {response.get('error_type')}: {response.get('error')}; "
                               f"see {self.log.name}")
        return response

    def close(self) -> None:
        process = self.process
        if process.poll() is None and not self.broken:
            try:
                self.request({"op": "shutdown"})
            except Exception:
                pass
        if process.poll() is None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        self.log.close()


class HybridModel:
    primary_model_type = "qwen"
    delegated_model_type = "gpt_oss_20b"

    def __init__(self, config: dict):
        self.config = config
        self.workers = {}
        self.runtime = {}
        self.latest_memory = {}
        try:
            for role, model_type in (("primary", self.primary_model_type),
                                     ("delegated", self.delegated_model_type)):
                spec = config["worker_models"][role]
                python = os.environ[spec["python_env"]]
                hf_home = os.environ[spec["hf_home_env"]]
                log = Path(os.environ["AGENT_SUPPORT_WORKER_LOG_DIR"]) / f"{role}_worker.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                self.workers[model_type] = WorkerClient(python, spec["model"], log,
                                                       hf_home=hf_home)
                self.latest_memory[model_type] = self.workers[model_type].initial_cuda_memory
                metadata = self.workers[model_type].metadata
                if (metadata.get("model_id") != spec["model"]["model_id"] or
                        metadata.get("model_revision") != spec["model"]["model_revision"]):
                    raise ValueError(f"{role} worker model identity or revision differs from configuration")
                self.runtime[role] = {"python": str(Path(python).resolve()),
                                      "hf_home": str(Path(hf_home).resolve()),
                                      "metadata": metadata}
        except BaseException:
            self.close()
            raise

    def metadata(self) -> dict:
        return {"model_type": "hybrid_delegation", "model_id": self.config["model_id"],
                "model_revision": self.config["model_revision"],
                "workers": self.runtime, "delegated_model_type": self.delegated_model_type}

    def generate(self, messages: list[dict]) -> dict:
        return self.generate_for(self.primary_model_type, messages)

    def generate_for(self, model_type: str, messages: list[dict]) -> dict:
        try:
            response = self.workers[model_type].request({"op": "generate", "messages": messages})
            self.latest_memory[model_type] = response["cuda_memory"]
            return {**response["generation"], "worker_cuda_memory": response["cuda_memory"],
                    "combined_worker_cuda_allocated_bytes": sum(
                        item.get("allocated_bytes", 0) for item in self.latest_memory.values()),
                    "combined_worker_cuda_reserved_bytes": sum(
                        item.get("reserved_bytes", 0) for item in self.latest_memory.values())}
        except Exception as exc:
            return {"text": "", "input_tokens": 0, "output_tokens": 0,
                    "inference_seconds": 0.0, "generation_status": "worker_error",
                    "generation_error": str(exc)}

    def count_messages_tokens_for(self, model_type: str, messages: list[dict]) -> int:
        return self.workers[model_type].request({"op": "count_messages_tokens", "messages": messages})["count"]

    def max_new_tokens_for(self, model_type: str) -> int:
        role = "primary" if model_type == self.primary_model_type else "delegated"
        return self.config["worker_models"][role]["model"]["max_new_tokens"]

    def begin_episode(self, task_id: str) -> None:
        for model_type, worker in self.workers.items():
            material = f"{self.config['seed']}\0{task_id}\0{model_type}".encode()
            seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
            worker.request({"op": "seed", "seed": seed})

    def memory_snapshot(self) -> dict:
        return {kind: worker.request({"op": "memory"})["cuda_memory"]
                for kind, worker in self.workers.items()}

    def close(self) -> None:
        errors = []
        for worker in self.workers.values():
            try:
                worker.close()
            except Exception as exc:
                errors.append(exc)
        self.workers.clear()
        if errors:
            raise RuntimeError(f"Failed to shut down {len(errors)} model worker(s)") from errors[0]


register_model("hybrid_delegation", HybridModel)
