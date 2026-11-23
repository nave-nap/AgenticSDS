"""CPU-only replay guards; no CUDA imports or benchmark episodes."""
from __future__ import annotations

import json

import pytest

from agent_support.common import canonical_digest
from agent_support.controller_backend import ControllerWorkerFatalCUDA
from agent_support.recovery_v4_eval import config_for
from reports.gptoss_seed51_cuda import audit, replay


def synthetic_plan():
    config = config_for("autonomous_recovery_v4")
    rows = []
    for index in range(1, 11):
        messages = [{"role": "user", "content": str(index)}]
        rows.append({"request_id": index, "decision": 1 if index <= 3 else 2,
                     "event_index": index - 1 if index <= 3 else index - 4,
                     "messages": messages, "messages_sha256": canonical_digest(messages),
                     "input_tokens": 5451 if index == 10 else 100,
                     "output_tokens": None if index == 10 else 4,
                     "max_new_tokens": 1024,
                     "status": "error" if index == 10 else "complete"})
    return {"schema_version": 1, "source_run_commit": audit.SOURCE_COMMIT,
            "source_run_sha256": audit.SOURCE_RUN_SHA256,
            "source_episode_sha256": audit.SOURCE_EPISODE_SHA256,
            "source_worker_log_sha256": audit.SOURCE_WORKER_LOG_SHA256,
            "source_policy": audit.SOURCE_POLICY, "source_task": audit.SOURCE_TASK,
            "config_sha256": canonical_digest(config), "seed": config["seed"],
            "total_controller_requests": 10,
            "evaluator_package_versions": {"textcraft": "0.0.3"},
            "worker_metadata": {"gpu_name": "NVIDIA L40S",
                                "gpu_total_memory_bytes": 47676129280,
                                "attention_implementation": "flex_attention"},
            "decisions": [{"decision": decision,
                           "qwen_messages": [{"role": "user", "content": "task"}],
                           "controller_requests": rows[:3] if decision == 1 else rows[3:]}
                          for decision in (1, 2)]}


def test_plan_accepts_ten_hashed_inputs_and_rejects_mutation():
    plan = synthetic_plan()
    assert len(replay.validate_plan(plan)) == 10
    plan["decisions"][1]["controller_requests"][1]["messages"][0]["content"] = "changed"
    with pytest.raises(ValueError, match="changed"):
        replay.validate_plan(plan)


def test_worker_accepts_l40s_memory_variation_but_checks_backend_and_identity():
    archived = {"model_id": "openai/gpt-oss-20b", "model_revision": "fixed",
        "backend_version": "5.22.2", "torch_version": "2.8.0+cu128",
        "torch_cuda_version": "12.8", "triton_version": "3.4.0",
        "kernels_version": "0.17.2", "quantization": "mxfp4_packed",
        "quantization_method": "mxfp4", "decoding": {"do_sample": True,
            "temperature": 1.0, "top_p": 1.0, "top_k": 0, "max_new_tokens": 1024},
        "reasoning_effort": "low", "prompt_date": "2026-11-18",
        "chat_template_sha256": "template", "gpu_name": "NVIDIA L40S",
        "gpu_compute_capability": [8, 9],
        "gpu_total_memory_bytes": 47676129280,
        "attention_implementation": "flex_attention"}
    eager = {**archived, "gpu_total_memory_bytes": 47673671680,
             "attention_implementation": "eager"}
    replay.verify_worker(eager, archived, "default")
    replay.verify_worker(archived, archived, "flex_attention")
    with pytest.raises(RuntimeError, match="active attention"):
        replay.verify_worker({**eager, "attention_implementation": "sdpa"}, archived,
                             "default")
    with pytest.raises(RuntimeError, match="model_revision"):
        replay.verify_worker({**eager, "model_revision": "different"}, archived,
                             "default")
    with pytest.raises(RuntimeError, match="torch_version"):
        replay.verify_worker({**eager, "torch_version": "different"}, archived,
                             "default")
    with pytest.raises(RuntimeError, match="decoding"):
        replay.verify_worker({**eager, "decoding": {"temperature": 0.5}}, archived,
                             "default")
    with pytest.raises(RuntimeError, match="gpu_compute_capability"):
        replay.verify_worker({**eager, "gpu_compute_capability": [7, 5]}, archived,
                             "default")


def test_replay_stops_on_first_fatal_without_retry(monkeypatch, tmp_path):
    for name in ("CUDA_LAUNCH_BLOCKING", "AGENT_SUPPORT_GPTOSS_WORKER_TRACEBACKS",
                 "AGENT_SUPPORT_GPU_MEMORY_DIAGNOSTICS"):
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("AGENT_SUPPORT_GPTOSS_CUDA_CACHE_RELEASE", "0")
    plan = synthetic_plan()
    workers = []

    class Worker:
        def __init__(self, *_args):
            self.generated = 0
            self.closed = False
            workers.append(self)

        def metadata(self):
            return {**plan["worker_metadata"],
                    "gpu_total_memory_bytes": 47673671680}

        def begin_episode(self, task, seed):
            assert task == audit.SOURCE_TASK and seed == plan["seed"]

        def count_messages_tokens(self, messages):
            return 5451 if messages[0]["content"] == "10" else 100

        def generate(self, messages, max_new_tokens=None):
            self.generated += 1
            if self.generated == 10:
                raise ControllerWorkerFatalCUDA("AcceleratorError",
                    "CUDA error: an illegal memory access was encountered",
                    worker_traceback="GPT-OSS request 10\nTraceback", request_id=10,
                    failed_phase="model_generate")
            return {"generation_status": "complete", "output_tokens": 4}

        def close(self):
            self.closed = True

    class Qwen:
        def __init__(self, _config):
            self.model = type("Model", (), {"hf_device_map": {"": "0"}})()

        def generate(self, _messages):
            return {}

    report = replay.run(plan, "flex_attention", tmp_path / "replay",
        worker_factory=Worker, qwen_factory=Qwen,
        version_lookup=lambda name: plan["evaluator_package_versions"][name])
    assert report["status"] == "stopped_on_fatal_cuda"
    assert report["completed_generations"] == 9
    assert report["first_failure"]["request_id"] == 10
    assert report["first_failure"]["output_tokens"] is None
    assert report["first_failure"]["worker_request_id"] == 10
    assert report["gpu_total_memory_bytes"] == {
        "archived": 47676129280, "active": 47673671680}
    assert workers[0].generated == 10 and workers[0].closed
    assert json.loads((tmp_path / "replay" / "replay_result.json").read_text())["status"] == (
        "stopped_on_fatal_cuda")
