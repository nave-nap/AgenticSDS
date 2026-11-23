"""Offline checks for the bounded archived-request diagnostic."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from agent_support.common import canonical_digest
from agent_support.controller_backend import ControllerWorkerFatalCUDA
from agent_support.recovery_v4_eval import config_for
from reports.flex89_bounded_diagnostic import replay, tokenize_plan


PLAN = Path(__file__).resolve().parents[1] / "reports/flex89_bounded_diagnostic/archived_replay_plan.json"


def test_archived_plan_has_exact_bounded_order_and_message_hashes():
    plan = json.loads(PLAN.read_text())
    requests = replay.validate_plan(plan, config_for("autonomous_recovery_v4"))
    assert [(request["request_id"], request["decision"]) for request in requests] == [
        (1, 1), (2, 1), (3, 2), (4, 3), (5, 4), (6, 5), (7, 6), (8, 7),
        (9, 1), (10, 1), (11, 2), (12, 2), (13, 2), (14, 3), (15, 3),
        (16, 3), (17, 4), (18, 5), (19, 5), (20, 5), (21, 6), (22, 6)]
    assert all(canonical_digest(item["messages"]) == item["messages_sha256"]
               for item in requests)
    assert requests[-1]["expected_input_tokens"] == 3724


def test_pinned_tokenizer_manifest_checks_every_archived_count(monkeypatch):
    plan = json.loads(PLAN.read_text())
    config = config_for("autonomous_recovery_v4")
    requests = replay.validate_plan(plan, config)
    by_message = {canonical_digest(item["messages"]): item for item in requests}
    monkeypatch.setattr(tokenize_plan, "_pinned_tokenizer",
                        lambda _config: (object(), "pinned-template"))
    monkeypatch.setattr(tokenize_plan, "_token_ids", lambda _tokenizer, _template, _config, messages:
                        list(range(by_message[canonical_digest(messages)]["expected_input_tokens"])))
    manifest = tokenize_plan.tokenize(plan, config)
    manifest["chat_template_sha256"] = plan["worker_metadata"]["chat_template_sha256"]
    replay.validate_token_manifest(manifest, plan, requests)
    assert len(manifest["rows"]) == 22
    assert manifest["rows"][-1]["input_tokens"] == 3724
    monkeypatch.setattr(tokenize_plan, "_token_ids",
                        lambda *_: [1, 2, 3])
    import pytest
    with pytest.raises(ValueError, match="Pinned tokenizer count differs at request 1"):
        tokenize_plan.tokenize(plan, config)


def test_replay_stops_at_first_fatal_and_isolated_mode_uses_fresh_worker(
        monkeypatch, tmp_path):
    plan = json.loads(PLAN.read_text())
    config = config_for("autonomous_recovery_v4")
    requests = replay.validate_plan(plan, config)
    by_message = {canonical_digest(request["messages"]): request for request in requests}
    token_manifest = {"schema_version": 1,
                      "model_revision": plan["worker_metadata"]["model_revision"],
                      "chat_template_sha256": plan["worker_metadata"]["chat_template_sha256"],
                      "rows": [{"request_id": request["request_id"],
                                "input_tokens": request["expected_input_tokens"],
                                "messages_sha256": request["messages_sha256"],
                                "input_ids_sha256": "a" * 64} for request in requests]}
    calls = []

    class Worker:
        def __init__(self, _spec, _directory):
            self.generated = 0
            calls.append(("worker_loaded", self))
        def metadata(self):
            return {**plan["worker_metadata"], "attention_implementation": "flex_attention"}
        def begin_episode(self, task_id, _seed):
            calls.append(("episode", task_id))
        def probe_input(self, messages):
            request = by_message[canonical_digest(messages)]
            return {"attention_implementation": "flex_attention",
                    "input_tokens": request["expected_input_tokens"],
                    "input_ids_sha256": "a" * 64}
        def generate(self, messages, max_new_tokens=None):
            request = by_message[canonical_digest(messages)]
            self.generated += 1
            calls.append(("generate", request["request_id"]))
            assert max_new_tokens == 1024
            if request["request_id"] == 22:
                raise ControllerWorkerFatalCUDA(
                    "AcceleratorError", "CUDA error: an illegal memory access was encountered",
                    "GPT-OSS request traceback", self.generated, "model_generate")
            return {"generation_status": "complete", "input_tokens": request["expected_input_tokens"],
                    "output_tokens": 10, "inference_seconds": 0.01}
        def close(self):
            calls.append(("closed", self.generated))

    class Qwen:
        def __init__(self, _config):
            self.model = SimpleNamespace(hf_device_map={"": 0})
            self.torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 1))
        def enable_gpu_memory_diagnostics(self, _directory):
            pass
        def generate(self, _messages):
            calls.append(("qwen", None))

    for name in ("CUDA_LAUNCH_BLOCKING", "AGENT_SUPPORT_GPTOSS_WORKER_TRACEBACKS",
                 "AGENT_SUPPORT_GPU_MEMORY_DIAGNOSTICS", "AGENT_SUPPORT_GPTOSS_PHASE_TRACE"):
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("AGENT_SUPPORT_GPTOSS_CUDA_CACHE_RELEASE", "0")
    monkeypatch.setattr(replay.importlib.metadata, "version",
                        lambda package: plan["evaluator_package_versions"][package])

    sequential = replay.run(plan, config, "flex_attention", tmp_path / "sequence",
                            token_manifest=token_manifest, worker_factory=Worker, qwen_factory=Qwen)
    assert sequential["status"] == "stopped_on_error"
    assert sequential["successful_generations"] == 21
    assert [row["request_id"] for row in sequential["generations"]] == list(range(1, 23))
    assert sequential["first_failure"]["failed_phase"] == "model_generate"
    assert sequential["first_failure"]["reported_worker_request_id"] == 22
    assert ("closed", 22) in calls
    assert sum(event[0] == "qwen" for event in calls) == 13

    calls.clear()
    isolated = replay.run(plan, config, "flex_attention", tmp_path / "isolated",
                          token_manifest=token_manifest, mode="isolated22",
                          worker_factory=Worker, qwen_factory=Qwen)
    assert isolated["status"] == "stopped_on_error"
    assert [row["request_id"] for row in isolated["generations"]] == [22]
    assert isolated["first_failure"]["reported_worker_request_id"] == 1
    assert calls.count(("qwen", None)) == 1
    assert ("closed", 1) in calls
