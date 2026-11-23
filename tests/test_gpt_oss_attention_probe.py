"""CPU checks for isolated attention inference and exact-input replay."""

from __future__ import annotations

import io
import json
import hashlib
import os
from collections import UserDict
from types import SimpleNamespace

import pytest

from agent_support import gpt_oss_attention_probe as probe
from agent_support.gpt_oss_model import GptOss20BModel
from agent_support.gpt_oss_controller_worker import serve
from agent_support.input_token_fingerprint import input_ids_sha256, single_input_token_ids


def fixture():
    return {"schema_version": 1, "task_id": "textcraft:seed:89", "seed": 42,
            "messages": [{"role": "system", "content": "Choose an action."},
                         {"role": "user", "content": "Inventory and craft history"}],
            "max_new_tokens": 1024, "expected_input_tokens": 7555}


def test_export_saved_controller_event_preserves_messages_and_ceiling(tmp_path):
    config = probe.config_for_probe()
    data = {"task_id": "textcraft:seed:89", "trace": [{"decision": 18,
        "support_trace": {"events": [{"controller_messages": fixture()["messages"],
            "controller_cost": {"input_tokens": 7555},
            "state": {"remaining_budget": {"additional_tokens": 9000}}}]}}]}
    path = tmp_path / "episode.json"
    path.write_text(json.dumps(data))
    result = probe.export_episode(path, 18, 0, config)
    assert probe.list_episode_inputs(path) == [{"decision": 18, "event_index": 0,
        "input_tokens": 7555, "generation_status": None, "error": None}]
    assert result["messages"] == fixture()["messages"]
    assert result["expected_input_tokens"] == 7555
    assert result["max_new_tokens"] == 1024
    assert result["source"]["kind"] == "archived_episode"


def test_fixture_rejects_changed_generation_settings():
    value = fixture()
    value["max_new_tokens"] = 1025
    with pytest.raises(ValueError, match="ceiling"):
        probe.validate_fixture(value, 1024)


@pytest.mark.parametrize("target", [512, 7555])
def test_synthetic_fixture_extracts_batch_encoding_and_matches_worker_hash(monkeypatch, target):
    class FakeTensor:
        def __init__(self, values):
            self.values = values
        def tolist(self):
            return [self.values]
    class BatchEncoding(UserDict):
        pass
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["return_tensors"] == "pt" and kwargs["return_dict"] is True
            ids = list(range(len(messages[1]["content"]) // 4))
            return BatchEncoding(input_ids=FakeTensor(ids), attention_mask=FakeTensor([1] * len(ids)))
    tokenizer = Tokenizer()
    monkeypatch.setattr(probe, "_pinned_tokenizer", lambda _config: (tokenizer, "template"))
    config = probe.config_for_probe()
    first = probe.synthetic_fixture(config, target)
    second = probe.synthetic_fixture(config, target)
    assert first == second
    assert first["expected_input_tokens"] == target
    assert first["source"]["kind"] == "synthetic"
    model = object.__new__(GptOss20BModel)
    model._encode_messages = lambda messages, tensors: tokenizer.apply_chat_template(
        messages, return_tensors="pt", return_dict=True)
    model.model = SimpleNamespace(config=SimpleNamespace(_attn_implementation="eager"))
    assert model.probe_input(first["messages"])["input_ids_sha256"] == first[
        "expected_input_ids_sha256"]


def test_model_probe_fingerprints_exact_ids_without_generation():
    model = object.__new__(GptOss20BModel)
    model._encode_messages = lambda _messages, tensors: UserDict(input_ids=[[2, 7, 11]])
    model.model = SimpleNamespace(config=SimpleNamespace(_attn_implementation="flex_attention"))
    result = model.probe_input(fixture()["messages"])
    assert result["input_tokens"] == 3
    assert result["input_ids_sha256"] == hashlib.sha256(b"[2,7,11]").hexdigest()
    assert result["attention_implementation"] == "flex_attention"


def test_single_prompt_ids_reject_batch_flattening_and_bad_tokens():
    assert single_input_token_ids(UserDict(input_ids=[[3, 1, 4]])) == [3, 1, 4]
    assert input_ids_sha256([3, 1, 4]) == hashlib.sha256(b"[3,1,4]").hexdigest()
    with pytest.raises(ValueError, match="exactly one sequence"):
        single_input_token_ids(UserDict(input_ids=[[3, 1], [4, 1]]))
    with pytest.raises(ValueError, match="nonnegative integer"):
        single_input_token_ids(UserDict(input_ids=[[3, True, 4]]))


def test_worker_probe_op_returns_token_fingerprint_without_generation():
    class Model:
        def __init__(self, _config):
            self.calls = []
        def metadata(self):
            return {}
        def probe_input(self, messages):
            self.calls.append("probe")
            return {"input_tokens": len(messages), "input_ids_sha256": "a" * 64,
                    "attention_implementation": "eager"}
    source = io.StringIO("\n".join([
        json.dumps({"op": "init", "config": {}}),
        json.dumps({"op": "probe_input", "messages": fixture()["messages"]}), ""]))
    sink = io.StringIO()
    serve(source=source, sink=sink, model_factory=Model)
    responses = [json.loads(line) for line in sink.getvalue().splitlines()]
    assert responses[1]["probe"] == {"input_tokens": 2, "input_ids_sha256": "a" * 64,
                                    "attention_implementation": "eager"}


@pytest.mark.parametrize("backend,error", [("eager", False), ("flex_attention", True)])
def test_run_records_backend_token_identity_memory_and_oom(monkeypatch, tmp_path, backend, error):
    monkeypatch.setattr(probe.importlib.metadata, "version", lambda _: "4.57.6")
    calls = []
    token_digest = hashlib.sha256(b"same-token-sequence").hexdigest()
    case = fixture()
    case["expected_input_ids_sha256"] = token_digest
    class Worker:
        def __init__(self, spec, path):
            calls.append("worker_loaded")
            self.path = path
            self.path.mkdir(parents=True)
        def metadata(self):
            return {"backend_version": "5.22.2", "torch_version": "2.8.0+cu128",
                    "torch_cuda_version": "12.8", "gpu_name": "NVIDIA L40S",
                    "model_id": "openai/gpt-oss-20b",
                    "model_revision": "6cee5e81ee83917806bbde320786a8fb61efebee",
                    "quantization": "mxfp4_packed",
                    "decoding": {"do_sample": True, "temperature": 1.0, "top_p": 1.0,
                                 "top_k": 0, "max_new_tokens": 1024}}
        def probe_input(self, messages):
            assert messages == case["messages"]
            return {"attention_implementation": backend, "input_tokens": 7555,
                    "input_ids_sha256": token_digest}
        def begin_episode(self, task_id, seed):
            calls.append(("seed", task_id, seed))
        def generate(self, messages, max_new_tokens=None):
            assert calls[0:3] == ["worker_loaded", "qwen_loaded", "qwen_warmup"]
            assert max_new_tokens == 1024
            diagnostics = self.path.parent / "diagnostics"
            diagnostics.mkdir(exist_ok=True)
            with (diagnostics / "gpt_oss_cuda_memory_fake.jsonl").open("w") as stream:
                for event in ({"event": "generation_start", "request_id": 1,
                               "memory": {"allocated_bytes": 12, "reserved_bytes": 13}},
                              {"event": "generation_end", "request_id": 1,
                               "status": "error" if error else "ok",
                               "error_type": "OutOfMemoryError" if error else None,
                               "memory": {"peak_allocated_bytes": 34}}):
                    stream.write(json.dumps(event) + "\n")
            if error:
                from agent_support.controller_backend import ControllerWorkerOOM
                raise ControllerWorkerOOM("CUDA allocation failed")
            return {"generation_status": "complete", "input_tokens": 7555,
                    "output_tokens": 12, "inference_seconds": 1.2,
                    "raw_completion": "<|return|>"}
        def close(self):
            calls.append("closed")
    class Qwen:
        def __init__(self, _config):
            calls.append("qwen_loaded")
            self.model = SimpleNamespace(hf_device_map={"": 0})
            self.torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 1))
        def enable_gpu_memory_diagnostics(self, _directory):
            pass
        def generate(self, _messages):
            calls.append("qwen_warmup")
            return {"input_tokens": 20, "output_tokens": 4, "inference_seconds": 0.1}
    destination = tmp_path / backend
    initial_backend_env = os.environ.get("AGENT_SUPPORT_GPTOSS_FLEX_ATTENTION")
    result = probe.run_probe(case, backend, destination, worker_factory=Worker,
                             qwen_factory=Qwen)
    assert result["evaluator_transformers_version"] == "4.57.6"
    assert result["worker_metadata"]["backend_version"] == "5.22.2"
    assert result["qwen_warmup"]["output_tokens"] == 4
    assert result["input_probe"]["input_ids_sha256"] == token_digest
    assert result["telemetry"]["end"]["memory"]["peak_allocated_bytes"] == 34
    assert result["generation"]["status"] == ("error" if error else "complete")
    if error:
        assert result["generation"]["error_type"] == "OutOfMemoryError"
        assert result["generation"]["error_detail"] == "CUDA allocation failed"
    assert os.environ.get("AGENT_SUPPORT_GPTOSS_FLEX_ATTENTION") == initial_backend_env
    assert calls[-1] == "closed"
    assert json.loads((destination / "probe_result.json").read_text()) == result
    with pytest.raises(FileExistsError):
        probe.run_probe(case, backend, destination, worker_factory=Worker,
                        qwen_factory=Qwen)


def test_probe_rejects_cache_release_and_wrong_qwen_placement(monkeypatch, tmp_path):
    monkeypatch.setenv(probe.CACHE_RELEASE_ENV, "1")
    with pytest.raises(ValueError, match="Disable GPT-OSS CUDA cache release"):
        probe.run_probe(fixture(), "eager", tmp_path / "new")
    assert not (tmp_path / "new").exists()
    assert probe._require_single_cuda_qwen(SimpleNamespace(
        model=SimpleNamespace(hf_device_map={"": 0}),
        torch=SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 1)))) == {"": "0"}
    with pytest.raises(RuntimeError, match="CUDA 0"):
        probe._require_single_cuda_qwen(SimpleNamespace(
            model=SimpleNamespace(hf_device_map={"": "cpu"}),
            torch=SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 1))))
