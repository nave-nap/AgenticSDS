"""CPU-only checks for opt-in CUDA telemetry and unchanged generation behavior."""

from __future__ import annotations

import contextlib
import io
import json
from types import SimpleNamespace

import pytest

from agent_support import gpt_oss_model, recovery_v4_eval
from agent_support.common import canonical_digest
from agent_support.controller_backend import GptOssControllerWorker
from agent_support.gpu_memory_diagnostics import (DIRECTORY_ENV, ENABLE_ENV,
                                                  CACHE_RELEASE_ENV, GpuMemoryDiagnostics,
                                                  cache_release_enabled, enabled)
from agent_support.gpt_oss_controller_worker import serve
from agent_support.model import QwenModel


class FakeCuda:
    def __init__(self):
        self.allocated = 100
        self.reserved = 140
        self.peak_allocated = 100
        self.peak_reserved = 140
        self.resets = 0
        self.syncs = 0
        self.releases = 0
        self.release_error = None

    def bump(self, allocated, reserved):
        self.allocated = allocated
        self.reserved = reserved
        self.peak_allocated = max(self.peak_allocated, allocated)
        self.peak_reserved = max(self.peak_reserved, reserved)

    def memory_allocated(self, _device):
        return self.allocated

    def memory_reserved(self, _device):
        return self.reserved

    def max_memory_allocated(self, _device):
        return self.peak_allocated

    def max_memory_reserved(self, _device):
        return self.peak_reserved

    def mem_get_info(self, _device):
        return (48000 - self.reserved, 48000)

    def memory_stats(self, _device):
        return {"inactive_split_bytes.all.current": 7}

    def reset_peak_memory_stats(self, _device):
        self.resets += 1
        self.peak_allocated = self.allocated
        self.peak_reserved = self.reserved

    def synchronize(self, _device):
        self.syncs += 1

    def device_count(self):
        return 1

    def get_device_name(self, _device):
        return "Fake L40S"

    def empty_cache(self):
        self.releases += 1
        if self.release_error is not None:
            raise self.release_error
        self.reserved = self.allocated


class FakeTorch:
    def __init__(self):
        self.cuda = FakeCuda()

    def inference_mode(self):
        return contextlib.nullcontext()

    def ones_like(self, value):
        return value


class Row:
    def __init__(self, values):
        self.values = values
        self.shape = (len(values),)

    def __getitem__(self, key):
        return Row(self.values[key])

    def tolist(self):
        return self.values


class InputIds:
    shape = (1, 3)

    def to(self, _device):
        return self

    def __getitem__(self, _index):
        return Row([1, 2, 3])


class InputBatch(dict):
    def to(self, _device):
        return self


def events(directory):
    files = list(directory.glob("*.jsonl"))
    assert len(files) == 1
    return [json.loads(line) for line in files[0].read_text().splitlines()]


def fake_gpt(torch_api, telemetry, generate):
    model = object.__new__(gpt_oss_model.GptOss20BModel)
    model.torch = torch_api
    model.model = SimpleNamespace(device="cuda:0", generate=generate)
    model.tokenizer = SimpleNamespace(pad_token_id=0, decode=lambda ids, **_: str(ids))
    model._encode_messages = lambda *_args, **_kwargs: InputBatch(input_ids=InputIds())
    model.max_new_tokens = 1024
    model.temperature = model.top_p = 1.0
    model.top_k = 0
    model.return_token_id = 5
    model.completion_parser = "harmony_final_v3"
    model.peak_gpu_memory_bytes = 100
    model.peak_gpu_reserved_bytes = 140
    model.memory_diagnostics = telemetry
    model._diagnostic_request_id = 0
    return model


def test_opt_in_switch_and_worker_directory(monkeypatch, tmp_path):
    monkeypatch.delenv(ENABLE_ENV, raising=False)
    monkeypatch.delenv(CACHE_RELEASE_ENV, raising=False)
    assert not enabled()
    assert not cache_release_enabled()
    captured = []
    class Process:
        stdin = stdout = None
        def poll(self):
            return 0
    monkeypatch.setattr("agent_support.controller_backend.subprocess.Popen",
                        lambda *args, **kwargs: captured.append(kwargs["env"]) or Process())
    monkeypatch.setattr(GptOssControllerWorker, "_request", lambda self, *_: {"metadata": {
        "model_type": "gpt_oss_20b", "model_id": "fake", "model_revision": "fixed"}})
    monkeypatch.setenv("FAKE_GPT_PYTHON", __import__("sys").executable)
    spec = {"python_env": "FAKE_GPT_PYTHON", "model": {
        "model_id": "fake", "model_revision": "fixed", "max_new_tokens": 12}}
    worker = GptOssControllerWorker(spec, tmp_path / "workers")
    worker.close()
    assert DIRECTORY_ENV not in captured[-1]
    assert not (tmp_path / "diagnostics").exists()

    monkeypatch.setenv(ENABLE_ENV, "1")
    worker = GptOssControllerWorker(spec, tmp_path / "workers")
    worker.close()
    assert captured[-1][DIRECTORY_ENV] == str(tmp_path / "diagnostics")

    monkeypatch.delenv(ENABLE_ENV)
    monkeypatch.setenv(CACHE_RELEASE_ENV, "1")
    assert cache_release_enabled() and enabled()
    worker = GptOssControllerWorker(spec, tmp_path / "workers")
    worker.close()
    assert captured[-1][DIRECTORY_ENV] == str(tmp_path / "diagnostics")


@pytest.mark.parametrize("failure", [False, True])
def test_worker_releases_only_after_generation_and_preserves_response(monkeypatch, failure):
    monkeypatch.setenv(CACHE_RELEASE_ENV, "1")
    order = []
    class OutOfMemoryError(RuntimeError):
        pass
    class Model:
        def __init__(self, _config):
            self.in_generation = False
        def metadata(self):
            return {}
        def generate(self, _messages, max_new_tokens=None):
            order.append("generate")
            self.in_generation = True
            try:
                if failure:
                    raise OutOfMemoryError("allocation failed")
                return {"text": "inventory"}
            finally:
                self.in_generation = False
        def release_cuda_cache(self):
            assert not self.in_generation
            order.append("release")
    class Sink(io.StringIO):
        def write(self, value):
            order.append("response")
            return super().write(value)
    source = io.StringIO("\n".join([json.dumps({"op": "init", "config": {}}),
        json.dumps({"op": "generate", "messages": [], "max_new_tokens": 40}),
        json.dumps({"op": "count_messages_tokens", "messages": []}), ""]))
    # Count fails on this stand-in; it must not trigger another release.
    sink = Sink()
    serve(source=source, sink=sink, model_factory=Model)
    responses = [json.loads(line) for line in sink.getvalue().splitlines()]
    assert order == ["response", "generate", "release", "response", "response"]
    if failure:
        assert responses[1] == {"ok": False, "error_type": "OutOfMemoryError",
                                "error": "allocation failed"}
    else:
        assert responses[1] == {"ok": True, "generation": {"text": "inventory"}}


@pytest.mark.parametrize("generation_error", [False, True])
def test_worker_default_skips_release_and_cleanup_exception_is_contained(monkeypatch,
                                                                          generation_error):
    calls = []
    class OutOfMemoryError(RuntimeError):
        pass
    class Model:
        def __init__(self, _config):
            pass
        def metadata(self):
            return {}
        def generate(self, _messages, max_new_tokens=None):
            calls.append("generate")
            if generation_error:
                raise OutOfMemoryError("allocation failed")
            return {"text": "inventory"}
        def release_cuda_cache(self):
            calls.append("release")
            raise RuntimeError("cache release failed")
    source = "\n".join([json.dumps({"op": "init", "config": {}}),
        json.dumps({"op": "generate", "messages": []}), ""])
    monkeypatch.delenv(CACHE_RELEASE_ENV, raising=False)
    sink = io.StringIO()
    serve(source=io.StringIO(source), sink=sink, model_factory=Model)
    assert calls == ["generate"]
    expected = ({"ok": False, "error_type": "OutOfMemoryError", "error": "allocation failed"}
                if generation_error else {"ok": True, "generation": {"text": "inventory"}})
    assert json.loads(sink.getvalue().splitlines()[1]) == expected

    monkeypatch.setenv(CACHE_RELEASE_ENV, "1")
    sink = io.StringIO()
    serve(source=io.StringIO(source), sink=sink, model_factory=Model)
    assert calls == ["generate", "generate", "release"]
    assert json.loads(sink.getvalue().splitlines()[1]) == expected


def test_cache_release_records_before_after_and_cleanup_failure(tmp_path):
    torch_api = FakeTorch()
    telemetry = GpuMemoryDiagnostics(tmp_path, "gpt_oss_cuda_memory", torch_api)
    model = fake_gpt(torch_api, telemetry, lambda **_: None)
    model._diagnostic_request_id = 4
    torch_api.cuda.bump(100, 320)
    model.release_cuda_cache()
    release = events(tmp_path)[0]
    assert release["event"] == "cache_release" and release["request_id"] == 4
    assert release["status"] == "ok" and release["error_type"] is None
    assert release["memory_before"]["allocated_bytes"] == 100
    assert release["memory_before"]["reserved_bytes"] == 320
    assert release["memory_after"]["reserved_bytes"] == 100
    assert torch_api.cuda.releases == 1 and torch_api.cuda.syncs == 0

    torch_api.cuda.release_error = RuntimeError("cleanup failed")
    model.release_cuda_cache()
    failed = events(tmp_path)[1]
    assert failed["status"] == "error" and failed["error_type"] == "RuntimeError"
    assert torch_api.cuda.releases == 2


def test_diagnostic_batch_uses_its_own_run_directory(monkeypatch, tmp_path):
    monkeypatch.setenv(ENABLE_ENV, "1")
    arm = "autonomous_recovery_v4"
    task = recovery_v4_eval.manifest()["tasks"][0]
    qwen_paths = []
    worker_paths = []
    class Qwen:
        def enable_gpu_memory_diagnostics(self, directory):
            qwen_paths.append(directory)
    class Worker:
        def __init__(self, _spec, directory):
            worker_paths.append(directory)
        def metadata(self):
            return {"gpu_name": "NVIDIA L40S"}
        def begin_episode(self, *_args):
            pass
        def close(self):
            pass
    def episode(_task, *_args, **_kwargs):
        return {"task_id": task["task_id"], "arm": arm,
                "initial_observation": task["initial_observation"],
                "success": False, "status": "max_decisions",
                "environment_actions": 0, "trace": []}
    plan = {"git_commit": "fixed", "config_sha256": {
        arm: canonical_digest(recovery_v4_eval.config_for(arm))},
        "python_version": "fixed", "package_versions": {}}
    run_dir = recovery_v4_eval._run_tasks(tmp_path, arm, [0], plan,
        qwen_factory=lambda _config: Qwen(), gpt_factory=Worker,
        memory_factory=lambda *_args: None, episode_runner=episode)
    assert qwen_paths == [run_dir / "diagnostics"]
    assert worker_paths == [run_dir / "workers"]


def test_gpt_per_request_peaks_and_cumulative_fields(monkeypatch, tmp_path):
    torch_api = FakeTorch()
    telemetry = GpuMemoryDiagnostics(tmp_path, "gpt_oss_cuda_memory", torch_api)
    peaks = [(180, 220), (150, 200)]
    def generate(**_kwargs):
        torch_api.cuda.bump(*peaks.pop(0))
        return [Row([1, 2, 3, 4, 5])]
    monkeypatch.setattr(gpt_oss_model, "parse_configured_completion",
                        lambda *_: {"text": "inventory", "generation_status": "complete"})
    model = fake_gpt(torch_api, telemetry, generate)
    first = model.generate([], max_new_tokens=40)
    torch_api.cuda.bump(100, 180)
    second = model.generate([], max_new_tokens=40)
    assert first["peak_gpu_memory_allocated_bytes"] == 180
    assert second["peak_gpu_memory_allocated_bytes"] == 180
    assert second["peak_gpu_memory_reserved_bytes"] == 220
    assert torch_api.cuda.resets == 2 and torch_api.cuda.syncs == 4
    records = events(tmp_path)
    assert [item["event"] for item in records] == [
        "generation_start", "generation_end", "generation_start", "generation_end"]
    assert [item["request_id"] for item in records] == [1, 1, 2, 2]
    assert records[2]["previous_request_id"] == 1
    assert records[0]["peak_reset"] is True
    assert records[1]["input_tokens"] == 3 and records[1]["max_output_tokens"] == 40
    assert records[1]["memory"]["peak_allocated_bytes"] == 180
    assert records[3]["memory"]["peak_allocated_bytes"] == 150
    assert records[1]["memory"]["global_total_bytes"] == 48000
    assert records[1]["memory"]["inactive_split_bytes"] == 7
    assert all(item["timestamp_utc"] for item in records)


def test_gpt_oom_logs_error_and_next_request_start(monkeypatch, tmp_path):
    class OutOfMemoryError(RuntimeError):
        pass
    torch_api = FakeTorch()
    telemetry = GpuMemoryDiagnostics(tmp_path, "gpt_oss_cuda_memory", torch_api)
    calls = 0
    def generate(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            torch_api.cuda.bump(300, 340)
            torch_api.cuda.allocated = 100
            raise OutOfMemoryError("allocation failed")
        torch_api.cuda.bump(160, 350)
        return [Row([1, 2, 3, 4, 5])]
    monkeypatch.setattr(gpt_oss_model, "parse_configured_completion",
                        lambda *_: {"text": "inventory"})
    model = fake_gpt(torch_api, telemetry, generate)
    with pytest.raises(OutOfMemoryError, match="allocation failed"):
        model.generate([], max_new_tokens=40)
    result = model.generate([], max_new_tokens=40)
    records = events(tmp_path)
    assert records[1]["status"] == "error"
    assert records[1]["error_type"] == "OutOfMemoryError"
    assert records[1]["input_tokens"] == 3
    assert records[1]["memory"]["peak_allocated_bytes"] == 300
    assert records[2]["memory"]["allocated_bytes"] == 100
    assert records[2]["memory"]["reserved_bytes"] == 340
    assert result["peak_gpu_memory_allocated_bytes"] == 300


def test_phase_trace_identifies_generate_failure_without_post_error_memory_query(
        monkeypatch, tmp_path):
    class AcceleratorError(RuntimeError):
        pass

    torch_api = FakeTorch()
    telemetry = GpuMemoryDiagnostics(tmp_path, "gpt_oss_cuda_memory", torch_api)
    model = fake_gpt(torch_api, telemetry, lambda **_: (_ for _ in ()).throw(
        AcceleratorError("CUDA error: an illegal memory access was encountered")))
    monkeypatch.setenv("AGENT_SUPPORT_GPTOSS_PHASE_TRACE", "1")
    with pytest.raises(AcceleratorError):
        model.generate([])
    records = events(tmp_path)
    phases = [(item["phase"], item["status"]) for item in records
              if item["event"] == "generation_phase"]
    assert phases == [("input_transfer", "start"), ("input_transfer", "ok"),
                      ("pre_generation_synchronize", "start"),
                      ("pre_generation_synchronize", "ok"),
                      ("model_generate", "start"), ("model_generate", "error")]
    assert model._last_failed_phase == "model_generate"
    assert model._generation_request_id == 1
    assert torch_api.cuda.syncs == 2
    assert records[-1]["memory"] == {"skipped_after_generation_error": True}


def test_qwen_placement_and_generation_telemetry(monkeypatch, tmp_path):
    monkeypatch.setenv(ENABLE_ENV, "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    torch_api = FakeTorch()
    model = object.__new__(QwenModel)
    model.torch = torch_api
    model.max_new_tokens = 100
    model.memory_diagnostics = None
    model._diagnostic_request_id = 0
    model.tokenizer = SimpleNamespace(
        eos_token_id=0, apply_chat_template=lambda *_args, **_kwargs: InputIds(),
        decode=lambda *_args, **_kwargs: "get 1 log")
    def generate(**_kwargs):
        torch_api.cuda.bump(180, 230)
        return [Row([1, 2, 3, 4])]
    model.model = SimpleNamespace(hf_device_map={"": 0},
                                  parameters=lambda: iter([SimpleNamespace(device="cuda:0")]),
                                  generate=generate)
    model.enable_gpu_memory_diagnostics(tmp_path)
    result = model.generate([])
    assert result["text"] == "get 1 log" and result["input_tokens"] == 3
    records = events(tmp_path)
    assert [item["event"] for item in records] == ["model_loaded", "generation_start", "generation_end"]
    assert records[0]["hf_device_map"] == {"": "0"}
    assert records[0]["cuda_visible_devices"] == "0"
    assert records[0]["visible_cuda_devices"] == [{"index": 0, "name": "Fake L40S"}]
    assert records[2]["memory"]["peak_reserved_bytes"] == 230
    assert torch_api.cuda.syncs == 0

    class OutOfMemoryError(RuntimeError):
        pass
    def oom(**_kwargs):
        torch_api.cuda.bump(260, 300)
        raise OutOfMemoryError("Qwen allocation failed")
    model.model.generate = oom
    with pytest.raises(OutOfMemoryError, match="Qwen allocation failed"):
        model.generate([])
    records = events(tmp_path)
    assert records[3]["previous_request_id"] == 1
    assert records[4]["request_id"] == 2
    assert records[4]["status"] == "error" and records[4]["error_type"] == "OutOfMemoryError"
    assert records[4]["memory"]["peak_allocated_bytes"] == 260
    assert torch_api.cuda.syncs == 0


def test_telemetry_write_failure_does_not_change_result_or_exception(monkeypatch, tmp_path):
    torch_api = FakeTorch()
    telemetry = GpuMemoryDiagnostics(tmp_path, "gpt_oss_cuda_memory", torch_api)
    telemetry.path = tmp_path / "missing" / "no-write.jsonl"
    monkeypatch.setattr(gpt_oss_model, "parse_configured_completion",
                        lambda *_: {"text": "inventory"})
    model = fake_gpt(torch_api, telemetry, lambda **_: [Row([1, 2, 3, 4, 5])])
    assert model.generate([])["text"] == "inventory"
    def broken(**_kwargs):
        raise RuntimeError("backend error")
    model.model.generate = broken
    with pytest.raises(RuntimeError, match="backend error"):
        model.generate([])
