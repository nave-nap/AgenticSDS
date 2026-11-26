"""CPU-only checks for overnight gates and attempt artifacts."""

import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_support import three_gpu_overnight as campaign
from agent_support.gpt_oss_controller_worker import serve


def test_cache_probe_reads_only_pinned_snapshot_and_keeps_attempts(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "require_clean_commit", lambda: "test-commit")
    monkeypatch.setattr(campaign, "config_for", lambda _:
                        {"autonomous_support": {"model": {"model": {"model_revision": "fixed"}}}})
    snapshot = tmp_path / "hf" / "hub" / "models--openai--gpt-oss-20b" / "snapshots" / "fixed"
    snapshot.mkdir(parents=True)
    checkpoint = snapshot / "model-00001.safetensors"
    checkpoint.write_bytes(b"x" * 8192)
    result = campaign.cache_probe(tmp_path / "attempt", tmp_path / "hf",
                                  sample_limit_bytes=8192, per_file_bytes=8192,
                                  min_sample_bytes=8192, min_speed_mib_per_second=0)
    assert result["status"] == "cache_valid"
    assert result["bytes_read"] == 8192
    assert checkpoint.read_bytes() == b"x" * 8192
    assert json.loads((tmp_path / "attempt" / "cache_result.json").read_text())["status"] == "cache_valid"
    with pytest.raises(FileExistsError):
        campaign.cache_probe(tmp_path / "attempt", tmp_path / "hf")


def _fake_gpu(monkeypatch, tmp_path, *, slow_first=False):
    monkeypatch.setattr(campaign, "require_clean_commit", lambda: "test-commit")
    monkeypatch.setattr(campaign, "preflight", lambda: {"protocol": "test"})
    monkeypatch.setattr(campaign, "config_for", lambda _: {"autonomous_support": {"model": {}}})
    monkeypatch.setattr(campaign, "_synthetic_messages", lambda _worker, target=0:
                        [{"role": "user", "content": f"{target}"}])
    monkeypatch.setattr(campaign, "_messages", lambda _worker, _path:
                        ([{"role": "user", "content": "8887"}], "synthetic_exact_length"))
    monkeypatch.setattr(campaign, "validate_stage", lambda _rows, _request_id, _count:
                        {"memory": {"cuda:0": {"peak_reserved_bytes": 1,
                                                "minimum_global_free_bytes": 1},
                                    "cuda:1": {"peak_reserved_bytes": 1,
                                                "minimum_global_free_bytes": 1}},
                         "timing": {"prefill_observed_seconds": 1}})
    torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 1,
                                                  get_device_name=lambda _: "NVIDIA L40S",
                                                  reset_peak_memory_stats=lambda _: None,
                                                  memory_allocated=lambda _: 1,
                                                  memory_reserved=lambda _: 2,
                                                  max_memory_allocated=lambda _: 3,
                                                  max_memory_reserved=lambda _: 4))
    monkeypatch.setitem(sys.modules, "torch", torch)
    from agent_support import controller_backend, model
    calls = []
    class Worker:
        def __init__(self, _spec, directory):
            diagnostics = directory.parent / "diagnostics"
            diagnostics.mkdir()
            (diagnostics / "gpt_oss_cuda_memory_test.jsonl").write_text("{}\n")
            self.diagnostics = diagnostics
            self.requests = 0
            self.timeout = 1800
        def metadata(self):
            return {"attention_implementation": campaign.CHUNKED_NAME,
                    "verified_expert_modules_by_device": {"cuda:0": 12, "cuda:1": 12}}
        def count_messages_tokens(self, messages):
            return int(messages[0]["content"])
        def generate(self, messages, max_new_tokens):
            calls.append("generate")
            self.requests += 1
            (self.diagnostics / f"gpt_oss_inference_{self.requests:04d}.jsonl").write_text(
                json.dumps({"event": "sample", "gpus": {"worker_cuda_ordinals": {
                    "cuda:0": {"gpu_utilization_percent": 40},
                    "cuda:1": {"gpu_utilization_percent": 60}}}}) + "\n")
            return {"input_tokens": 70 if messages[0]["content"] == "Reply briefly."
                    else int(messages[0]["content"]),
                    "output_tokens": max_new_tokens, "inference_seconds": 1}
        def profile_generate(self, _messages):
            calls.append("profile")
            return {"profile_path": str(tmp_path / "trace.json"),
                    "generation": {"inference_seconds": 2}}
        def close(self):
            calls.append("close")
    monkeypatch.setattr(controller_backend, "GptOssControllerWorker", Worker)
    monkeypatch.setattr(model, "QwenModel", lambda _: SimpleNamespace(
        model=SimpleNamespace(hf_device_map={"": 0})))
    for key in (campaign.SHARD_ENV, "AGENT_SUPPORT_GPTOSS_CHUNKED_EAGER",
                "AGENT_SUPPORT_GPU_MEMORY_DIAGNOSTICS",
                "AGENT_SUPPORT_GPTOSS_MODEL_LOAD_DIAGNOSTICS",
                "AGENT_SUPPORT_GPTOSS_INFERENCE_SAMPLING",
                "AGENT_SUPPORT_GPTOSS_ALLOW_PROFILER"):
        monkeypatch.setenv(key, "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv(campaign.VISIBLE_ENV, "1,2")
    if slow_first:
        monkeypatch.setattr(campaign.time, "monotonic", iter((0, 1, 2, 3, 4, 700)).__next__)
    return calls


def test_gpu_campaign_reuses_worker_and_reaches_long_only_after_gates(tmp_path, monkeypatch):
    calls = _fake_gpu(monkeypatch, tmp_path)
    result = campaign.run(tmp_path / "gpu", None, None)
    assert result["status"] == "performance_valid"
    assert [stage["name"] for stage in result["stages"]] == [row[0] for row in campaign.STAGES]
    assert all(stage["status"] == "valid" for stage in result["stages"])
    assert calls == ["generate"] * 4 + ["close"]
    assert (tmp_path / "gpu" / "summary.md").exists()


def test_slow_stage_profiles_once_and_blocks_later_stages(tmp_path, monkeypatch):
    calls = _fake_gpu(monkeypatch, tmp_path)
    monkeypatch.setattr(campaign, "STAGES", (("warmup_first", 8, -1, 1200),
                                                ("warmup_repeat", 8, 120, 300)))
    result = campaign.run(tmp_path / "gpu", None, None)
    assert result["status"] == "failed"
    assert result["profiler"]["status"] == "complete"
    assert [stage["name"] for stage in result["stages"]] == ["warmup_first"]
    assert calls == ["generate", "profile", "close"]


def test_invalid_cpu_dependency_stops_before_gpu_load(tmp_path, monkeypatch):
    calls = _fake_gpu(monkeypatch, tmp_path)
    cache = tmp_path / "cache_result.json"
    cache.write_text(json.dumps({"status": "failed", "git_commit": "test-commit"}))
    result = campaign.run(tmp_path / "gpu", None, cache)
    assert result["status"] == "failed"
    assert "cache dependency" in result["error_detail"]
    assert calls == []


def test_profiler_requires_opt_in_and_bounded_request(monkeypatch):
    monkeypatch.delenv("AGENT_SUPPORT_GPTOSS_ALLOW_PROFILER", raising=False)
    class Model:
        def __init__(self, _config):
            pass
        def metadata(self):
            return {}
        def count_messages_tokens(self, _messages):
            return 10
        def generate(self, _messages, max_new_tokens=None):
            raise AssertionError("Profiler must not generate before its guard")
    requests = [{"op": "init", "config": {}},
                {"op": "profile_generate", "messages": [], "max_new_tokens": 2}]
    sink = io.StringIO()
    serve(source=io.StringIO("\n".join(map(json.dumps, requests)) + "\n"),
          sink=sink, model_factory=Model)
    assert "not enabled" in json.loads(sink.getvalue().splitlines()[1])["error"]

    monkeypatch.setenv("AGENT_SUPPORT_GPTOSS_ALLOW_PROFILER", "1")
    requests[1]["max_new_tokens"] = 3
    sink = io.StringIO()
    serve(source=io.StringIO("\n".join(map(json.dumps, requests)) + "\n"),
          sink=sink, model_factory=Model)
    assert "limited to two tokens" in json.loads(sink.getvalue().splitlines()[1])["error"]
