"""CPU checks for the opt-in three-GPU runtime contract."""

from __future__ import annotations

import json
import sys
import time
from types import SimpleNamespace

import pytest

from agent_support.controller_backend import GptOssControllerWorker
from agent_support.gpt_oss_model import verify_packed_mxfp4
from agent_support.gpt_oss_two_gpu import (SHARD_ENV, VISIBLE_ENV, assert_device_map,
                                            assert_two_gpu_runtime, layer_device_map)
from agent_support.gpu_memory_diagnostics import TwoGpuMemoryDiagnostics
from agent_support import three_gpu_feasibility as feasibility
from agent_support.three_gpu_feasibility import TARGET_TOKENS, _synthetic_messages


def test_fixed_decoder_split_and_placement_rejections():
    mapping = layer_device_map(24)
    assert mapping["model.embed_tokens"] == 0
    assert [mapping[f"model.layers.{i}"] for i in range(24)] == [0] * 12 + [1] * 12
    assert mapping["model.norm"] == mapping["lm_head"] == 1
    assert_device_map(SimpleNamespace(hf_device_map=mapping), mapping)
    with pytest.raises(RuntimeError, match="Unsupported"):
        layer_device_map(23)
    with pytest.raises(RuntimeError, match="loaded device map differs"):
        assert_device_map(SimpleNamespace(hf_device_map={"": 0}), mapping)
    bad = {**mapping, "model.layers.12": 0}
    with pytest.raises(RuntimeError, match="model.layers.12"):
        assert_device_map(SimpleNamespace(hf_device_map=bad), mapping)


def test_packed_expert_backing_and_scales_are_checked_per_shard():
    class StorageTensor:
        dtype = "uint8"
        is_meta = False
        def __init__(self, device, count):
            self.device, self.count = device, count
        def numel(self):
            return self.count
        def element_size(self):
            return 1
    class Layout:
        pass
    class StridedLayout:
        pass
    fp4 = object()
    class Packed:
        def __init__(self, device, shape, count, dtype, layout):
            self.dtype, self.shape = dtype, shape
            self.storage = SimpleNamespace(data=StorageTensor(device, count), layout=layout)
    class Expert:
        def __init__(self, device):
            self._is_hf_initialized = True
            self.num_experts = 1
            self.hidden_size = self.intermediate_size = 32
            for projection, shape, count in (
                    ("gate_up_proj", (1, 32, 64), 1024),
                    ("down_proj", (1, 32, 32), 512)):
                setattr(self, projection, Packed(device, shape, count, fp4, Layout()))
                scale = Packed(device, (1, 1, shape[2]), count // 16, "uint8", StridedLayout())
                setattr(self, f"{projection}_precision_config", SimpleNamespace(weight_scale=scale))
            self._parameters = {}
    mapping = layer_device_map(24)
    experts = {f"model.layers.{i}.mlp.experts": Expert(f"cuda:{mapping[f'model.layers.{i}']}")
               for i in range(24)}
    parameters = [(f"model.layers.{i}.input_layernorm.weight",
                   StorageTensor(f"cuda:{mapping[f'model.layers.{i}']}", 32)) for i in range(24)]
    model = SimpleNamespace(config=SimpleNamespace(quantization_config={"quant_method": "mxfp4"},
                                                   num_hidden_layers=24),
                            named_modules=lambda: experts.items(),
                            named_parameters=lambda: parameters)
    hub = SimpleNamespace(tensor=SimpleNamespace(Tensor=Packed, FP4=fp4),
                          tensor_details=SimpleNamespace(layout=SimpleNamespace(
                              make_default_matmul_mxfp4_w_layout=lambda mx_axis: (Layout, None),
                              StridedLayout=StridedLayout)))
    torch_api = SimpleNamespace(Tensor=StorageTensor, uint8="uint8",
                                device=lambda kind, index: f"{kind}:{index}")
    verified = verify_packed_mxfp4(model, torch_api, hub, Expert, device_map=mapping)
    assert verified["verified_expert_modules_by_device"] == {"cuda:0": 12, "cuda:1": 12}
    assert verified["packed_weight_storage_bytes_by_device"] == {"cuda:0": 18432, "cuda:1": 18432}
    assert verified["scale_storage_bytes_by_device"] == {"cuda:0": 1152, "cuda:1": 1152}
    experts["model.layers.12.mlp.experts"].gate_up_proj.storage.data.device = "cuda:0"
    with pytest.raises(RuntimeError, match="packed backing storage"):
        verify_packed_mxfp4(model, torch_api, hub, Expert, device_map=mapping)


def test_runtime_requires_two_l40s():
    class Cuda:
        def __init__(self, names):
            self.names = names
        def device_count(self):
            return len(self.names)
        def get_device_name(self, index):
            return self.names[index]
        def get_device_capability(self, _index):
            return (8, 9)
    assert_two_gpu_runtime(SimpleNamespace(cuda=Cuda(["L40S", "L40S"])))
    with pytest.raises(RuntimeError, match="exactly two"):
        assert_two_gpu_runtime(SimpleNamespace(cuda=Cuda(["L40S"])))
    with pytest.raises(RuntimeError, match="not an L40S"):
        assert_two_gpu_runtime(SimpleNamespace(cuda=Cuda(["L40S", "A100"])))


def test_worker_isolates_requested_gpus_and_rejects_unsharded_load(monkeypatch, tmp_path):
    captured = []
    class Process:
        stdin = stdout = None
        def poll(self):
            return 0
    monkeypatch.setattr("agent_support.controller_backend.subprocess.Popen",
                        lambda *args, **kwargs: captured.append(kwargs["env"]) or Process())
    monkeypatch.setattr(GptOssControllerWorker, "_request", lambda self, *_: {"metadata": {
        "model_type": "gpt_oss_20b", "model_id": "fake", "model_revision": "fixed",
        "sharding": "explicit_decoder_layers_12_12",
        "verified_expert_modules_by_device": {"cuda:0": 12, "cuda:1": 12}}})
    monkeypatch.setenv("FAKE_GPT_PYTHON", sys.executable)
    monkeypatch.setenv(SHARD_ENV, "1")
    monkeypatch.setenv(VISIBLE_ENV, "1,2")
    monkeypatch.setenv("AGENT_SUPPORT_GPU_MEMORY_DIAGNOSTICS", "1")
    spec = {"python_env": "FAKE_GPT_PYTHON", "model": {
        "model_id": "fake", "model_revision": "fixed", "max_new_tokens": 12}}
    worker = GptOssControllerWorker(spec, tmp_path / "workers")
    worker.close()
    assert captured[-1]["CUDA_VISIBLE_DEVICES"] == "1,2"
    assert captured[-1]["AGENT_SUPPORT_GPU_MEMORY_DIAGNOSTICS_DIR"] == str(tmp_path / "diagnostics")
    monkeypatch.setenv(VISIBLE_ENV, "1,1")
    with pytest.raises(ValueError, match="two distinct"):
        GptOssControllerWorker(spec, tmp_path / "workers")
    monkeypatch.setenv(VISIBLE_ENV, "1,2")
    monkeypatch.setattr(GptOssControllerWorker, "_request", lambda self, *_: {"metadata": {
        "model_type": "gpt_oss_20b", "model_id": "fake", "model_revision": "fixed"}})
    with pytest.raises(RuntimeError, match="did not load packed experts"):
        GptOssControllerWorker(spec, tmp_path / "workers")


def test_two_gpu_memory_records_peaks_minimum_free_and_oom(tmp_path):
    class Cuda:
        allocated = [100, 200]
        reserved = [120, 240]
        peaks = [100, 200]
        def mem_get_info(self, index):
            return (1000 - self.reserved[index], 1000)
        def memory_allocated(self, index):
            return self.allocated[index]
        def memory_reserved(self, index):
            return self.reserved[index]
        def max_memory_allocated(self, index):
            return self.peaks[index]
        def max_memory_reserved(self, index):
            return self.reserved[index]
        def memory_stats(self, _index):
            return {"inactive_split_bytes.all.current": 0}
        def reset_peak_memory_stats(self, index):
            self.peaks[index] = self.allocated[index]
    cuda = Cuda()
    telemetry = TwoGpuMemoryDiagnostics(tmp_path, "two_gpu", SimpleNamespace(cuda=cuda))
    telemetry.start(1, 1024)
    cuda.allocated = [300, 450]
    cuda.reserved = [340, 490]
    cuda.peaks = [300, 450]
    time.sleep(0.06)
    telemetry.finish(1, TARGET_TOKENS, 1024, "OutOfMemoryError", "requested allocation failed")
    rows = [json.loads(line) for line in telemetry.path.read_text().splitlines()]
    end = rows[-1]
    assert end["status"] == "error" and end["error_detail"] == "requested allocation failed"
    assert end["memory"]["devices"]["cuda:1"]["peak_allocated_bytes"] == 450
    assert end["minimum_global_free_bytes"] == {"cuda:0": 660, "cuda:1": 510}
    assert all(value > 0 for value in end["free_memory_samples"].values())


def test_synthetic_feasibility_request_has_exact_token_count():
    class Worker:
        def count_messages_tokens(self, messages):
            return len(messages[0]["content"].split()) + 7
    worker = Worker()
    messages = _synthetic_messages(worker)
    assert worker.count_messages_tokens(messages) == TARGET_TOKENS


def test_smoke_gate_requires_valid_result_on_same_commit(monkeypatch, tmp_path):
    monkeypatch.setattr(feasibility, "preflight", lambda: {"protocol": "fixed"})
    monkeypatch.setattr(feasibility, "require_clean_commit", lambda: "abc")
    path = tmp_path / "feasibility_result.json"
    valid = {"protocol": "fixed", "status": "infrastructure_valid", "git_commit": "abc"}
    path.write_text(json.dumps(valid))
    assert feasibility.gate(path)["status"] == "gate_passed"
    for bad in ({**valid, "status": "failed"}, {**valid, "git_commit": "old"},
                {**valid, "protocol": "different"}):
        path.write_text(json.dumps(bad))
        with pytest.raises(RuntimeError, match="not infrastructure valid"):
            feasibility.gate(path)
