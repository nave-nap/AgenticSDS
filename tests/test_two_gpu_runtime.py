"""CPU-only checks for the isolated two-GPU process placement."""
from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from agent_support.controller_backend import GptOssControllerWorker
from agent_support.gpu_memory_diagnostics import GpuMemoryDiagnostics
from agent_support.model import QwenModel
from agent_support.recovery_v4_eval import config_for


def test_qwen_loader_pins_gpu_zero_only_when_requested(monkeypatch):
    calls = []
    torch = ModuleType("torch")
    torch.float16 = "float16"
    torch.manual_seed = lambda _seed: None
    torch.cuda = SimpleNamespace(is_available=lambda: True, manual_seed_all=lambda _seed: None)
    transformers = ModuleType("transformers")

    class Tokenizer:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return object()

    class Model:
        @staticmethod
        def from_pretrained(*_args, **kwargs):
            calls.append(kwargs["device_map"])
            return SimpleNamespace(eval=lambda: None)

    transformers.AutoTokenizer = Tokenizer
    transformers.AutoModelForCausalLM = Model
    transformers.BitsAndBytesConfig = lambda **_kwargs: object()
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    config = {"seed": 42, "model_id": "fake", "model_revision": "fixed",
              "max_new_tokens": 100}
    monkeypatch.setenv("AGENT_SUPPORT_QWEN_GPU", "0")
    QwenModel(config)
    monkeypatch.delenv("AGENT_SUPPORT_QWEN_GPU")
    QwenModel(config)
    assert calls == [{"": 0}, "auto"]
    monkeypatch.setenv("AGENT_SUPPORT_QWEN_GPU", "1")
    with pytest.raises(ValueError, match="must be 0"):
        QwenModel(config)


def test_gptoss_worker_sees_only_allocation_gpu_one(monkeypatch, tmp_path):
    spec = config_for("autonomous_recovery_v4")["autonomous_support"]["model"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("AGENT_SUPPORT_GPTOSS_WORKER_GPU", "1")
    monkeypatch.setenv(spec["python_env"], sys.executable)
    monkeypatch.setenv(spec["hf_home_env"], str(tmp_path))
    launched = []

    def fake_popen(_argv, **kwargs):
        launched.append(kwargs["env"])
        return SimpleNamespace()

    monkeypatch.setattr("agent_support.controller_backend.subprocess.Popen", fake_popen)
    monkeypatch.setattr(GptOssControllerWorker, "_request", lambda self, _payload: {
        "metadata": {"model_type": "gpt_oss_20b",
                     "model_id": spec["model"]["model_id"],
                     "model_revision": spec["model"]["model_revision"]}})
    worker = GptOssControllerWorker(spec, tmp_path / "workers")
    worker.log.close()
    assert launched[0]["CUDA_VISIBLE_DEVICES"] == "1"
    assert launched[0]["HF_HOME"] == str(tmp_path)
    monkeypatch.setenv("AGENT_SUPPORT_GPTOSS_WORKER_GPU", "2")
    with pytest.raises(ValueError, match="outside the visible allocation"):
        GptOssControllerWorker(spec, tmp_path / "invalid")
    assert len(launched) == 1


def test_memory_events_identify_process_gpu(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    telemetry = GpuMemoryDiagnostics(tmp_path, "gpt_oss_cuda_memory", object())
    telemetry.write(event="placement_check")
    event = json.loads(next(tmp_path.glob("*.jsonl")).read_text())
    assert event["cuda_visible_devices"] == "1"
    assert event["logical_cuda_device"] == 0
