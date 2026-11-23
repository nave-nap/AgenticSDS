"""CPU checks for the optional GPT-OSS attention backend selection."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from agent_support import gpt_oss_model
from agent_support.gpt_oss_model import FLEX_ATTENTION_ENV, attention_load_kwargs


def test_default_preserves_loader_kwargs(monkeypatch):
    monkeypatch.delenv(FLEX_ATTENTION_ENV, raising=False)
    assert attention_load_kwargs(object()) == {}
    monkeypatch.setenv(FLEX_ATTENTION_ENV, "0")
    assert attention_load_kwargs(object()) == {}


def test_opt_in_selects_only_supported_flex_backend(monkeypatch):
    utils = ModuleType("transformers.utils")
    utils.is_torch_flex_attn_available = lambda: True
    monkeypatch.setitem(sys.modules, "transformers", ModuleType("transformers"))
    monkeypatch.setitem(sys.modules, "transformers.utils", utils)
    torch_api = SimpleNamespace(cuda=SimpleNamespace(get_device_capability=lambda _: (8, 9)))
    monkeypatch.setenv(FLEX_ATTENTION_ENV, "1")
    assert attention_load_kwargs(torch_api) == {"attn_implementation": "flex_attention"}

    utils.is_torch_flex_attn_available = lambda: False
    with pytest.raises(RuntimeError, match="compatible PyTorch CUDA runtime"):
        attention_load_kwargs(torch_api)
    utils.is_torch_flex_attn_available = lambda: True
    torch_api.cuda.get_device_capability = lambda _: (7, 5)
    with pytest.raises(RuntimeError, match="compatible PyTorch CUDA runtime"):
        attention_load_kwargs(torch_api)


def test_invalid_switch_fails_closed(monkeypatch):
    monkeypatch.setenv(FLEX_ATTENTION_ENV, "sdpa")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        attention_load_kwargs(object())


@pytest.mark.parametrize(("enabled", "backend_mismatch"),
                         [(False, False), (True, False), (True, True)])
def test_loader_passes_opt_in_backend_without_changing_default(monkeypatch, enabled,
                                                               backend_mismatch):
    calls = []
    torch = ModuleType("torch")
    torch.manual_seed = lambda _: None
    torch.cuda = SimpleNamespace(is_available=lambda: True,
        get_device_capability=lambda _: (8, 9), manual_seed_all=lambda _: None,
        set_device=lambda _: None, reset_peak_memory_stats=lambda _: None,
        max_memory_allocated=lambda _: 12, max_memory_reserved=lambda _: 13)
    transformers = ModuleType("transformers")
    transformers.utils = ModuleType("transformers.utils")
    transformers.utils.is_torch_flex_attn_available = lambda: True

    class Tokenizer:
        chat_template = '{{ strftime_now("%Y-%m-%d") }}'

        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

        def convert_tokens_to_ids(self, _):
            return 1

    class Model:
        @classmethod
        def from_pretrained(cls, *_args, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(config=SimpleNamespace(_attn_implementation=(
                "flex_attention" if "attn_implementation" in kwargs and not backend_mismatch
                else "eager")),
                eval=lambda: None)

    transformers.AutoTokenizer = Tokenizer
    transformers.AutoModelForCausalLM = Model
    integrations = ModuleType("transformers.integrations")
    integrations.mxfp4 = SimpleNamespace(triton_kernels_hub=None, Mxfp4GptOssExperts=object)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "transformers.utils", transformers.utils)
    monkeypatch.setitem(sys.modules, "transformers.integrations", integrations)
    monkeypatch.setattr(gpt_oss_model, "check_mxfp4_dependencies", lambda _: {"transformers": "5.22.2"})
    monkeypatch.setattr(gpt_oss_model, "verify_packed_mxfp4", lambda *_args, **_kwargs: {})
    monkeypatch.setenv(FLEX_ATTENTION_ENV, "1" if enabled else "0")
    config = {"backend_version": "5.22.2", "seed": 42, "model_id": "fake",
        "model_revision": "fixed", "max_new_tokens": 1024, "reasoning_effort": "low",
        "temperature": 1.0, "top_p": 1.0, "top_k": 0, "prompt_date": "2026-11-18"}
    if backend_mismatch:
        with pytest.raises(RuntimeError, match="did not activate the requested"):
            gpt_oss_model.GptOss20BModel(config)
        return
    model = gpt_oss_model.GptOss20BModel(config)
    assert model.flex_attention_enabled is enabled
    assert model.attention_implementation == ("flex_attention" if enabled else "eager")
    assert calls[0].get("attn_implementation") == ("flex_attention" if enabled else None)
    assert calls[0]["device_map"] == {"": 0} and calls[0]["dtype"] == "auto"
