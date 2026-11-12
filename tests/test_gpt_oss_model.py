"""CPU-only contract tests for the GPT-OSS adapter and configuration."""

from pathlib import Path
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from agent_support.common import read_json, validate_config, validate_manifest
from agent_support.components import load_components
from agent_support.gpt_oss_model import (ANALYSIS_MARKER, END_TOKEN, FINAL_MARKER,
                                         RETURN_TOKEN, GptOss20BModel, parse_harmony_completion)
from agent_support.model import make_model, MODEL_REGISTRY
from agent_support.runner import run_episode


class HarmonyTokenizer:
    special = {"<|start|>": 1000, "<|channel|>": 1001, "<|message|>": 1002,
               END_TOKEN: 1003, RETURN_TOKEN: 1004}
    reverse = {value: key for key, value in special.items()}

    def encode(self, text, add_special_tokens=False):
        ids = []
        while text:
            marker = next((part for part in self.special if text.startswith(part)), None)
            if marker:
                ids.append(self.special[marker])
                text = text[len(marker):]
            else:
                ids.append(ord(text[0]))
                text = text[1:]
        return ids

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.reverse.get(token, chr(token)) for token in ids)

    def convert_tokens_to_ids(self, token):
        return self.special[token]

    def apply_chat_template(self, messages, **kwargs):
        self.template_args = kwargs
        ids = self.encode("<|start|>assistant")
        if kwargs.get("return_tensors"):
            return FakeInputs(ids)
        return ids

    pad_token_id = 1003


class FakeIds:
    def __init__(self, ids):
        self.ids = ids
        self.shape = (1, len(ids))

    def __getitem__(self, sl):
        return FakeIds(self.ids[sl])

    def tolist(self):
        return self.ids


class FakeInputs(dict):
    def __init__(self, ids):
        super().__init__(input_ids=FakeIds(ids))

    def to(self, device):
        return self


class FakeBackend:
    device = "cuda:0"

    def __init__(self, output):
        self.output = output

    def generate(self, **kwargs):
        self.generation_args = kwargs
        return [FakeIds(kwargs["input_ids"].ids + self.output)]


def parse(text, limit=1024):
    tokenizer = HarmonyTokenizer()
    return parse_harmony_completion(tokenizer, tokenizer.encode(text), limit)


def test_harmony_final_only_and_reasoning_usage():
    result = parse(ANALYSIS_MARKER + "think" + END_TOKEN +
                   "<|start|>assistant" + FINAL_MARKER + "Action: inventory" + RETURN_TOKEN)
    assert result["text"] == "Action: inventory"
    assert result["reasoning_tokens"] == 5
    assert result["final_tokens"] == len("Action: inventory")
    assert result["generation_status"] == "complete"
    assert "think" in result["raw_completion"]


@pytest.mark.parametrize("completion,status", [
    (ANALYSIS_MARKER + "thinking" + END_TOKEN, "missing_final"),
    (FINAL_MARKER + "Action: inventory", "malformed_final"),
    (FINAL_MARKER + "", "malformed_final"),
    (FINAL_MARKER + "Action: inventory" + RETURN_TOKEN + FINAL_MARKER +
     "Action: look" + RETURN_TOKEN, "malformed_harmony"),
    (FINAL_MARKER + "<|start|>oops" + RETURN_TOKEN, "malformed_final"),
])
def test_malformed_output_never_exposes_action(completion, status):
    result = parse(completion)
    assert result["text"] == ""
    assert result["generation_status"] == status


def test_generation_limit_marks_missing_or_truncated_final():
    no_final = ANALYSIS_MARKER + "unfinished"
    assert parse(no_final, len(HarmonyTokenizer().encode(no_final)))["generation_status"] == "truncated_before_final"
    final = FINAL_MARKER + "Action: inventory"
    assert parse(final, len(HarmonyTokenizer().encode(final)))["generation_status"] == "truncated_final"


def test_mocked_adapter_generation_uses_harmony_and_counts_tokens():
    adapter = object.__new__(GptOss20BModel)
    adapter.tokenizer = HarmonyTokenizer()
    completion = ANALYSIS_MARKER + "think" + END_TOKEN + "<|start|>assistant" + FINAL_MARKER + "Action: inventory" + RETURN_TOKEN
    adapter.model = FakeBackend(adapter.tokenizer.encode(completion))
    adapter.torch = SimpleNamespace(cuda=SimpleNamespace(
        synchronize=lambda *_: None, max_memory_allocated=lambda *_: 123,
        max_memory_reserved=lambda *_: 456), inference_mode=nullcontext)
    adapter.max_new_tokens = 1024
    adapter.reasoning_effort = "medium"
    adapter.chat_template = "fixed template"
    adapter.temperature = 1.0
    adapter.top_p = 1.0
    adapter.top_k = 0
    adapter.return_token_id = adapter.tokenizer.convert_tokens_to_ids(RETURN_TOKEN)
    adapter.peak_gpu_memory_bytes = 0
    adapter.peak_gpu_reserved_bytes = 0
    messages = [{"role": "user", "content": "task"}]
    result = adapter.generate(messages)
    assert result["text"] == "Action: inventory"
    assert result["input_tokens"] == len(adapter.tokenizer.encode("<|start|>assistant"))
    assert result["output_tokens"] == len(adapter.tokenizer.encode(completion))
    assert result["reasoning_tokens"] == 5
    assert result["peak_gpu_memory_allocated_bytes"] == 123
    assert adapter.tokenizer.template_args["reasoning_effort"] == "medium"
    assert adapter.tokenizer.template_args["chat_template"] == "fixed template"
    assert adapter.model.generation_args["eos_token_id"] == adapter.return_token_id
    assert adapter.count_messages_tokens(messages) == result["input_tokens"]


def test_config_selects_adapter_without_runner_branch(monkeypatch):
    path = Path("configs/standalone_textcraft_dev_gptoss20b_never.json")
    config = read_json(path)
    manifest = read_json(config["manifest"])
    validate_manifest(manifest)
    loaded = load_components(config)
    validate_config(config, manifest["benchmark"])
    assert "agent_support.gpt_oss_model" in loaded
    assert len(manifest["tasks"]) == 20
    sentinel = object()
    monkeypatch.setitem(MODEL_REGISTRY, "gpt_oss_20b", lambda _: sentinel)
    assert make_model(config) is sentinel


def test_comparison_configs_share_environment_and_disable_support():
    qwen = read_json("configs/standalone_textcraft_dev_qwen_never.json")
    gpt = read_json("configs/standalone_textcraft_dev_gptoss20b_never.json")
    for config in (qwen, gpt):
        load_components(config)
        validate_config(config, "textcraft")
        assert config["supports"] == []
        assert config["max_support_calls"] == 0
    for key in ("manifest", "partition_manifests", "seed", "history_window",
                "max_actions", "max_decisions", "max_model_calls", "selection_policy"):
        assert qwen[key] == gpt[key]
    assert qwen["max_new_tokens"] == 100 and gpt["max_new_tokens"] == 1024


class Episode:
    task_prompt = "task"
    observation = "observation"
    initial_observation = "observation"

    def reset(self):
        pass

    def step(self, action):
        return {"observation": "done", "done": True, "success": False,
                "failed_action": True}

    def close(self):
        pass


class InvalidFinalModel:
    def generate(self, messages):
        return {"text": "", "input_tokens": 10, "output_tokens": 12,
                "inference_seconds": 1.0, "reasoning_tokens": 8, "final_tokens": 0,
                "generation_status": "truncated_before_final",
                "generation_error": "truncated_before_final",
                "raw_completion": "<|channel|>analysis<|message|>...",
                "peak_gpu_memory_allocated_bytes": 123,
                "peak_gpu_memory_reserved_bytes": 456}


def test_failed_generation_is_audited_and_never_substituted():
    config = read_json("configs/standalone_textcraft_dev_gptoss20b_never.json")
    config["max_decisions"] = 1
    result = run_episode("textcraft", {"task_id": "mock"}, config, InvalidFinalModel(),
                         episode_factory=lambda *_: Episode())
    assert result["environment_actions"] == 0
    assert result["parser_errors"] == 1
    assert result["generation_failures"] == 1
    assert result["reasoning_tokens"] == 8
    assert result["peak_gpu_memory_allocated_bytes"] == 123
    assert result["trace"][0]["generation_status"] == "truncated_before_final"
