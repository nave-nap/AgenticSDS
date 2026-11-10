"""Qwen's existing greedy generation and token accounting remain intact."""

from contextlib import nullcontext
from types import SimpleNamespace

from agent_support.model import QwenModel


class Tensor:
    def __init__(self, values):
        self.values = values
        self.shape = (len(values), len(values[0])) if values and isinstance(values[0], list) else (len(values),)
    def to(self, device):
        return self
    def __getitem__(self, index):
        return Tensor(self.values[index])


class Tokenizer:
    eos_token_id = 99
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, return_tensors=None):
        assert tokenize and add_generation_prompt
        return Tensor([[1, 2]]) if return_tensors else [1, 2]
    def decode(self, output_ids, skip_special_tokens):
        assert skip_special_tokens and output_ids.values == [3, 4]
        return "Action: inventory"
    def encode(self, text, add_special_tokens):
        return [1, 2, 3]


class Backend:
    def __init__(self):
        self.calls = []
    def parameters(self):
        yield SimpleNamespace(device="cuda:0")
    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return Tensor([[1, 2, 3, 4]])


def test_qwen_adapter_keeps_chat_template_greedy_decoding_and_usage():
    backend = Backend()
    model = QwenModel.__new__(QwenModel)
    model.tokenizer = Tokenizer()
    model.model = backend
    model.torch = SimpleNamespace(inference_mode=nullcontext,
                                  ones_like=lambda tensor: Tensor([[1, 1]]),
                                  cuda=SimpleNamespace(get_device_name=lambda: "mock GPU"),
                                  version=SimpleNamespace(cuda="12.8"))
    model.max_new_tokens = 32
    model.model_id = "Qwen/Qwen2.5-7B-Instruct"
    model.model_revision = "a" * 40
    result = model.generate([{"role": "user", "content": "Choose."}])
    assert result["text"] == "Action: inventory"
    assert result["input_tokens"] == 2 and result["output_tokens"] == 2
    assert backend.calls[0]["do_sample"] is False
    assert backend.calls[0]["max_new_tokens"] == 32
    assert backend.calls[0]["pad_token_id"] == 99
    model.generate([{"role": "user", "content": "Choose."}], max_new_tokens=5)
    assert backend.calls[1]["max_new_tokens"] == 5
    assert model.count_messages_tokens([]) == 2
    assert model.metadata()["model_revision"] == "a" * 40
