"""Pinned GPT-OSS-20B MXFP4 adapter for standalone local inference."""

from __future__ import annotations

import importlib.metadata
import hashlib
import random
import time

from .model import register_model


FINAL_MARKER = "<|channel|>final<|message|>"
ANALYSIS_MARKER = "<|channel|>analysis<|message|>"
RETURN_TOKEN = "<|return|>"
END_TOKEN = "<|end|>"


def _positions(tokens: list[int], pattern: list[int]) -> list[int]:
    if not pattern:
        raise ValueError("Harmony marker has no token IDs")
    return [index for index in range(len(tokens) - len(pattern) + 1)
            if tokens[index:index + len(pattern)] == pattern]


def parse_harmony_completion(tokenizer, generated_ids: list[int], limit: int) -> dict:
    """Expose only a completed final channel to the benchmark parser."""
    raw = tokenizer.decode(generated_ids, skip_special_tokens=False)
    final_marker = tokenizer.encode(FINAL_MARKER, add_special_tokens=False)
    analysis_marker = tokenizer.encode(ANALYSIS_MARKER, add_special_tokens=False)
    return_id = tokenizer.convert_tokens_to_ids(RETURN_TOKEN)
    end_id = tokenizer.convert_tokens_to_ids(END_TOKEN)
    final_positions = _positions(generated_ids, final_marker)
    analysis_positions = _positions(generated_ids, analysis_marker)
    reasoning_tokens = 0
    for start in analysis_positions:
        body_start = start + len(analysis_marker)
        body_end = next((index for index in range(body_start, len(generated_ids))
                         if generated_ids[index] == end_id), len(generated_ids))
        reasoning_tokens += body_end - body_start
    status = "complete"
    text = ""
    final_tokens = 0
    if len(final_positions) != 1:
        status = "malformed_harmony" if len(final_positions) > 1 else (
            "truncated_before_final" if len(generated_ids) >= limit else "missing_final")
    else:
        body_start = final_positions[0] + len(final_marker)
        return_positions = [index for index in range(body_start, len(generated_ids))
                            if generated_ids[index] == return_id]
        if len(return_positions) != 1 or return_positions[0] != len(generated_ids) - 1:
            status = "truncated_final" if len(generated_ids) >= limit else "malformed_final"
        else:
            body = generated_ids[body_start:return_positions[0]]
            final_tokens = len(body)
            text = tokenizer.decode(body, skip_special_tokens=False).strip()
            if not text or any(marker in text for marker in
                               ("<|start|>", "<|end|>", "<|channel|>", "<|message|>", RETURN_TOKEN)):
                status = "malformed_final"
                text = ""
    return {"text": text, "raw_completion": raw, "reasoning_tokens": reasoning_tokens,
            "final_tokens": final_tokens, "generation_status": status,
            "generation_error": None if status == "complete" else status}


class GptOss20BModel:
    def __init__(self, config: dict):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("GPT-OSS-20B requires a CUDA GPU")
        if torch.cuda.get_device_capability(0) < (7, 5):
            raise RuntimeError("MXFP4 kernels require CUDA compute capability at least 7.5")
        installed_backend = importlib.metadata.version("transformers")
        if installed_backend != config["backend_version"]:
            raise RuntimeError(f"Transformers {installed_backend} differs from configured "
                               f"backend_version {config['backend_version']}")
        seed = config["seed"]
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats(0)
        self.torch = torch
        self.model_id = config["model_id"]
        self.model_revision = config["model_revision"]
        self.max_new_tokens = config["max_new_tokens"]
        self.reasoning_effort = config["reasoning_effort"]
        self.temperature = config["temperature"]
        self.top_p = config["top_p"]
        self.top_k = config["top_k"]
        self.backend_version = installed_backend
        self.prompt_date = config["prompt_date"]
        revision = {"revision": self.model_revision}
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, **revision)
        template = self.tokenizer.chat_template
        date_expression = 'strftime_now("%Y-%m-%d")'
        if not isinstance(template, str) or template.count(date_expression) != 1:
            raise RuntimeError("Pinned GPT-OSS chat template has an unexpected date expression")
        self.chat_template = template.replace(date_expression, f'"{self.prompt_date}"')
        self.return_token_id = self.tokenizer.convert_tokens_to_ids(RETURN_TOKEN)
        if not isinstance(self.return_token_id, int) or self.return_token_id < 0:
            raise RuntimeError("GPT-OSS tokenizer lacks the Harmony final-return token")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, device_map={"": 0}, dtype="auto", low_cpu_mem_usage=True,
            **revision)
        self.model.eval()
        quantization = getattr(self.model.config, "quantization_config", {})
        method = (quantization.get("quant_method") if isinstance(quantization, dict)
                  else getattr(quantization, "quant_method", None))
        packed = [(name, parameter) for name, parameter in self.model.named_parameters()
                  if name.endswith(("gate_up_proj", "down_proj")) and
                  parameter.dtype == torch.uint8]
        if method != "mxfp4" or not packed:
            raise RuntimeError("GPT-OSS expert weights were not retained as packed MXFP4")
        if any(parameter.device.type != "cuda" for _, parameter in self.model.named_parameters()):
            raise RuntimeError("GPT-OSS model was partly offloaded from the requested GPU")
        self.packed_expert_parameters = len(packed)
        self.peak_gpu_memory_bytes = int(torch.cuda.max_memory_allocated(0))
        self.peak_gpu_reserved_bytes = int(torch.cuda.max_memory_reserved(0))

    def metadata(self) -> dict:
        return {"model_type": "gpt_oss_20b", "model_id": self.model_id,
                "model_revision": self.model_revision, "tokenizer_revision": self.model_revision,
                "backend": "transformers", "backend_version": self.backend_version,
                "quantization": "mxfp4_packed", "packed_expert_parameters": self.packed_expert_parameters,
                "chat_template": "pinned_tokenizer_harmony", "prompt_date": self.prompt_date,
                "chat_template_sha256": hashlib.sha256(self.chat_template.encode()).hexdigest(),
                "reasoning_effort": self.reasoning_effort,
                "decoding": {"do_sample": True, "temperature": self.temperature,
                             "top_p": self.top_p, "top_k": self.top_k,
                             "max_new_tokens": self.max_new_tokens},
                "gpu_name": self.torch.cuda.get_device_name(0),
                "gpu_compute_capability": list(self.torch.cuda.get_device_capability(0)),
                "gpu_total_memory_bytes": self.torch.cuda.get_device_properties(0).total_memory,
                "torch_version": self.torch.__version__,
                "torch_cuda_version": self.torch.version.cuda,
                "triton_version": importlib.metadata.version("triton"),
                "kernels_version": importlib.metadata.version("kernels"),
                "tokenizers_version": importlib.metadata.version("tokenizers"),
                "safetensors_version": importlib.metadata.version("safetensors"),
                "huggingface_hub_version": importlib.metadata.version("huggingface-hub")}

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _encode_messages(self, messages: list[dict], *, tensors: bool):
        return self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            reasoning_effort=self.reasoning_effort,
            chat_template=self.chat_template,
            **({"return_tensors": "pt", "return_dict": True} if tensors else {}))

    def count_messages_tokens(self, messages: list[dict]) -> int:
        return len(self._encode_messages(messages, tensors=False))

    def generate(self, messages: list[dict], max_new_tokens: int | None = None) -> dict:
        inputs = self._encode_messages(messages, tensors=True).to(self.model.device)
        input_ids = inputs["input_ids"]
        limit = self.max_new_tokens if max_new_tokens is None else max_new_tokens
        self.torch.cuda.synchronize(0)
        started = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs, do_sample=True, temperature=self.temperature,
                top_p=self.top_p, top_k=self.top_k, max_new_tokens=limit,
                eos_token_id=self.return_token_id,
                pad_token_id=self.tokenizer.pad_token_id)
        self.torch.cuda.synchronize(0)
        elapsed = round(time.perf_counter() - started, 3)
        generated_ids = output[0][input_ids.shape[1]:].tolist()
        parsed = parse_harmony_completion(self.tokenizer, generated_ids, limit)
        self.peak_gpu_memory_bytes = max(self.peak_gpu_memory_bytes,
                                         int(self.torch.cuda.max_memory_allocated(0)))
        self.peak_gpu_reserved_bytes = max(self.peak_gpu_reserved_bytes,
                                           int(self.torch.cuda.max_memory_reserved(0)))
        return {**parsed, "input_tokens": int(input_ids.shape[1]),
                "output_tokens": len(generated_ids), "inference_seconds": elapsed,
                "peak_gpu_memory_allocated_bytes": self.peak_gpu_memory_bytes,
                "peak_gpu_memory_reserved_bytes": self.peak_gpu_reserved_bytes}


register_model("gpt_oss_20b", GptOss20BModel)
