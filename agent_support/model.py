"""Qwen 4-bit inference; imported only in GPU execution paths."""

from __future__ import annotations

import os
import random
import time

from .gpu_memory_diagnostics import GpuMemoryDiagnostics, enabled


class QwenModel:
    def __init__(self, config: dict):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if not torch.cuda.is_available():
            raise RuntimeError("Qwen baseline requires a CUDA GPU; use Colab GPU runtime")
        seed = int(config.get("seed", 42))
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.torch = torch
        self.max_new_tokens = config["max_new_tokens"]
        revision = {"revision": config["model_revision"]} if config.get("model_revision") else {}
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_id"], **revision)
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        qwen_gpu = os.environ.get("AGENT_SUPPORT_QWEN_GPU")
        if qwen_gpu not in (None, "0"):
            raise ValueError("AGENT_SUPPORT_QWEN_GPU must be 0 when set")
        self.model = AutoModelForCausalLM.from_pretrained(
            config["model_id"], device_map={"": 0} if qwen_gpu == "0" else "auto",
            quantization_config=quantization, **revision
        )
        self.model.eval()
        self.memory_diagnostics = None
        self._diagnostic_request_id = 0

    def enable_gpu_memory_diagnostics(self, directory) -> None:
        if not enabled():
            return
        self.memory_diagnostics = GpuMemoryDiagnostics(directory, "qwen_cuda_memory", self.torch)
        try:
            cuda = self.torch.cuda
            self.memory_diagnostics.write(
                event="model_loaded", cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                visible_cuda_devices=[{"index": index, "name": cuda.get_device_name(index)}
                                      for index in range(cuda.device_count())],
                hf_device_map={str(key): str(value) for key, value in
                               getattr(self.model, "hf_device_map", {}).items()},
                memory=self.memory_diagnostics.snapshot())
        except Exception:
            pass

    def generate(self, messages: list[dict]) -> dict:
        diagnostics = self.memory_diagnostics
        request_id = None
        if diagnostics is not None:
            self._diagnostic_request_id += 1
            request_id = self._diagnostic_request_id
            diagnostics.start(request_id, self.max_new_tokens)
        input_count = None
        error_type = None
        try:
            input_ids = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            )
            input_ids = input_ids.to(next(self.model.parameters()).device)
            input_count = int(input_ids.shape[1])
            started = time.perf_counter()
            with self.torch.inference_mode():
                generated = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=self.torch.ones_like(input_ids),
                    do_sample=False,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            output_ids = generated[0][input_ids.shape[1]:]
            return {
                "text": self.tokenizer.decode(output_ids, skip_special_tokens=True).strip(),
                "input_tokens": input_count,
                "output_tokens": int(output_ids.shape[0]),
                "inference_seconds": round(time.perf_counter() - started, 3),
            }
        except Exception as exc:
            error_type = type(exc).__name__
            raise
        finally:
            if diagnostics is not None:
                diagnostics.finish(request_id, input_count, self.max_new_tokens, error_type)
