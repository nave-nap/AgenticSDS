"""Qwen 4-bit inference; imported only in GPU execution paths."""

from __future__ import annotations

import random
import time


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
        self.model = AutoModelForCausalLM.from_pretrained(
            config["model_id"], device_map="auto", quantization_config=quantization, **revision
        )
        self.model.eval()

    def generate(self, messages: list[dict], max_new_tokens: int | None = None) -> dict:
        input_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        input_ids = input_ids.to(next(self.model.parameters()).device)
        started = time.perf_counter()
        with self.torch.inference_mode():
            generated = self.model.generate(
                input_ids=input_ids,
                attention_mask=self.torch.ones_like(input_ids),
                do_sample=False,
                max_new_tokens=self.max_new_tokens if max_new_tokens is None else max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        output_ids = generated[0][input_ids.shape[1]:]
        return {
            "text": self.tokenizer.decode(output_ids, skip_special_tokens=True).strip(),
            "input_tokens": int(input_ids.shape[1]),
            "output_tokens": int(output_ids.shape[0]),
            "inference_seconds": round(time.perf_counter() - started, 3),
        }
