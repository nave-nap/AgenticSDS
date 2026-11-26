"""Pinned GPT-OSS-20B MXFP4 adapter for the isolated controller worker.

Reused from the repository's delegation_token_reservation_fix branch at
4ec819ce0f1808336aef6f01d053d25681796220. Historical runners are unchanged.
"""

from __future__ import annotations

import importlib.metadata
import hashlib
import os
import random
import re
import time

from packaging.version import Version

from .gpu_memory_diagnostics import (DIRECTORY_ENV, GpuMemoryDiagnostics,
                                     TwoGpuMemoryDiagnostics, enabled)
from .gpt_oss_two_gpu import (assert_device_map, assert_two_gpu_runtime,
                              enabled as two_gpu_enabled, layer_device_map)
from .gpt_oss_chunked_eager import enabled as chunked_eager_enabled, register_chunked_eager
from .gpt_oss_load_diagnostics import monitor as model_load_monitor



FINAL_MARKER = "<|channel|>final<|message|>"
ANALYSIS_MARKER = "<|channel|>analysis<|message|>"
RETURN_TOKEN = "<|return|>"
END_TOKEN = "<|end|>"
CONSTRAIN_TOKEN = "<|constrain|>"


_FINAL_MESSAGE_V3 = re.compile(
    r"\A(?:<\|channel\|>analysis<\|message\|>(?P<analysis>.*?)<\|end\|>)?"
    r"<\|start\|>assistant<\|channel\|>final"
    r"(?P<format> ?<\|constrain\|>(?:json|JSON|schema))?"
    r"<\|message\|>(?P<body>.*?)<\|return\|>\Z", re.DOTALL)
_STRUCTURAL_TOKEN = re.compile(r"<\|[^<>]{1,64}\|>")


def check_mxfp4_dependencies(backend_version: str, version_lookup=importlib.metadata.version) -> dict:
    """Reject this pinned stack before Transformers can silently dequantize."""
    try:
        versions = {name: version_lookup(name) for name in
                    ("transformers", "kernels", "triton", "accelerate")}
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"GPT-OSS MXFP4 dependency missing: {exc.name}") from exc
    if versions["transformers"] != backend_version:
        raise RuntimeError(f"Transformers {versions['transformers']} differs from configured "
                           f"backend_version {backend_version}")
    if versions["kernels"] != "0.17.2":
        raise RuntimeError(f"Transformers {backend_version} MXFP4 requires the pinned "
                           f"kernels==0.17.2; found kernels=={versions['kernels']}")
    if Version(versions["triton"]) < Version("3.4.0"):
        raise RuntimeError(f"GPT-OSS MXFP4 on CUDA requires Triton >=3.4.0; "
                           f"found {versions['triton']}")
    return versions


def verify_packed_mxfp4(model, torch_api, hub, expert_type, cuda_index: int = 0,
                        device_map: dict[str, int] | None = None) -> dict:
    """Inspect the actual Transformers 5.22.2 Triton representation on every layer."""
    quantization = getattr(model.config, "quantization_config", None)
    method = (quantization.get("quant_method") if isinstance(quantization, dict)
              else getattr(quantization, "quant_method", None))
    dequantize = (quantization.get("dequantize", False) if isinstance(quantization, dict)
                  else getattr(quantization, "dequantize", False))
    if method != "mxfp4" or dequantize:
        raise RuntimeError(f"GPT-OSS MXFP4 disabled: quant_method={method!r}, dequantize={dequantize!r}")
    if hub is None:
        raise RuntimeError("GPT-OSS MXFP4 Triton kernel module was not loaded")
    tensor_type = hub.tensor.Tensor
    fp4_dtype = hub.tensor.FP4
    layouts = hub.tensor_details.layout
    weight_layout_type, _ = layouts.make_default_matmul_mxfp4_w_layout(mx_axis=1)
    scale_layout_type = layouts.StridedLayout
    expected_device = torch_api.device("cuda", cuda_index)
    layer_count = getattr(model.config, "num_hidden_layers", None)
    if type(layer_count) is not int or layer_count <= 0:
        raise RuntimeError("GPT-OSS config has no positive num_hidden_layers")
    expected = {f"model.layers.{index}.mlp.experts" for index in range(layer_count)}
    actual = {name: module for name, module in model.named_modules()
              if isinstance(module, expert_type)}
    if set(actual) != expected:
        missing = sorted(expected - set(actual))
        extra = sorted(set(actual) - expected)
        raise RuntimeError(f"GPT-OSS MXFP4 expert-module mismatch: missing={missing}, extra={extra}")

    packed_bytes = scale_bytes = 0
    expert_counts = {"cuda:0": 0, "cuda:1": 0} if device_map else {}
    packed_bytes_by_device = {"cuda:0": 0, "cuda:1": 0} if device_map else {}
    scale_bytes_by_device = {"cuda:0": 0, "cuda:1": 0} if device_map else {}
    for name in sorted(expected):
        module = actual[name]
        module_device = (torch_api.device("cuda", device_map[name.rsplit(".mlp.experts", 1)[0]])
                         if device_map else expected_device)
        if device_map:
            expert_counts[str(module_device)] += 1
        if getattr(module, "_is_hf_initialized", False) is not True:
            raise RuntimeError(f"{name}: MXFP4 expert module is not initialized")
        experts = getattr(module, "num_experts", None)
        hidden = getattr(module, "hidden_size", None)
        intermediate = getattr(module, "intermediate_size", None)
        if any(type(value) is not int or value <= 0 for value in (experts, hidden, intermediate)):
            raise RuntimeError(f"{name}: invalid expert dimensions")
        for projection, logical_shape in (
                ("gate_up_proj", (experts, hidden, intermediate * 2)),
                ("down_proj", (experts, intermediate, hidden))):
            location = f"{name}.{projection}"
            if projection in getattr(module, "_parameters", {}):
                raise RuntimeError(f"{location}: remains an nn.Parameter, not a Triton FP4 tensor")
            weight = getattr(module, projection, None)
            if not isinstance(weight, tensor_type):
                raise RuntimeError(f"{location}: missing Triton packed projection")
            if weight.dtype != fp4_dtype or tuple(weight.shape) != logical_shape:
                raise RuntimeError(f"{location}: expected FP4 logical tensor shape {logical_shape}")
            storage = getattr(weight, "storage", None)
            backing = getattr(storage, "data", None)
            if not isinstance(backing, torch_api.Tensor) or backing.dtype != torch_api.uint8:
                raise RuntimeError(f"{location}: packed backing storage must be torch.uint8")
            if not isinstance(getattr(storage, "layout", None), weight_layout_type):
                raise RuntimeError(f"{location}: wrong MXFP4 weight storage layout")
            if backing.is_meta or backing.device != module_device or backing.numel() <= 0:
                raise RuntimeError(f"{location}: packed backing storage is uninitialized or not on {module_device}")
            logical_elements = experts * logical_shape[1] * logical_shape[2]
            if (isinstance(storage.layout, scale_layout_type)
                    and backing.numel() * 2 != logical_elements):
                raise RuntimeError(f"{location}: packed backing storage has the wrong byte count")
            if backing.numel() * 2 < logical_elements:
                raise RuntimeError(f"{location}: packed backing storage is incomplete")

            precision = getattr(module, f"{projection}_precision_config", None)
            scale = getattr(precision, "weight_scale", None)
            if not isinstance(scale, tensor_type):
                raise RuntimeError(f"{location}: missing MXFP4 scale tensor")
            expected_scale_shape = (experts, logical_shape[1] // 32, logical_shape[2])
            if logical_shape[1] % 32 or tuple(scale.shape) != expected_scale_shape:
                raise RuntimeError(f"{location}: wrong MXFP4 scale shape; expected {expected_scale_shape}")
            scale_storage = getattr(scale, "storage", None)
            scale_backing = getattr(scale_storage, "data", None)
            if (scale.dtype != torch_api.uint8 or not isinstance(scale_backing, torch_api.Tensor)
                    or scale_backing.dtype != torch_api.uint8):
                raise RuntimeError(f"{location}: MXFP4 scale storage must be torch.uint8")
            if not isinstance(getattr(scale_storage, "layout", None), scale_layout_type):
                raise RuntimeError(f"{location}: wrong MXFP4 scale storage layout")
            if (scale_backing.is_meta or scale_backing.device != module_device
                    or scale_backing.numel() != logical_elements // 32):
                raise RuntimeError(f"{location}: scale storage is uninitialized, incomplete, or not on {module_device}")
            packed_bytes += backing.numel() * backing.element_size()
            scale_bytes += scale_backing.numel() * scale_backing.element_size()
            if device_map:
                packed_bytes_by_device[str(module_device)] += backing.numel() * backing.element_size()
                scale_bytes_by_device[str(module_device)] += scale_backing.numel() * scale_backing.element_size()

    parameters = list(model.named_parameters())
    if not parameters:
        raise RuntimeError("GPT-OSS has no ordinary model parameters")
    for name, parameter in parameters:
        matching = [prefix for prefix in device_map or {} if name == prefix or name.startswith(prefix + ".")]
        if device_map and not matching:
            raise RuntimeError(f"{name}: ordinary model parameter has no assigned shard")
        parameter_device = (torch_api.device("cuda", device_map[max(matching, key=len)])
                            if device_map else expected_device)
        if parameter.is_meta or parameter.device != parameter_device:
            raise RuntimeError(f"{name}: ordinary model parameter is uninitialized or not on {parameter_device}")
    if device_map and (expert_counts != {"cuda:0": layer_count // 2, "cuda:1": layer_count // 2} or
                       not all(packed_bytes_by_device.values()) or
                       not all(scale_bytes_by_device.values())):
        raise RuntimeError("GPT-OSS packed experts did not occupy both GPUs")
    return {"quantization_method": "mxfp4", "verified_expert_modules": len(expected),
            "verified_packed_projections": len(expected) * 2,
            "representation_type": f"{tensor_type.__module__}.{tensor_type.__qualname__}",
            "weight_layout_type": weight_layout_type.__name__,
            "scale_layout_type": scale_layout_type.__name__,
            "weight_storage_dtype": str(torch_api.uint8),
            "scale_storage_dtype": str(torch_api.uint8),
            "backing_storage_device": "multi" if device_map else str(expected_device),
            "packed_weight_storage_bytes": packed_bytes,
            "scale_storage_bytes": scale_bytes,
            **({"verified_expert_modules_by_device": expert_counts,
                "packed_weight_storage_bytes_by_device": packed_bytes_by_device,
                "scale_storage_bytes_by_device": scale_bytes_by_device} if device_map else {})}


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


def extract_decoded_final_v3(raw: str) -> dict | None:
    """Read only a fully framed final message from decoded Harmony output.

    This also supports read-only replay of saved completions without loading
    the GPU tokenizer. Model-time acceptance additionally checks token IDs.
    """
    if not isinstance(raw, str):
        return None
    match = _FINAL_MESSAGE_V3.fullmatch(raw)
    if match is None or _STRUCTURAL_TOKEN.search(match.group("analysis") or ""):
        return None
    body = match.group("body")
    if not body.strip() or _STRUCTURAL_TOKEN.search(body):
        return None
    label = match.group("format")
    return {"text": body.strip(), "final_channel_format": (
        "canonical" if label is None else "constrain_" + label.split(CONSTRAIN_TOKEN, 1)[1])}


def parse_harmony_completion_v3(tokenizer, generated_ids: list[int], limit: int) -> dict:
    """Accept observed constrained final headers with exact Harmony boundaries."""
    parsed = parse_harmony_completion(tokenizer, generated_ids, limit)
    if parsed["generation_status"] == "complete":
        return {**parsed, "final_channel_format": "canonical", "recovered_final_channel": False}
    if parsed["generation_status"] not in {"missing_final", "truncated_before_final"}:
        return {**parsed, "final_channel_format": None, "recovered_final_channel": False}
    decoded = extract_decoded_final_v3(parsed["raw_completion"])
    if decoded and decoded["final_channel_format"] != "canonical":
        ids = {name: tokenizer.convert_tokens_to_ids(name) for name in
               ("<|start|>", "<|channel|>", "<|message|>", CONSTRAIN_TOKEN,
                END_TOKEN, RETURN_TOKEN)}
        has_analysis = parsed["raw_completion"].startswith(ANALYSIS_MARKER)
        expected = {"<|start|>": 1, "<|channel|>": 2 if has_analysis else 1,
                    "<|message|>": 2 if has_analysis else 1,
                    CONSTRAIN_TOKEN: 1, END_TOKEN: 1 if has_analysis else 0,
                    RETURN_TOKEN: 1}
        if (all(type(ids[name]) is int and generated_ids.count(ids[name]) == count
                for name, count in expected.items()) and
                generated_ids[-1] == ids[RETURN_TOKEN]):
            message_positions = [index for index, token in enumerate(generated_ids)
                                 if token == ids["<|message|>"]]
            body_ids = generated_ids[message_positions[-1] + 1:-1]
            if tokenizer.decode(body_ids, skip_special_tokens=False).strip() == decoded["text"]:
                return {**parsed, "text": decoded["text"], "final_tokens": len(body_ids),
                        "generation_status": "complete", "generation_error": None,
                        "final_channel_format": decoded["final_channel_format"],
                        "recovered_final_channel": True}
    status = ("malformed_final_header" if CONSTRAIN_TOKEN in parsed["raw_completion"] and
              "<|channel|>final" in parsed["raw_completion"] else parsed["generation_status"])
    return {**parsed, "generation_status": status, "generation_error": status,
            "final_channel_format": None, "recovered_final_channel": False}


def parse_configured_completion(tokenizer, generated_ids: list[int], limit: int,
                                completion_parser: str) -> dict:
    if completion_parser == "strict_final_v1":
        return parse_harmony_completion(tokenizer, generated_ids, limit)
    if completion_parser == "harmony_final_v3":
        return parse_harmony_completion_v3(tokenizer, generated_ids, limit)
    raise ValueError("Unknown GPT-OSS completion parser")


class GptOss20BModel:
    def __init__(self, config: dict):
        dependencies = check_mxfp4_dependencies(config["backend_version"])
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("GPT-OSS-20B requires a CUDA GPU")
        if torch.cuda.get_device_capability(0) < (7, 5):
            raise RuntimeError("MXFP4 kernels require CUDA compute capability at least 7.5")
        self.two_gpu = two_gpu_enabled()
        self.chunked_eager = self.two_gpu and chunked_eager_enabled()
        if self.two_gpu:
            assert_two_gpu_runtime(torch)
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
        self.completion_parser = config.get("completion_parser", "strict_final_v1")
        if self.completion_parser not in {"strict_final_v1", "harmony_final_v3"}:
            raise ValueError("Unknown GPT-OSS completion parser")
        self.reasoning_effort = config["reasoning_effort"]
        self.temperature = config["temperature"]
        self.top_p = config["top_p"]
        self.top_k = config["top_k"]
        self.backend_version = dependencies["transformers"]
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
        directory = os.environ.get(DIRECTORY_ENV) if enabled() else None
        if self.two_gpu and not directory:
            raise RuntimeError("GPT-OSS two-GPU runtime requires a diagnostics directory")
        shard_diagnostics = (TwoGpuMemoryDiagnostics(directory, "gpt_oss_cuda_memory", torch)
                             if self.two_gpu else None)
        if shard_diagnostics is not None:
            if shard_diagnostics.path is None:
                raise RuntimeError("GPT-OSS two-GPU diagnostics could not create a telemetry file")
            for index in (0, 1):
                torch.cuda.reset_peak_memory_stats(index)
            shard_diagnostics.write(event="model_load_start", memory=shard_diagnostics.snapshot())
        device_map = None
        if self.two_gpu:
            model_config = AutoConfig.from_pretrained(self.model_id, **revision)
            device_map = layer_device_map(model_config.num_hidden_layers)
        attention = (register_chunked_eager(shard_diagnostics) if self.chunked_eager else "eager")
        try:
            with model_load_monitor(directory, torch):
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id, device_map=device_map or {"": 0}, dtype="auto",
                    low_cpu_mem_usage=True,
                    **({"attn_implementation": attention} if self.two_gpu else {}), **revision)
        except Exception as exc:
            if self.two_gpu:
                shard_diagnostics.write(event="model_load_error", error_type=type(exc).__name__,
                                        error_detail=str(exc), memory=shard_diagnostics.snapshot())
                raise RuntimeError(f"Pinned GPT-OSS MXFP4 explicit layer sharding unsupported: {exc}") from exc
            raise
        self.model.eval()
        try:
            if self.two_gpu:
                assert_device_map(self.model, device_map)
                if self.model.config._attn_implementation != attention:
                    raise RuntimeError("GPT-OSS two-GPU runtime selected the wrong attention backend")
            from transformers.integrations import mxfp4 as mxfp4_integration
            self.quantization_verification = verify_packed_mxfp4(
                self.model, torch, getattr(mxfp4_integration, "triton_kernels_hub", None),
                mxfp4_integration.Mxfp4GptOssExperts, cuda_index=0,
                device_map=device_map)
        except Exception as exc:
            if shard_diagnostics is not None:
                shard_diagnostics.write(event="model_validation_error", error_type=type(exc).__name__,
                                        error_detail=str(exc), memory=shard_diagnostics.snapshot())
            raise
        self.peak_gpu_memory_bytes = int(torch.cuda.max_memory_allocated(0))
        self.peak_gpu_reserved_bytes = int(torch.cuda.max_memory_reserved(0))
        self.memory_diagnostics = (shard_diagnostics if self.two_gpu else
                                   GpuMemoryDiagnostics(directory, "gpt_oss_cuda_memory", torch)
                                   if directory else None)
        if self.two_gpu:
            for index, layer in enumerate(self.model.model.layers):
                expected_device = f"cuda:{0 if index < 12 else 1}"
                def before_projection(_module, args, *, layer_index=index, expected=expected_device):
                    inputs = args[0]
                    if inputs.shape[-2] > 1:
                        observed = str(inputs.device)
                        self.memory_diagnostics.write(event="layer_q_proj_enter", layer=layer_index,
                                                      request_id=self._diagnostic_request_id,
                                                      query_tokens=inputs.shape[-2], execution_device=observed,
                                                      memory=self.memory_diagnostics.snapshot())
                        if observed != expected:
                            raise RuntimeError(f"GPT-OSS layer {layer_index} executed on {observed}, expected {expected}")
                def after_layer(_module, _args, output, *, layer_index=index):
                    if output.shape[-2] > 1:
                        observed = str(output.device)
                        self.memory_diagnostics.write(event="layer_complete", layer=layer_index,
                                                      request_id=self._diagnostic_request_id,
                                                      query_tokens=output.shape[-2], execution_device=observed,
                                                      memory=self.memory_diagnostics.snapshot())
                layer.self_attn.q_proj.register_forward_pre_hook(before_projection)
                layer.register_forward_hook(after_layer)
            self.memory_diagnostics.write(
                event="model_loaded", cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                hf_device_map={key: str(value) for key, value in self.model.hf_device_map.items()},
                memory=self.memory_diagnostics.snapshot(),
                packed_experts=self.quantization_verification)
        self._diagnostic_request_id = 0

    def metadata(self) -> dict:
        return {"model_type": "gpt_oss_20b", "model_id": self.model_id,
                "model_revision": self.model_revision, "tokenizer_revision": self.model_revision,
                "backend": "transformers", "backend_version": self.backend_version,
                "quantization": "mxfp4_packed", **self.quantization_verification,
                "chat_template": "pinned_tokenizer_harmony", "prompt_date": self.prompt_date,
                "chat_template_sha256": hashlib.sha256(self.chat_template.encode()).hexdigest(),
                "completion_parser": self.completion_parser,
                "reasoning_effort": self.reasoning_effort,
                "decoding": {"do_sample": True, "temperature": self.temperature,
                             "top_p": self.top_p, "top_k": self.top_k,
                             "max_new_tokens": self.max_new_tokens},
                "gpu_name": self.torch.cuda.get_device_name(0),
                **({"sharding": "explicit_decoder_layers_12_12",
                    "hf_device_map": {key: str(value) for key, value in self.model.hf_device_map.items()},
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "gpu_names": [self.torch.cuda.get_device_name(index) for index in (0, 1)],
                    "attention_implementation": self.model.config._attn_implementation}
                   if self.two_gpu else {}),
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
        inputs = self._encode_messages(messages, tensors=True)
        return int(inputs["input_ids"].shape[1])

    def release_cuda_cache(self) -> None:
        """Release unused allocator blocks after a completed worker request."""
        diagnostics = self.memory_diagnostics
        before = diagnostics.snapshot() if diagnostics is not None else None
        error_type = None
        try:
            self.torch.cuda.empty_cache()
        except Exception as exc:
            error_type = type(exc).__name__
        finally:
            if diagnostics is not None:
                diagnostics.write(event="cache_release", request_id=self._diagnostic_request_id,
                                  status="error" if error_type else "ok", error_type=error_type,
                                  memory_before=before, memory_after=diagnostics.snapshot())

    def generate(self, messages: list[dict], max_new_tokens: int | None = None) -> dict:
        limit = self.max_new_tokens if max_new_tokens is None else max_new_tokens
        diagnostics = self.memory_diagnostics
        request_id = None
        if diagnostics is not None:
            self._diagnostic_request_id += 1
            request_id = self._diagnostic_request_id
            diagnostics.current_request_id = request_id
            before = diagnostics.start(request_id, limit)
            # Keep the historical since-load high-water fields despite per-request resets.
            self.peak_gpu_memory_bytes = max(self.peak_gpu_memory_bytes,
                                             before.get("peak_allocated_bytes", 0))
            self.peak_gpu_reserved_bytes = max(self.peak_gpu_reserved_bytes,
                                               before.get("peak_reserved_bytes", 0))
        input_count = None
        error_type = None
        try:
            inputs = self._encode_messages(messages, tensors=True).to(self.model.device)
            input_ids = inputs["input_ids"]
            input_count = int(input_ids.shape[1])
            self.torch.cuda.synchronize(0)
            if getattr(self, "two_gpu", False):
                self.torch.cuda.synchronize(1)
            started = time.perf_counter()
            with self.torch.inference_mode():
                output = self.model.generate(
                    **inputs, do_sample=True, temperature=self.temperature,
                    top_p=self.top_p, top_k=self.top_k, max_new_tokens=limit,
                    eos_token_id=self.return_token_id,
                    pad_token_id=self.tokenizer.pad_token_id)
            self.torch.cuda.synchronize(0)
            if getattr(self, "two_gpu", False):
                self.torch.cuda.synchronize(1)
            elapsed = round(time.perf_counter() - started, 3)
            generated_ids = output[0][input_ids.shape[1]:].tolist()
            parsed = parse_configured_completion(self.tokenizer, generated_ids, limit,
                                                 self.completion_parser)
            self.peak_gpu_memory_bytes = max(self.peak_gpu_memory_bytes,
                                             int(self.torch.cuda.max_memory_allocated(0)))
            self.peak_gpu_reserved_bytes = max(self.peak_gpu_reserved_bytes,
                                               int(self.torch.cuda.max_memory_reserved(0)))
            return {**parsed, "input_tokens": input_count,
                    "output_tokens": len(generated_ids), "inference_seconds": elapsed,
                    "generation_prompt_tail": self.tokenizer.decode(input_ids[0][-32:].tolist(),
                                                                      skip_special_tokens=False),
                    "last_generated_token_id": generated_ids[-1] if generated_ids else None,
                    "stopped_by_final_return": bool(generated_ids and generated_ids[-1] == self.return_token_id),
                    "hit_token_limit": len(generated_ids) >= limit,
                    "peak_gpu_memory_allocated_bytes": self.peak_gpu_memory_bytes,
                    "peak_gpu_memory_reserved_bytes": self.peak_gpu_reserved_bytes}
        except Exception as exc:
            error_type = type(exc).__name__
            error_detail = str(exc)
            raise
        finally:
            if diagnostics is not None:
                diagnostics.finish(request_id, input_count, limit, error_type,
                                   error_detail if error_type else None)
