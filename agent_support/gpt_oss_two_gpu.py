"""Opt-in, explicit two-device placement for pinned GPT-OSS MXFP4."""

from __future__ import annotations

import os


SHARD_ENV = "AGENT_SUPPORT_GPTOSS_TWO_GPU"
VISIBLE_ENV = "AGENT_SUPPORT_GPTOSS_CUDA_VISIBLE_DEVICES"


def enabled() -> bool:
    return os.environ.get(SHARD_ENV) == "1"


def layer_device_map(layer_count: int) -> dict[str, int]:
    if type(layer_count) is not int or layer_count != 24:
        raise RuntimeError(f"Unsupported GPT-OSS layer count for two-GPU split: {layer_count!r}")
    middle = layer_count // 2
    return {"model.embed_tokens": 0, "model.rotary_emb": 0,
            **{f"model.layers.{index}": 0 if index < middle else 1
               for index in range(layer_count)},
            "model.norm": 1, "lm_head": 1}


def assert_two_gpu_runtime(torch_api) -> None:
    if torch_api.cuda.device_count() != 2:
        raise RuntimeError("GPT-OSS two-GPU runtime requires exactly two visible CUDA devices")
    for index in (0, 1):
        if "L40S" not in torch_api.cuda.get_device_name(index):
            raise RuntimeError(f"GPT-OSS shard device {index} is not an L40S")
        if torch_api.cuda.get_device_capability(index) < (7, 5):
            raise RuntimeError(f"GPT-OSS shard device {index} cannot run MXFP4")


def assert_device_map(model, expected: dict[str, int]) -> None:
    actual = getattr(model, "hf_device_map", None)
    if actual is None or set(actual) != set(expected):
        raise RuntimeError(f"GPT-OSS sharding unsupported: loaded device map differs: {actual!r}")
    for module, device in expected.items():
        value = actual[module]
        if str(value) not in (str(device), f"cuda:{device}"):
            raise RuntimeError(f"GPT-OSS sharding unsupported: {module} loaded on {value!r}")
