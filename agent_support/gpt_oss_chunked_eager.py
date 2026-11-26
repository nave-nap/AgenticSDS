"""Bound GPT-OSS eager attention memory by slicing only the query dimension.

The pinned eager function still computes each query's complete key range,
attention sink, mask, softmax, and value product. No checkpoint or policy input
is changed. Transformers' public attention and mask registries select it.
"""

from __future__ import annotations

import os


ENABLE_ENV = "AGENT_SUPPORT_GPTOSS_CHUNKED_EAGER"
NAME = "gptoss_chunked_eager_v1"
QUERY_CHUNK = 256


def enabled() -> bool:
    return os.environ.get(ENABLE_ENV) == "1"


def query_chunks(reference_forward, concatenate, module, query, key, value,
                 attention_mask, scaling, dropout=0.0, *, chunk_size=QUERY_CHUNK,
                 on_chunk=None, **kwargs):
    """Apply pinned eager math to full-key, query-sliced blocks."""
    q_length = query.shape[-2]
    if chunk_size <= 0:
        raise ValueError("Query chunk size must be positive")
    if attention_mask is None and q_length > 1:
        raise RuntimeError("Chunked GPT-OSS prefill requires the eager causal mask")
    if attention_mask is not None and (attention_mask.ndim != 4 or
                                       attention_mask.shape[-2] != q_length or
                                       attention_mask.shape[-1] != key.shape[-2]):
        raise RuntimeError("Chunked GPT-OSS received an incompatible eager mask")
    outputs = []
    for start in range(0, q_length, chunk_size):
        end = min(start + chunk_size, q_length)
        if on_chunk is not None:
            on_chunk("start", start, end)
        mask = None if attention_mask is None else attention_mask[..., start:end, :]
        try:
            output, weights = reference_forward(
                module, query[:, :, start:end, :], key, value, mask,
                scaling=scaling, dropout=dropout, **kwargs)
            del weights
        except Exception:
            if on_chunk is not None:
                on_chunk("error", start, end)
            raise
        outputs.append(output)
        if on_chunk is not None:
            on_chunk("end", start, end)
    return concatenate(outputs, dim=1), None


def register_chunked_eager(diagnostics=None) -> str:
    """Register the attention function and its matching eager mask formatter."""
    import torch
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.gpt_oss.modeling_gpt_oss import eager_attention_forward

    def forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        if attention_mask is not None and attention_mask.device != query.device:
            raise RuntimeError("GPT-OSS eager mask is not on the executing attention device")
        layer = getattr(module, "layer_idx", None)
        def on_chunk(event, start, end):
            if diagnostics is not None and (event == "error" or start == 0 or end == query.shape[-2]):
                diagnostics.write(event="attention_chunk_" + event, layer=layer,
                                  request_id=getattr(diagnostics, "current_request_id", None),
                                  query_start=start, query_end=end,
                                  query_tokens=query.shape[-2], key_tokens=key.shape[-2],
                                  execution_device=str(query.device), memory=diagnostics.snapshot())
        # A one-token decode has one chunk. Logging both edges for every layer
        # would add 48 memory snapshots and shared-filesystem writes per token.
        return query_chunks(eager_attention_forward, torch.cat, module, query, key,
                            value, attention_mask, scaling, dropout,
                            on_chunk=on_chunk if query.shape[-2] > 1 else None, **kwargs)

    ALL_ATTENTION_FUNCTIONS.register(NAME, forward)
    ALL_MASK_ATTENTION_FUNCTIONS.register(NAME, ALL_MASK_ATTENTION_FUNCTIONS["eager"])
    return NAME
