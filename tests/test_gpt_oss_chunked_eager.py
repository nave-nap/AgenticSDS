"""Offline checks for query-sliced GPT-OSS eager math and the GPU gate."""

from __future__ import annotations

import math
import io
import json
import sys
from types import ModuleType
from types import SimpleNamespace

import pytest

from agent_support.gpt_oss_chunked_eager import NAME, query_chunks, register_chunked_eager
from agent_support.gpt_oss_controller_worker import serve
from agent_support.gpt_oss_two_gpu import SHARD_ENV
from agent_support.three_gpu_feasibility import validate_prefill_execution


class Query:
    device = "cuda:0"
    def __init__(self, rows):
        self.rows = rows
        self.shape = (1, 1, len(rows), 1)
    def __getitem__(self, key):
        return Query(self.rows[key[2]])


class Mask:
    ndim = 4
    device = "cuda:0"
    def __init__(self, rows):
        self.rows = rows
        self.shape = (1, 1, len(rows), len(rows[0]))
    def __getitem__(self, key):
        return Mask(self.rows[key[-2]])


def reference_eager(module, query, key, value, mask, scaling, dropout=0, **_kwargs):
    assert dropout == 0
    results = []
    for q, mask_row in zip(query.rows, mask.rows):
        logits = [q * k * scaling + bias for k, bias in zip(key.rows, mask_row)]
        logits.append(module.sink)
        offset = max(logits)
        exp = [math.exp(score - offset) for score in logits]
        total = sum(exp)
        results.append(sum(weight * v for weight, v in zip(exp[:-1], value.rows)) / total)
    return SimpleNamespace(rows=results), None


def concatenate(outputs, dim):
    assert dim == 1
    return SimpleNamespace(rows=[value for output in outputs for value in output.rows])


@pytest.mark.parametrize("window", [None, 2])
def test_query_chunks_preserve_full_and_sliding_attention_with_sink(window):
    query = Query([0.1, -0.4, 0.2, 0.8, -0.1, 0.3, 0.7])
    key = Query([0.2, 0.5, -0.3, 0.6, -0.2, 0.1, 0.9])
    value = SimpleNamespace(rows=[2.0, -1.0, 3.0, 0.5, -0.5, 1.2, 4.0])
    mask = Mask([[0.0 if j <= i and (window is None or j > i - window) else -1e30
                  for j in range(7)] for i in range(7)])
    module = SimpleNamespace(sink=-0.3)
    full, _ = reference_eager(module, query, key, value, mask, scaling=0.5)
    events = []
    chunked, weights = query_chunks(reference_eager, concatenate, module, query, key,
                                    value, mask, scaling=0.5, chunk_size=3,
                                    on_chunk=lambda *args: events.append(args))
    assert chunked.rows == pytest.approx(full.rows, abs=1e-12)
    assert weights is None
    assert [event[0] for event in events] == ["start", "end"] * 3


def test_query_chunks_reject_missing_or_wrong_prefill_mask():
    query = Query([0.1, 0.2])
    key = Query([0.3, 0.4])
    value = SimpleNamespace(rows=[1, 2])
    with pytest.raises(RuntimeError, match="requires the eager causal mask"):
        query_chunks(reference_eager, concatenate, SimpleNamespace(sink=0),
                     query, key, value, None, scaling=1.0)
    with pytest.raises(RuntimeError, match="incompatible eager mask"):
        query_chunks(reference_eager, concatenate, SimpleNamespace(sink=0),
                     query, key, value, Mask([[0, 0]]), scaling=1.0)


def test_long_prefill_gate_requires_all_layers_on_expected_devices():
    rows = []
    for index in range(24):
        device = f"cuda:{0 if index < 12 else 1}"
        rows += [{"event": event, "layer": index, "query_tokens": 8887,
                  "execution_device": device} for event in ("layer_q_proj_enter", "layer_complete")]
    rows += [{"event": "attention_chunk_end", "layer": index, "query_tokens": 8887}
             for index in (0, 23)]
    assert validate_prefill_execution(rows, 8887)["prefill_layers_completed"] == 24
    rows[25]["execution_device"] = "cuda:0"  # dispatch may return a completed layer to the caller device
    assert validate_prefill_execution(rows, 8887)["prefill_layers_completed"] == 24
    with pytest.raises(RuntimeError, match="did not execute all 24"):
        validate_prefill_execution(rows[:-3], 8887)
    rows[24]["execution_device"] = "cuda:0"  # layer 12 must execute on cuda:1
    with pytest.raises(RuntimeError, match="wrong GPU"):
        validate_prefill_execution(rows, 8887)


def test_backend_registers_matching_eager_mask(monkeypatch):
    class Registry(dict):
        def register(self, name, value):
            self[name] = value
    attention = Registry()
    masks = Registry(eager=object())
    modules = {
        "torch": {"cat": concatenate},
        "transformers": {},
        "transformers.masking_utils": {"ALL_MASK_ATTENTION_FUNCTIONS": masks},
        "transformers.modeling_utils": {"ALL_ATTENTION_FUNCTIONS": attention},
        "transformers.models": {},
        "transformers.models.gpt_oss": {},
        "transformers.models.gpt_oss.modeling_gpt_oss": {"eager_attention_forward": reference_eager},
    }
    for name, attrs in modules.items():
        fake = ModuleType(name)
        fake.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, fake)
    assert register_chunked_eager() == NAME
    assert callable(attention[NAME]) and masks[NAME] is masks["eager"]
    events = []
    diagnostics = SimpleNamespace(write=lambda **row: events.append(row),
                                  snapshot=lambda: {}, current_request_id=7)
    register_chunked_eager(diagnostics)
    module = SimpleNamespace(sink=0.0, layer_idx=0)
    attention[NAME](module, Query([0.1]), Query([0.2]),
                    SimpleNamespace(rows=[1.0]), Mask([[0.0]]), scaling=1.0)
    assert events == []  # decode telemetry must stay bounded per token
    attention[NAME](module, Query([0.1, 0.2]), Query([0.2, 0.3]),
                    SimpleNamespace(rows=[1.0, 2.0]), Mask([[0.0, -1e30], [0.0, 0.0]]), scaling=1.0)
    assert [row["event"] for row in events] == ["attention_chunk_start", "attention_chunk_end"]
    assert {row["request_id"] for row in events} == {7}


def test_two_gpu_worker_prints_full_generation_traceback(monkeypatch, capsys):
    monkeypatch.setenv(SHARD_ENV, "1")
    class Model:
        def __init__(self, _config):
            pass
        def metadata(self):
            return {}
        def generate(self, *_args, **_kwargs):
            raise RuntimeError("attention allocation failed")
    request = "\n".join((json.dumps({"op": "init", "config": {}}),
                         json.dumps({"op": "generate", "messages": []}), ""))
    output = io.StringIO()
    serve(source=io.StringIO(request), sink=output, model_factory=Model)
    trace = capsys.readouterr().err
    assert "Traceback (most recent call last):" in trace
    assert "in generate" in trace and "attention allocation failed" in trace
    assert "attention allocation failed" in output.getvalue()
