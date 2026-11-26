"""CPU-only checks for the bounded prefill latency and memory gate."""

from datetime import datetime, timedelta, timezone

import pytest

from agent_support.three_gpu_feasibility import MIN_FREE_BYTES
from agent_support.three_gpu_performance_probe import validate_stage


def probe_rows():
    origin = datetime(2026, 11, 26, tzinfo=timezone.utc)
    def row(event, seconds, **values):
        return {"event": event, "request_id": 1,
                "timestamp_utc": (origin + timedelta(seconds=seconds)).isoformat(), **values}
    rows = [row("generation_start", 0)]
    for index in range(24):
        device = f"cuda:{0 if index < 12 else 1}"
        rows.extend((row("layer_q_proj_enter", index + 1, layer=index,
                         query_tokens=70, execution_device=device),
                     row("attention_chunk_start", index + 1.1, layer=index, query_tokens=70),
                     row("attention_chunk_end", index + 1.5, layer=index, query_tokens=70),
                     row("layer_complete", index + 1.9, layer=index,
                         query_tokens=70, execution_device=device)))
    devices = {f"cuda:{index}": {"peak_allocated_bytes": 8 * 1024**3,
                                  "peak_reserved_bytes": 9 * 1024**3} for index in (0, 1)}
    rows.append(row("generation_end", 26, status="ok", memory={"devices": devices},
                    minimum_global_free_bytes={key: MIN_FREE_BYTES + 1 for key in devices},
                    free_memory_samples={key: 2 for key in devices}, sampling_errors=[]))
    return rows


def test_probe_requires_correct_two_gpu_prefill_and_headroom():
    rows = probe_rows()
    result = validate_stage(rows, 1, 70)
    assert result["forward"]["prefill_layers_completed"] == 24
    assert result["timing"]["prefill_observed_seconds"] == 23.9
    assert result["timing"]["attention_observed_seconds_by_layer"]["0"] == 0.4
    assert result["timing"]["boundary_gap_observed_seconds"] == 0.1
    assert result["timing"]["after_prefill_observed_seconds"] == 1.1
    rows[-1]["minimum_global_free_bytes"]["cuda:1"] = MIN_FREE_BYTES - 1
    with pytest.raises(RuntimeError, match="headroom"):
        validate_stage(rows, 1, 70)
    rows = probe_rows()
    rows[49]["execution_device"] = "cuda:0"  # layer 12 q_proj
    with pytest.raises(RuntimeError, match="wrong GPU"):
        validate_stage(rows, 1, 70)
