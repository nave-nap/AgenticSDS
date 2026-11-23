"""CPU tests for read-only joining and censored OOM interpretation."""

from __future__ import annotations

import json

import pytest

from agent_support.gpu_memory_analysis import GIB, analyze


def memory(allocated, reserved, peak, free):
    return {"allocated_bytes": allocated * GIB, "reserved_bytes": reserved * GIB,
            "peak_allocated_bytes": peak * GIB, "global_free_bytes": free * GIB}


def write_events(path, events):
    path.write_text("".join(json.dumps(event) + "\n" for event in events))


def test_pairs_requests_cache_release_and_censors_oom(tmp_path):
    path = tmp_path / "gpt_oss_cuda_memory_test.jsonl"
    write_events(path, [
        {"event": "generation_start", "request_id": 1, "max_output_tokens": 1024,
         "memory": memory(12, 13, 20, 25)},
        {"event": "generation_end", "request_id": 1, "input_tokens": 4000,
         "max_output_tokens": 1024, "status": "ok", "error_type": None,
         "memory": memory(13, 21, 24, 17)},
        {"event": "cache_release", "request_id": 1, "status": "ok",
         "memory_before": memory(12, 21, 24, 17),
         "memory_after": memory(12, 13, 24, 25)},
        {"event": "generation_start", "request_id": 2, "previous_request_id": 1,
         "max_output_tokens": 1024, "memory": memory(12, 13, 24, 25)},
        {"event": "generation_end", "request_id": 2, "input_tokens": 7555,
         "max_output_tokens": 1024, "status": "error", "error_type": "OutOfMemoryError",
         "memory": memory(13, 33, 33, 5)},
    ])
    result = analyze([tmp_path])
    first, second = result["generations"]
    assert first["peak_minus_start_allocated_gib"] == 12
    assert first["cache_release"] == [{"status": "ok", "reserved_before_gib": 21,
                                         "reserved_after_gib": 13}]
    assert second["previous_request_id"] == 1
    assert second["oom_peak_is_lower_bound"] is True
    assert result["summary"]["gpt_oss"]["successful_count"] == 1
    assert result["summary"]["gpt_oss"]["oom_requests"][0]["request_id"] == 2
    assert result["summary"]["gpt_oss"]["successful_context_bins"] == [
        {"input_tokens_from": 4000, "input_tokens_below": 5000, "count": 1,
         "median_peak_minus_start_allocated_gib": 12}]
    assert result["summary"]["gpt_oss"][
        "successful_peak_excess_slope_gib_per_million_tokens_squared"] is None


def test_qwen_model_load_and_missing_end_are_visible(tmp_path):
    path = tmp_path / "qwen_cuda_memory_test.jsonl"
    write_events(path, [
        {"event": "model_loaded", "hf_device_map": {"": "0"},
         "visible_cuda_devices": [{"index": 0, "name": "L40S"}]},
        {"event": "generation_start", "request_id": 1, "max_output_tokens": 100,
         "memory": memory(6, 7, 6, 25)},
    ])
    result = analyze([path])
    assert result["model_loads"][0]["event"]["hf_device_map"] == {"": "0"}
    assert result["generations"][0]["status"] == "missing_generation_end"
    assert result["summary"]["qwen"]["successful_count"] == 0


def test_duplicate_and_malformed_events_are_rejected(tmp_path):
    path = tmp_path / "gpt_oss_cuda_memory_test.jsonl"
    write_events(path, [{"event": "generation_start", "request_id": 1},
                        {"event": "generation_start", "request_id": 1}])
    with pytest.raises(ValueError, match="duplicate generation_start"):
        analyze([path])
    path.write_text("{invalid json\n")
    with pytest.raises(ValueError, match="invalid JSON"):
        analyze([path])
