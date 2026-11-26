"""CPU-only checks for the opt-in checkpoint-loading telemetry."""

import json
from pathlib import Path
from types import SimpleNamespace

from agent_support import gpt_oss_load_diagnostics as diagnostics


def records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_gpu_samples_map_both_worker_ordinals_and_keep_raw_rows(monkeypatch):
    output = ("0, GPU-qwen, NVIDIA L40S, 7000, 1, 2\n"
              "1, GPU-gpt-a, NVIDIA L40S, 8000, 20, 30\n"
              "2, GPU-gpt-b, NVIDIA L40S, 9000, 40, 50\n")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,2")
    monkeypatch.setattr(diagnostics.subprocess, "run",
                        lambda *args, **kwargs: SimpleNamespace(stdout=output))
    sample = diagnostics._gpu_snapshot()
    assert sample["worker_cuda_ordinals"]["cuda:0"]["uuid"] == "GPU-gpt-a"
    assert sample["worker_cuda_ordinals"]["cuda:1"]["gpu_utilization_percent"] == 40
    assert len(sample["physical_gpus"]) == 3
    cuda = SimpleNamespace(device_count=lambda: 2,
                           memory_allocated=lambda index: 100 + index,
                           memory_reserved=lambda index: 200 + index)
    memory = diagnostics._torch_gpu_memory(SimpleNamespace(cuda=cuda))
    assert memory == {"cuda:0": {"allocated_bytes": 100, "reserved_bytes": 200},
                      "cuda:1": {"allocated_bytes": 101, "reserved_bytes": 201}}


def test_progress_and_counter_deltas_are_timestamped(tmp_path, monkeypatch):
    log = tmp_path / "worker.log"
    log.write_text("Loading weights: 0%| | 0/411 [00:00<?, ?it/s]\r"
                   "Loading weights: 1%| | 4/411 [09:43<12:40, 12s/it]\r"
                   "Loading weights: 5%| | 19/411 [10:30<39:41, 6s/it]\r")
    monkeypatch.setattr(diagnostics, "_gpu_snapshot", lambda: {"worker_cuda_ordinals": {}})
    snapshots = iter([
        {"io": {"rchar": 100, "read_bytes": 0}, "cpu_user_seconds": 1,
         "cpu_system_seconds": 1, "rss_bytes": 1000, "minor_faults": 0, "major_faults": 0},
        {"io": {"rchar": 300, "read_bytes": 100}, "cpu_user_seconds": 3,
         "cpu_system_seconds": 1, "rss_bytes": 2000, "minor_faults": 2, "major_faults": 1},
    ])
    monkeypatch.setattr(diagnostics, "_process_snapshot", lambda: next(snapshots))
    monitor = diagnostics.ModelLoadDiagnostics(tmp_path, log)
    monitor.started_at = 10.0
    monitor.last_progress_at = 10.0
    monitor._sample(10.0)
    monitor._progress(20.0)
    monitor._sample(20.0)
    rows = records(monitor.path)
    progress = [row for row in rows if row["event"] == "weight_progress"]
    assert [(row["completed"], row["loader_elapsed_seconds"]) for row in progress] == [(4, 583), (19, 630)]
    assert [row["progress_interval_seconds"] for row in progress] == [583, 47]
    assert all(row["single_weight_interval_seconds"] is None for row in progress)
    assert rows[-1]["io_delta"] == {"rchar": 200, "read_bytes": 100}
    assert rows[-1]["cpu_percent_one_core"] == 20.0
    assert rows[-1]["process"]["rss_bytes"] == 2000


def test_context_records_error_and_does_not_suppress_it(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics, "_gpu_snapshot", lambda: {"worker_cuda_ordinals": {}})
    try:
        with diagnostics.ModelLoadDiagnostics(tmp_path, None, sample_seconds=60) as monitor:
            raise RuntimeError("checkpoint read failed")
    except RuntimeError:
        pass
    ends = [row for row in records(monitor.path) if row["event"] == "model_load_end"]
    assert ends[0]["status"] == "error"
    assert ends[0]["error_type"] == "RuntimeError"


def test_stall_events_are_emitted_without_loader_progress(tmp_path, monkeypatch):
    monitor = diagnostics.ModelLoadDiagnostics(tmp_path, None, sample_seconds=15,
                                               stall_seconds=120)
    monitor.started_at = monitor.last_progress_at = 0
    ticks = iter([121.0, 242.0])
    monkeypatch.setattr(diagnostics.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(monitor, "_sample", lambda _now: None)
    class StopAfterTwoSamples:
        calls = 0
        def wait(self, _seconds):
            self.calls += 1
            return self.calls > 2
    monitor.stop_event = StopAfterTwoSamples()
    monitor._loop()
    stalls = [row for row in records(monitor.path) if row["event"] == "progress_stall"]
    assert [row["seconds_since_progress"] for row in stalls] == [121.0, 242.0]


def test_opt_in_requires_environment_and_diagnostics_directory(monkeypatch, tmp_path):
    monkeypatch.delenv(diagnostics.ENABLE_ENV, raising=False)
    assert diagnostics.monitor(tmp_path).__class__.__name__ == "nullcontext"
    monkeypatch.setenv(diagnostics.ENABLE_ENV, "1")
    assert diagnostics.monitor(None).__class__.__name__ == "nullcontext"
    assert isinstance(diagnostics.monitor(tmp_path), diagnostics.ModelLoadDiagnostics)
