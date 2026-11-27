"""CPU-only regression checks for feasibility startup failures."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_support import three_gpu_feasibility as feasibility
from agent_support import three_gpu_feasibility_entry as entry
from agent_support.controller_backend import GptOssControllerWorker


def test_preflight_failure_writes_traceback_and_stages(tmp_path, monkeypatch):
    def broken_preflight():
        raise RuntimeError("memory corpus unavailable")
    monkeypatch.setattr(feasibility, "preflight", broken_preflight)
    result = feasibility.run(tmp_path / "attempt", None)
    saved = json.loads((tmp_path / "attempt" / "feasibility_result.json").read_text())
    assert result == saved
    assert result["status"] == "failed"
    assert result["startup_stage"] == "preflight"
    assert "broken_preflight" in result["traceback"]
    assert "memory corpus unavailable" in result["error_detail"]
    events = (tmp_path / "attempt" / "startup_events.jsonl").read_text()
    assert '"stage": "preflight"' in events


def test_clean_commit_failure_is_structured(tmp_path, monkeypatch):
    monkeypatch.setattr(feasibility, "preflight", lambda: {"manifest_sha256": "abc"})
    monkeypatch.setattr(feasibility, "require_clean_commit",
                        lambda: (_ for _ in ()).throw(ValueError("checkout is dirty")))
    result = feasibility.run(tmp_path / "attempt", None)
    assert result["status"] == "failed"
    assert result["startup_stage"] == "clean_commit"
    assert result["manifest_sha256"] == "abc"
    assert "checkout is dirty" in result["traceback"]


def test_worker_initialization_failure_includes_log_and_exit_code(tmp_path, monkeypatch):
    monkeypatch.setattr(feasibility, "preflight", lambda: {})
    monkeypatch.setattr(feasibility, "require_clean_commit", lambda: "fixed")
    for key in (feasibility.SHARD_ENV, feasibility.CHUNKED_ENV,
                "AGENT_SUPPORT_GPU_MEMORY_DIAGNOSTICS"):
        monkeypatch.setenv(key, "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv(feasibility.VISIBLE_ENV, "1,2")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        device_count=lambda: 1, get_device_name=lambda _: "NVIDIA L40S")))
    from agent_support import controller_backend
    class BrokenWorker:
        def __init__(self, _spec, directory):
            directory.mkdir(parents=True)
            (directory / "gpt_oss_controller_worker.log").write_text("worker import failed\n")
            error = OSError("worker executable unavailable")
            error.worker_exit_code = 127
            raise error
    monkeypatch.setattr(controller_backend, "GptOssControllerWorker", BrokenWorker)
    result = feasibility.run(tmp_path / "attempt", None)
    assert result["status"] == "failed"
    assert result["startup_stage"] == "worker_initialize"
    assert result["worker_evidence"]["worker_exit_code"] == 127
    assert result["worker_evidence"]["worker_log_tail"] == "worker import failed\n"
    assert "worker executable unavailable" in result["traceback"]


def test_atomic_result_failure_keeps_original_error_in_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(feasibility, "preflight",
                        lambda: (_ for _ in ()).throw(RuntimeError("original preflight failure")))
    original_write = feasibility.write_json_atomic
    calls = 0
    def fail_final_write(*args):
        nonlocal calls
        calls += 1
        if calls == 1:
            return original_write(*args)
        raise OSError("atomic write failed")
    monkeypatch.setattr(feasibility, "write_json_atomic", fail_final_write)
    result = feasibility.run(tmp_path / "attempt", None)
    assert (tmp_path / "attempt" / "feasibility_result.json").exists()
    fallback = json.loads((tmp_path / "attempt" / "feasibility_result_fallback.json").read_text())
    assert result == fallback
    assert result["error_detail"] == "original preflight failure"
    assert "original preflight failure" in result["traceback"]
    assert "atomic write failed" in result["result_write_error"]


def test_startup_check_uses_worker_python_with_two_visible_l40s(tmp_path, monkeypatch):
    monkeypatch.setenv("PINNED_GPT_PYTHON", sys.executable)
    monkeypatch.setenv("PINNED_HF_HOME", str(tmp_path))
    seen = []
    def fake_run(command, **kwargs):
        seen.append({"command": command, "env": kwargs["env"], "timeout": kwargs["timeout"]})
        value = ({"count": 2, "names": ["NVIDIA L40S", "NVIDIA L40S"]}
                 if "cuda_identity" in command else {"worker_module_imported": True})
        return SimpleNamespace(returncode=0, stdout=json.dumps(value) + "\n", stderr="")
    monkeypatch.setattr(feasibility.subprocess, "run", fake_run)
    result = feasibility._check_worker_process(
        {"python_env": "PINNED_GPT_PYTHON", "hf_home_env": "PINNED_HF_HOME"}, ["1", "2"], tmp_path)
    assert result["count"] == 2
    assert [item["timeout"] for item in seen] == [120, 30]
    assert all(item["env"]["CUDA_VISIBLE_DEVICES"] == "1,2" for item in seen)
    assert all(item["env"]["HF_HOME"] == str(tmp_path) for item in seen)
    assert all("agent_support.three_gpu_startup_probe" in item["command"] for item in seen)
    assert (tmp_path / "cuda_identity_result.json").exists()
    assert (tmp_path / "worker_import_result.json").exists()


def test_startup_check_records_cuda_identity_exit_code(tmp_path, monkeypatch):
    monkeypatch.setenv("PINNED_GPT_PYTHON", sys.executable)
    monkeypatch.setattr(feasibility.subprocess, "run", lambda *_args, **_kwargs:
                        SimpleNamespace(returncode=7, stdout="", stderr="import failed"))
    with pytest.raises(RuntimeError, match="exited 7") as caught:
        feasibility._check_worker_process({"python_env": "PINNED_GPT_PYTHON"},
                                          ["1", "2"], tmp_path)
    assert caught.value.probe_evidence["exit_code"] == 7
    assert json.loads((tmp_path / "cuda_identity_result.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("identity", [
    {"count": 2, "names": ["NVIDIA L40S"]},
    {"count": True, "names": ["NVIDIA L40S", "NVIDIA L40S"]},
    {"count": 2, "names": ["NVIDIA L40S", "NVIDIA A100"]},
])
def test_malformed_cuda_evidence_stops_before_worker_import(tmp_path, monkeypatch, identity):
    monkeypatch.setenv("PINNED_GPT_PYTHON", sys.executable)
    phases = []
    def fake_run(command, **_kwargs):
        phases.append(command[4])
        return SimpleNamespace(returncode=0, stdout=json.dumps(identity) + "\n", stderr="")
    monkeypatch.setattr(feasibility.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="malformed or not two L40S"):
        feasibility._check_worker_process({"python_env": "PINNED_GPT_PYTHON"},
                                          ["1", "2"], tmp_path)
    assert phases == ["cuda_identity"]
    assert not (tmp_path / "worker_import_result.json").exists()


def test_startup_timeout_preserves_partial_timing_and_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(feasibility, "preflight", lambda: {})
    monkeypatch.setattr(feasibility, "require_clean_commit", lambda: "fixed")
    monkeypatch.setattr(feasibility, "config_for", lambda _: {"autonomous_support": {
        "model": {"python_env": "PINNED_GPT_PYTHON"}}})
    monkeypatch.setenv("PINNED_GPT_PYTHON", sys.executable)
    for key in (feasibility.SHARD_ENV, feasibility.CHUNKED_ENV,
                "AGENT_SUPPORT_GPU_MEMORY_DIAGNOSTICS"):
        monkeypatch.setenv(key, "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv(feasibility.VISIBLE_ENV, "1,2")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        device_count=lambda: 1, get_device_name=lambda _: "NVIDIA L40S")))
    def timeout(command, **kwargs):
        Path(command[-1]).write_text(json.dumps({"stage": "python_started",
            "monotonic_seconds": 0, "elapsed_seconds": 0,
            "timestamp_utc": "2026-11-27T00:00:00+00:00"}) + "\n")
        raise subprocess.TimeoutExpired(command, kwargs["timeout"], output=b"partial")
    monkeypatch.setattr(feasibility.subprocess, "run", timeout)
    result = feasibility.run(tmp_path / "attempt", None, startup_only=True)
    assert result["status"] == "failed"
    assert result["startup_stage"] == "worker_process_check"
    assert result["startup_probe_failure"]["status"] == "timeout"
    assert result["startup_probe_failure"]["stdout_tail"] == "partial"
    assert result["startup_probe_failure"]["events"][0]["stage"] == "python_started"
    assert (tmp_path / "attempt" / "cuda_identity_result.json").exists()
    assert not (tmp_path / "attempt" / "worker_import_result.json").exists()
    assert json.loads((tmp_path / "attempt" / "feasibility_result.json").read_text()) == result


def test_worker_import_timeout_follows_valid_cuda_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("PINNED_GPT_PYTHON", sys.executable)
    phases = []
    def fake_run(command, **kwargs):
        phases.append(command[4])
        if command[4] == "cuda_identity":
            return SimpleNamespace(returncode=0, stdout=json.dumps({"count": 2,
                "names": ["NVIDIA L40S", "NVIDIA L40S"]}) + "\n", stderr="")
        raise subprocess.TimeoutExpired(command, kwargs["timeout"], stderr=b"import stalled")
    monkeypatch.setattr(feasibility.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="worker_import exceeded the 30-second") as caught:
        feasibility._check_worker_process({"python_env": "PINNED_GPT_PYTHON"},
                                          ["1", "2"], tmp_path)
    assert phases == ["cuda_identity", "worker_import"]
    assert caught.value.probe_evidence["status"] == "timeout"
    assert caught.value.probe_evidence["stderr_tail"] == "import stalled"
    assert json.loads((tmp_path / "worker_import_result.json").read_text())["status"] == "timeout"


def test_failed_startup_command_exits_nonzero(monkeypatch, tmp_path):
    monkeypatch.setattr(feasibility, "run", lambda *_args, **_kwargs: {"status": "failed"})
    monkeypatch.setattr(sys, "argv", ["feasibility", "startup-check", "--output-dir",
                                  str(tmp_path / "attempt")])
    with pytest.raises(SystemExit) as caught:
        feasibility.main()
    assert caught.value.code == 1


def test_startup_valid_result_cannot_pass_full_feasibility_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(feasibility, "preflight", lambda: {"protocol": feasibility.PROTOCOL})
    monkeypatch.setattr(feasibility, "require_clean_commit", lambda: "same-commit")
    path = tmp_path / "startup_result.json"
    path.write_text(json.dumps({"status": "startup_valid", "protocol": feasibility.PROTOCOL,
                                "git_commit": "same-commit"}))
    with pytest.raises(RuntimeError, match="not infrastructure valid"):
        feasibility.gate(path)


def test_startup_only_path_writes_valid_result_without_loading_models(tmp_path, monkeypatch):
    monkeypatch.setattr(feasibility, "preflight", lambda: {"manifest_sha256": "fixed"})
    monkeypatch.setattr(feasibility, "require_clean_commit", lambda: "fixed")
    monkeypatch.setattr(feasibility, "config_for", lambda _: {"autonomous_support": {"model": {}}})
    monkeypatch.setattr(feasibility, "_check_worker_process", lambda *_:
                        {"count": 2, "names": ["L40S", "L40S"]})
    for key in (feasibility.SHARD_ENV, feasibility.CHUNKED_ENV,
                "AGENT_SUPPORT_GPU_MEMORY_DIAGNOSTICS"):
        monkeypatch.setenv(key, "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv(feasibility.VISIBLE_ENV, "1,2")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        device_count=lambda: 1, get_device_name=lambda _: "NVIDIA L40S")))
    from agent_support import controller_backend
    monkeypatch.setattr(controller_backend, "GptOssControllerWorker",
                        lambda *_: (_ for _ in ()).throw(AssertionError("model load attempted")))
    result = feasibility.run(tmp_path / "startup", None, startup_only=True)
    assert result["status"] == "startup_valid"
    assert result["startup_stage"] == "startup_valid"
    assert json.loads((tmp_path / "startup" / "feasibility_result.json").read_text()) == result


def test_launcher_explains_existing_output_before_python(tmp_path):
    root = tmp_path / "root"
    (root / "feasibility_8887" / "workers").mkdir(parents=True)
    worker_log = root / "feasibility_8887" / "workers" / "gpt_oss_controller_worker.log"
    worker_log.write_text("")
    venv = tmp_path / "venv" / "bin"
    venv.mkdir(parents=True)
    python = venv / "python"
    python.write_text("#!/bin/sh\nexit 99\n")
    python.chmod(0o755)
    cache = tmp_path / "cache"
    cache.mkdir()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    for name in ("corpus_200.json", "index_200.json"):
        (artifacts / name).write_text("{}")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    smi = bin_dir / "nvidia-smi"
    smi.write_text("#!/bin/sh\necho 'mock L40S'\n")
    smi.chmod(0o755)
    repo = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
           "THREE_GPU_ROOT": str(root), "THREE_GPU_PHASE": "feasibility",
           "VENV_PATH": str(venv.parent), "AGENT_SUPPORT_GPTOSS_PYTHON": str(python),
           "AGENT_SUPPORT_GPTOSS_HF_HOME": str(cache), "HF_HOME": str(cache),
           "AGENT_SUPPORT_M2_200_ARTIFACT_DIR": str(artifacts),
           "CUDA_VISIBLE_DEVICES": "0,1,2", "SLURM_SUBMIT_DIR": str(repo)}
    completed = subprocess.run(["bash", str(repo / "scripts" / "three_gpu_chunked_eager_startup.sbatch")],
                               env=env, capture_output=True, text=True)
    assert completed.returncode == 1
    assert "Feasibility output already exists" in completed.stderr
    assert worker_log.read_text() == ""


def test_entry_import_failure_writes_missing_result(tmp_path, monkeypatch):
    monkeypatch.setattr(entry.importlib, "import_module",
                        lambda *_: (_ for _ in ()).throw(ImportError("parent import broke")))
    output = tmp_path / "attempt"
    monkeypatch.setattr(sys, "argv", ["entry", "test", "--output-dir", str(output)])
    with pytest.raises(SystemExit) as caught:
        entry.main()
    assert caught.value.code == 1
    result = json.loads((output / "feasibility_result.json").read_text())
    assert result["status"] == "failed"
    assert result["error_type"] == "ImportError"
    assert "parent import broke" in result["traceback"]


def test_entry_never_replaces_existing_attempt(tmp_path, monkeypatch):
    directory = tmp_path / "old_attempt"
    directory.mkdir()
    (directory / "gpt_oss_controller_worker.log").write_text("prior log")
    monkeypatch.setattr(entry.importlib, "import_module",
                        lambda *_: (_ for _ in ()).throw(ImportError("parent import broke")))
    monkeypatch.setattr(sys, "argv", ["entry", "test", "--output-dir", str(directory)])
    with pytest.raises(SystemExit):
        entry.main()
    assert not (directory / "feasibility_result.json").exists()
    assert (directory / "gpt_oss_controller_worker.log").read_text() == "prior log"


def test_worker_popen_failure_keeps_original_error_and_log_path(tmp_path, monkeypatch):
    from agent_support import controller_backend
    monkeypatch.setenv("PINNED_GPT_PYTHON", sys.executable)
    monkeypatch.setattr(controller_backend.subprocess, "Popen",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(
                            FileNotFoundError("worker exec failed")))
    spec = {"python_env": "PINNED_GPT_PYTHON", "model": {"max_new_tokens": 1024}}
    with pytest.raises(FileNotFoundError, match="worker exec failed") as caught:
        GptOssControllerWorker(spec, tmp_path / "workers")
    assert caught.value.worker_exit_code is None
    assert caught.value.worker_log_path == str(tmp_path / "workers" /
                                                "gpt_oss_controller_worker.log")
    assert (tmp_path / "workers" / "gpt_oss_controller_worker.log").exists()
