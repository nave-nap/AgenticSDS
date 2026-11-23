"""CPU-only checks for structured worker OOM classification."""

import io
import json
from types import SimpleNamespace

import pytest

from agent_support import controller_backend
from agent_support.gpt_oss_controller_worker import serve


def worker_response(monkeypatch, response):
    worker = object.__new__(controller_backend.GptOssControllerWorker)
    worker.closed = False
    worker.timeout = 1
    worker.process = SimpleNamespace(
        poll=lambda: None, stdin=io.StringIO(),
        stdout=io.StringIO(json.dumps(response) + "\n"))
    monkeypatch.setattr(controller_backend.select, "select",
                        lambda ready, *_: (ready, [], []))
    return worker


def test_worker_reports_structured_out_of_memory_error():
    class OutOfMemoryError(Exception):
        pass

    class Model:
        def __init__(self, config):
            pass

        def metadata(self):
            return {}

        def generate(self, messages, max_new_tokens=None):
            raise OutOfMemoryError("CUDA allocation failed at layer 7")

    source = io.StringIO('\n'.join([
        json.dumps({"op": "init", "config": {}}),
        json.dumps({"op": "generate", "messages": []}), ""]))
    sink = io.StringIO()
    serve(source=source, sink=sink, model_factory=Model)
    responses = [json.loads(line) for line in sink.getvalue().splitlines()]
    assert responses[1] == {"ok": False, "error_type": "OutOfMemoryError",
                            "error": "CUDA allocation failed at layer 7"}


def test_only_structured_out_of_memory_error_gets_dedicated_exception(monkeypatch):
    detail = "CUDA out of memory. Tried to allocate 2 GiB"
    worker = worker_response(monkeypatch, {"ok": False, "error_type": "OutOfMemoryError",
                                               "error": detail})
    with pytest.raises(controller_backend.ControllerWorkerOOM) as caught:
        worker._request({"op": "generate", "messages": []})
    assert caught.value.detail == detail
    assert detail in str(caught.value)
    assert "OutOfMemoryError" in str(caught.value)

    worker = worker_response(monkeypatch, {"ok": False, "error_type": "RuntimeError",
                                               "error": detail})
    with pytest.raises(RuntimeError, match="RuntimeError") as ordinary:
        worker._request({"op": "generate", "messages": []})
    assert not isinstance(ordinary.value, controller_backend.ControllerWorkerOOM)
    assert detail in str(ordinary.value)


@pytest.mark.parametrize("error_type,detail", [
    ("AcceleratorError", "CUDA error: an illegal memory access was encountered"),
    ("RuntimeError", "CUDA error: device-side assert triggered"),
    ("AcceleratorError", "CUDA error: unspecified launch failure"),
])
def test_structured_fatal_cuda_poisoned_worker_is_typed_and_not_retried(
        monkeypatch, error_type, detail):
    worker = worker_response(monkeypatch, {"ok": False, "error_type": error_type,
                                          "error": detail, "traceback": "worker stack"})
    with pytest.raises(controller_backend.ControllerWorkerFatalCUDA) as caught:
        worker._request({"op": "generate", "messages": []})
    assert caught.value.error_type == error_type
    assert caught.value.detail == detail
    assert caught.value.worker_traceback == "worker stack"
    assert caught.value.termination_reason == "controller_cuda_fatal"
    assert worker.unusable is True
    with pytest.raises(controller_backend.ControllerWorkerUnavailable, match="terminated"):
        worker._request({"op": "generate", "messages": []})


@pytest.mark.parametrize("error_type,detail", [
    ("ValueError", "CUDA error: an illegal memory access was encountered"),
    ("AcceleratorError", "failed to parse CUDA error: device-side assert triggered"),
    ("RuntimeError", "temporary kernel compilation failed"),
    ("RuntimeError", "task-level invalid action"),
])
def test_nonfatal_worker_errors_stay_ordinary(monkeypatch, error_type, detail):
    worker = worker_response(monkeypatch, {"ok": False, "error_type": error_type,
                                          "error": detail})
    with pytest.raises(RuntimeError) as caught:
        worker._request({"op": "generate", "messages": []})
    assert not isinstance(caught.value, controller_backend.ControllerWorkerFatalCUDA)
    assert not getattr(worker, "unusable", False)


def test_opt_in_worker_traceback_preserves_original_error(monkeypatch):
    class AcceleratorError(Exception):
        pass

    class Model:
        def __init__(self, config):
            pass
        def metadata(self):
            return {}
        def generate(self, messages, max_new_tokens=None):
            raise AcceleratorError("CUDA error: an illegal memory access was encountered")

    source = io.StringIO('\n'.join([json.dumps({"op": "init", "config": {}}),
                                     json.dumps({"op": "generate", "messages": []}), ""]))
    sink = io.StringIO()
    monkeypatch.setenv("AGENT_SUPPORT_GPTOSS_WORKER_TRACEBACKS", "1")
    serve(source=source, sink=sink, model_factory=Model)
    error = [json.loads(line) for line in sink.getvalue().splitlines()][1]
    assert error["error_type"] == "AcceleratorError"
    assert error["error"] == "CUDA error: an illegal memory access was encountered"
    assert "test_opt_in_worker_traceback_preserves_original_error" in error["traceback"]


def test_worker_traceback_includes_request_and_phase(monkeypatch):
    class AcceleratorError(RuntimeError):
        pass

    class Model:
        def __init__(self, _config):
            self._generation_request_id = 0
            self._last_failed_phase = None
        def metadata(self):
            return {}
        def generate(self, _messages, max_new_tokens=None):
            self._generation_request_id = 22
            self._last_failed_phase = "model_generate"
            raise AcceleratorError("CUDA error: an illegal memory access was encountered")

    source = io.StringIO("\n".join([json.dumps({"op": "init", "config": {}}),
                                     json.dumps({"op": "generate", "messages": []}), ""]))
    sink = io.StringIO()
    monkeypatch.setenv("AGENT_SUPPORT_GPTOSS_WORKER_TRACEBACKS", "1")
    serve(source=source, sink=sink, model_factory=Model)
    error = [json.loads(line) for line in sink.getvalue().splitlines()][1]
    assert error["request_id"] == 22 and error["failed_phase"] == "model_generate"
    assert error["traceback"].startswith("GPT-OSS request 22\n")


def test_driver_error_traceback_survives_worker_protocol(monkeypatch, capsys):
    class AcceleratorError(RuntimeError):
        pass

    class Model:
        def __init__(self, _config):
            self._generation_request_id = 0
            self._last_failed_phase = None
        def metadata(self):
            return {}
        def generate(self, _messages, max_new_tokens=None):
            self._generation_request_id = 22
            self._last_failed_phase = "model_generate"
            raise AcceleratorError("CUDA driver error: an illegal memory access was encountered")

    source = io.StringIO("\n".join([json.dumps({"op": "init", "config": {}}),
                                     json.dumps({"op": "generate", "messages": []}), ""]))
    sink = io.StringIO()
    monkeypatch.setenv("AGENT_SUPPORT_GPTOSS_WORKER_TRACEBACKS", "1")
    serve(source=source, sink=sink, model_factory=Model)
    response = [json.loads(line) for line in sink.getvalue().splitlines()][1]
    assert "test_driver_error_traceback_survives_worker_protocol" in response["traceback"]
    assert response["traceback"] in capsys.readouterr().err

    worker = worker_response(monkeypatch, response)
    with pytest.raises(controller_backend.ControllerWorkerFatalCUDA) as caught:
        worker._request({"op": "generate", "messages": []})
    assert caught.value.worker_traceback == response["traceback"]
    assert caught.value.request_id == 22
    assert caught.value.failed_phase == "model_generate"


def test_nonfatal_worker_traceback_is_retained_for_replay(monkeypatch):
    worker = worker_response(monkeypatch, {"ok": False, "error_type": "ValueError",
                                          "error": "invalid input", "traceback": "original worker stack",
                                          "request_id": 4, "failed_phase": "input_transfer"})
    with pytest.raises(RuntimeError) as caught:
        worker._request({"op": "generate", "messages": []})
    assert caught.value.worker_traceback == "original worker stack"
    assert caught.value.request_id == 4
    assert caught.value.failed_phase == "input_transfer"

@pytest.mark.parametrize("failure", ["closed", "exited", "closed_stream", "eof", "broken_pipe"])
def test_definitive_worker_lifecycle_failures_are_typed(monkeypatch, failure):
    worker = worker_response(monkeypatch, {"ok": True})
    if failure == "closed":
        worker.closed = True
    elif failure == "exited":
        worker.process.poll = lambda: 137
    elif failure == "closed_stream":
        worker.process.stdout.close()
    elif failure == "eof":
        worker.process.stdout = io.StringIO("")
    else:
        class BrokenPipe:
            closed = False
            def write(self, value):
                raise BrokenPipeError("pipe closed")
            def flush(self):
                pass
        worker.process.stdin = BrokenPipe()
    with pytest.raises(controller_backend.ControllerWorkerUnavailable) as caught:
        worker._request({"op": "generate", "messages": []})
    assert caught.value.detail
    assert caught.value.termination_reason == "controller_worker_unavailable"


def test_terminated_timeout_is_unavailable_but_unrelated_runtime_error_is_not(monkeypatch):
    worker = worker_response(monkeypatch, {"ok": True})
    worker.process.terminate = lambda: None
    monkeypatch.setattr(controller_backend.select, "select", lambda *_: ([], [], []))
    with pytest.raises(controller_backend.ControllerWorkerUnavailable) as timed_out:
        worker._request({"op": "generate", "messages": []})
    assert timed_out.value.detail == "worker timed out and was terminated"
    assert worker.unusable is True
    with pytest.raises(controller_backend.ControllerWorkerUnavailable, match="terminated"):
        worker._request({"op": "generate", "messages": []})

    worker = worker_response(monkeypatch, {"ok": False, "error_type": "RuntimeError",
                                               "error": "worker unavailable"})
    with pytest.raises(RuntimeError) as ordinary:
        worker._request({"op": "generate", "messages": []})
    assert not isinstance(ordinary.value, controller_backend.ControllerWorkerUnavailable)

    worker = worker_response(monkeypatch, {"ok": True})
    monkeypatch.setattr(controller_backend.select, "select",
                        lambda *_: (_ for _ in ()).throw(InterruptedError("temporary interrupt")))
    with pytest.raises(InterruptedError) as temporary:
        worker._request({"op": "generate", "messages": []})
    assert not isinstance(temporary.value, controller_backend.ControllerWorkerUnavailable)
