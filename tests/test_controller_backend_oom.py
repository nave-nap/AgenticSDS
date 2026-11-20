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
