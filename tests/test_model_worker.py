"""Worker protocol tests without model weights or CUDA."""

import io
import json

import pytest

from agent_support.hybrid_model import HybridModel, WorkerClient
from agent_support.model_worker import serve


class FakeModel:
    def metadata(self):
        return {"model_id": "fake", "model_revision": "r1"}

    def generate(self, messages):
        print("diagnostic that must not reach protocol stdout")
        return {"text": "Action: inventory", "input_tokens": 2,
                "output_tokens": 2, "inference_seconds": 0.1}

    def count_messages_tokens(self, messages):
        return 2


def test_worker_protocol_is_structured_and_stdout_is_clean(capsys):
    requests = [{"op": "init", "config": {}},
                {"op": "count_messages_tokens", "messages": []},
                {"op": "generate", "messages": []}, {"op": "shutdown"}]
    source = io.StringIO("".join(json.dumps(item) + "\n" for item in requests))
    sink = io.StringIO()
    serve(source, sink, lambda config: FakeModel())
    messages = [json.loads(line) for line in sink.getvalue().splitlines()]
    assert len(messages) == 4 and all(item["ok"] for item in messages)
    assert messages[0]["metadata"]["model_revision"] == "r1"
    assert messages[1]["count"] == 2
    assert messages[2]["generation"]["text"] == "Action: inventory"
    assert "diagnostic" not in sink.getvalue()
    assert "diagnostic" in capsys.readouterr().err


def test_worker_reports_generation_failure_without_corrupting_next_request():
    class Failing(FakeModel):
        calls = 0
        def generate(self, messages):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("generation failed")
            return super().generate(messages)
    source = io.StringIO("".join(json.dumps(item) + "\n" for item in [
        {"op": "init", "config": {}}, {"op": "generate", "messages": []},
        {"op": "generate", "messages": []}, {"op": "shutdown"}]))
    sink = io.StringIO()
    serve(source, sink, lambda config: Failing())
    messages = [json.loads(line) for line in sink.getvalue().splitlines()]
    assert messages[1]["ok"] is False and messages[1]["error"] == "generation failed"
    assert messages[2]["ok"] is True


def fake_executable(tmp_path, init_error=False):
    path = tmp_path / "fake_worker"
    path.write_text("#!/usr/bin/env python3\n"
                    "import json, sys\n"
                    "for line in sys.stdin:\n"
                    "    request = json.loads(line)\n"
                    "    op = request['op']\n"
                    f"    error = {init_error!r} and op == 'init'\n"
                    "    response = {'ok': not error, 'error_type': 'RuntimeError', "
                    "'error': 'startup failure'} if error else "
                    "{'ok': True, 'metadata': {'model_id': 'fake', 'model_revision': 'r1'}, "
                    "'cuda_memory': {}, 'generation': {'text': 'Action: inventory'}}\n"
                    "    print(json.dumps(response), flush=True)\n"
                    "    if op == 'shutdown': break\n")
    path.chmod(0o755)
    return path


def test_client_initializes_once_and_shuts_down(tmp_path):
    worker = WorkerClient(str(fake_executable(tmp_path)), {}, tmp_path / "worker.log", timeout=2)
    assert worker.metadata["model_revision"] == "r1"
    assert worker.request({"op": "generate", "messages": []})["generation"]["text"] == "Action: inventory"
    process = worker.process
    worker.close()
    assert process.poll() == 0


def test_client_reports_startup_failure(tmp_path):
    with pytest.raises(RuntimeError, match="startup failure"):
        WorkerClient(str(fake_executable(tmp_path, True)), {}, tmp_path / "worker.log", timeout=2)


def test_worker_episode_seeds_are_stable_and_model_specific():
    class Stub:
        def __init__(self):
            self.seeds = []
        def request(self, payload):
            self.seeds.append(payload["seed"])
    model = object.__new__(HybridModel)
    model.config = {"seed": 42}
    model.workers = {"qwen": Stub(), "gpt_oss_20b": Stub()}
    model.begin_episode("task-a")
    model.begin_episode("task-b")
    model.begin_episode("task-a")
    assert all(worker.seeds[0] == worker.seeds[2] for worker in model.workers.values())
    assert all(worker.seeds[0] != worker.seeds[1] for worker in model.workers.values())
    assert model.workers["qwen"].seeds[0] != model.workers["gpt_oss_20b"].seeds[0]


def test_hybrid_shutdown_closes_both_workers_after_one_close_error():
    class Stub:
        def __init__(self, fail=False):
            self.fail = fail
            self.closed = False
        def close(self):
            self.closed = True
            if self.fail:
                raise RuntimeError("close failed")
    model = object.__new__(HybridModel)
    first, second = Stub(True), Stub()
    model.workers = {"qwen": first, "gpt_oss_20b": second}
    with pytest.raises(RuntimeError, match="Failed to shut down 1"):
        model.close()
    assert first.closed and second.closed and not model.workers
