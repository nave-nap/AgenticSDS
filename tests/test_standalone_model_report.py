"""CPU-only checks for strict standalone comparison output."""

from pathlib import Path

import pytest

from agent_support.common import canonical_digest, read_json
from agent_support import standalone_model_report as report


def test_report_pairs_tasks_and_keeps_runtime_provenance(monkeypatch):
    qconfig = read_json("configs/standalone_textcraft_dev_qwen_never.json")
    gconfig = read_json("configs/standalone_textcraft_dev_gptoss20b_never.json")
    manifest = {"partition": "development", "tasks": [{"task_id": "a"}, {"task_id": "b"}]}
    def meta(config):
        return {"schema_version": 2, "status": "complete", "model_revision": config["model_revision"],
                "model_id": config["model_id"], "model_type": config["model_type"],
                "model_metadata": {"model_revision": config["model_revision"]},
                "selected_manifest_sha256": canonical_digest(manifest), "git_dirty": False,
                "git_commit": "commit", "limit": None}
    def episode(task, success, **extra):
        return {"task_id": task, "success": success, "support_calls": 0,
                "task_sha256": canonical_digest(task), "environment_actions": 2,
                "failed_actions": 1, "model_calls": 2, "input_tokens": 10,
                "output_tokens": 5, "parser_errors": 0, "inference_seconds": 1.5, **extra}
    runs = {
        "qwen": (meta(qconfig), qconfig, manifest,
                 {"a": episode("a", True), "b": episode("b", False)}),
        "gpt": (meta(gconfig), gconfig, manifest,
                {"a": episode("a", False, reasoning_tokens=3, generation_failures=1,
                              peak_gpu_memory_allocated_bytes=100),
                 "b": episode("b", True, reasoning_tokens=4, generation_failures=0,
                              peak_gpu_memory_allocated_bytes=120)})}
    monkeypatch.setattr(report, "load_run", lambda path: runs[Path(path).name])
    result, markdown = report.build_report(Path("qwen"), Path("gpt"), expected_tasks=2)
    assert result["paired"]["qwen_only_success"] == ["a"]
    assert result["paired"]["gpt_only_success"] == ["b"]
    assert result["totals"]["gpt_oss_20b"]["reasoning_tokens"] == 7
    assert result["totals"]["gpt_oss_20b"]["generation_failures"] == 1
    assert result["totals"]["gpt_oss_20b"]["peak_gpu_memory_allocated_bytes"] == 120
    assert result["totals"]["qwen"]["peak_gpu_memory_allocated_bytes"] is None
    assert "standalone capability" in markdown
    runs["gpt"][0]["git_commit"] = "different"
    with pytest.raises(ValueError, match="different framework commits"):
        report.build_report(Path("qwen"), Path("gpt"), expected_tasks=2)
