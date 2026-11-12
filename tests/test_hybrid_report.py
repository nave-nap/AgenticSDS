"""Matched hybrid report checks without GPU artifacts."""

from pathlib import Path

import pytest

from agent_support.common import canonical_digest, read_json, write_json_atomic
from agent_support.hybrid_report import build_report


def test_hybrid_report_pairs_tasks_and_rejects_tampering(tmp_path):
    manifest = {"benchmark": "textcraft", "partition": "development",
                "tasks": [{"task_id": f"task-{i}"} for i in range(20)]}
    configs = {
        "qwen": read_json("configs/standalone_textcraft_dev_qwen_never.json"),
        "gpt_oss_20b": read_json("configs/standalone_textcraft_dev_gptoss20b_low_never.json"),
        "random": read_json("configs/hybrid_textcraft_dev_random.json"),
        "failure_triggered": read_json("configs/hybrid_textcraft_dev_failure_triggered.json")}
    paths = {}
    for name, config in configs.items():
        root = tmp_path / name
        (root / "episodes").mkdir(parents=True)
        paths[name] = root
        write_json_atomic(root / "config.json", config)
        write_json_atomic(root / "manifest.json", manifest)
        model_meta = {"model_revision": config["model_revision"]}
        if name in ("random", "failure_triggered"):
            model_meta["workers"] = {
                "primary": {"metadata": {"model_revision": configs["qwen"]["model_revision"]}},
                "delegated": {"metadata": {"model_revision": configs["gpt_oss_20b"]["model_revision"]}}}
        metadata = {"schema_version": 2, "status": "complete", "git_commit": "same-commit",
                    "git_dirty": False, "limit": None, "model_id": config["model_id"],
                    "model_type": config["model_type"], "model_revision": config["model_revision"],
                    "model_metadata": model_meta, "config_sha256": canonical_digest(config),
                    "selected_manifest_sha256": canonical_digest(manifest),
                    "selected_task_count": 20, "started_utc": "2026-11-12T00:00:00+00:00",
                    "finished_utc": "2026-11-12T00:10:00+00:00"}
        write_json_atomic(root / "run.json", metadata)
        checkpoint = {}
        for index, task in enumerate(manifest["tasks"]):
            hybrid = name in ("random", "failure_triggered")
            episode = {"task_id": task["task_id"], "benchmark": "textcraft",
                       "task_sha256": canonical_digest(task), "success": index == (1 if hybrid else 0),
                       "environment_actions": 1, "model_decisions": 1, "parser_errors": 0,
                       "failed_actions": 0, "input_tokens": 10, "output_tokens": 5,
                       "inference_seconds": 1.0, "elapsed_seconds": 1.1,
                       "trace": [], "model_calls": 1, "support_calls": int(hybrid),
                       "support_model_calls": 0, "support_input_tokens": 0,
                       "support_output_tokens": 0, "support_parse_errors": 0,
                       "retrieval_seconds": 0.0, "actions_changed_by_support": 0}
            if name == "gpt_oss_20b" or hybrid:
                episode.update(reasoning_tokens=2, final_tokens=2, generation_failures=0,
                               peak_gpu_memory_allocated_bytes=100,
                               peak_gpu_memory_reserved_bytes=120)
            if hybrid:
                usage = {key: {"model_calls": int(key == "gpt_oss_20b"),
                               "input_tokens": 10 if key == "gpt_oss_20b" else 0,
                               "output_tokens": 5 if key == "gpt_oss_20b" else 0,
                               "reasoning_tokens": 2 if key == "gpt_oss_20b" else 0,
                               "final_tokens": 2 if key == "gpt_oss_20b" else 0,
                               "inference_seconds": 1.0 if key == "gpt_oss_20b" else 0.0,
                               "peak_gpu_memory_allocated_bytes": 100,
                               "peak_gpu_memory_reserved_bytes": 120}
                         for key in ("qwen", "gpt_oss_20b")}
                episode.update(model_usage=usage, delegation_selections=1,
                               blocked_delegations=0, delegation_attempts=1,
                               valid_delegated_actions=1, delegated_actions_without_failure=1,
                               fallback_calls=0,
                               peak_combined_worker_cuda_allocated_bytes=100,
                               peak_combined_worker_cuda_reserved_bytes=120)
            filename = f"{index:04d}.json"
            write_json_atomic(root / "episodes" / filename, episode)
            checkpoint[filename] = canonical_digest(episode)
        write_json_atomic(root / "checkpoint.json", {"episodes": checkpoint})
    report, markdown = build_report(paths)
    assert report["paired"]["random"]["helped_vs_qwen"] == ["task-1"]
    assert report["paired"]["random"]["hurt_vs_qwen"] == ["task-0"]
    assert report["totals"]["random"]["delegation_attempts"] == 20
    assert "Output tokens include reasoning tokens" in markdown
    tampered = paths["random"] / "episodes" / "0000.json"
    episode = read_json(tampered)
    episode["success"] = True
    write_json_atomic(tampered, episode)
    with pytest.raises(ValueError, match="integrity"):
        build_report(paths)
