import json

import pytest

from agent_support.analysis import compare_support
from agent_support.common import canonical_digest, write_json_atomic


def make_run(root, condition, manifest, config, successes, commit):
    root.mkdir()
    (root / "episodes").mkdir()
    metadata = {
        "status": "complete", "benchmark": "textcraft", "condition": condition,
        "partition": "development", "run_id": root.name, "git_commit": commit,
        "limit": None, "selected_task_count": len(manifest["tasks"]),
        "total_manifest_task_count": len(manifest["tasks"]),
        "manifest_sha256": canonical_digest(manifest),
        "selected_manifest_sha256": canonical_digest(manifest),
        "config_sha256": canonical_digest(config),
        "package_versions": {"textcraft": "0.0.3"}, "python_version": "3.13.16",
        "alfworld_config_sha256": None, "textcraft_recipe_order_sha256": None,
    }
    if condition == "m2_always":
        metadata.update({"memory_corpus_sha256": "corpus", "memory_index_sha256": "index",
                         "memory_selection_policy": "positive-v1"})
    write_json_atomic(root / "run.json", metadata)
    write_json_atomic(root / "config.json", config)
    write_json_atomic(root / "manifest.json", manifest)
    for i, task in enumerate(manifest["tasks"]):
        episode = {"task_id": task["task_id"], "success": successes[i],
                   "status": "success" if successes[i] else "max_actions",
                   "environment_actions": 5, "model_decisions": 5,
                   "input_tokens": 100, "output_tokens": 10, "inference_seconds": 2.0,
                   "failed_actions": 1, "parser_errors": 0}
        if condition == "m2_always":
            episode.update({"model_calls": 7, "support_calls": 5, "support_model_calls": 2,
                            "support_input_tokens": 40, "support_output_tokens": 4,
                            "actions_changed_by_support": 1, "retrieval_seconds": 0.1})
        write_json_atomic(root / "episodes" / f"{i:04d}.json", episode)
    return root


def test_paired_support_comparison_accepts_different_commits_and_counts_cost(tmp_path):
    manifest = {"benchmark": "textcraft", "partition": "development",
                "tasks": [{"task_id": "task-a"}, {"task_id": "task-b"}]}
    base_config = {"benchmark": "textcraft", "condition": "baseline", "model_id": "Qwen",
                   "quantization": "nf4_4bit", "max_actions": 30, "max_decisions": 30,
                   "history_window": 8, "max_new_tokens": 100, "seed": 42}
    m2_config = {**base_config, "condition": "m2_always", "memory_corpus": "corpus.json",
                 "memory_index": "index.json", "memory_top_k": 3}
    baseline = make_run(tmp_path / "baseline", "baseline", manifest, base_config,
                        [False, False], "old-commit")
    supported = make_run(tmp_path / "m2", "m2_always", manifest, m2_config,
                         [True, False], "new-commit")
    result = compare_support(baseline, supported)
    assert result["paired"]["helped"] == 1
    assert result["paired"]["hurt"] == 0
    assert result["paired"]["success_rate_delta"] == 0.5
    assert result["paired"]["model_call_delta"] == 4
    assert result["m2"]["support_hit_rate"] == 0.4
    assert result["m2"]["support_input_tokens"] == 80
    assert result["baseline_git_commit"] == "old-commit"
    assert result["m2_git_commit"] == "new-commit"

    changed = {**m2_config, "max_actions": 29}
    write_json_atomic(supported / "config.json", changed)
    metadata = json.loads((supported / "run.json").read_text())
    metadata["config_sha256"] = canonical_digest(changed)
    write_json_atomic(supported / "run.json", metadata)
    with pytest.raises(ValueError, match="agent settings differ"):
        compare_support(baseline, supported)

    write_json_atomic(supported / "config.json", m2_config)
    metadata["config_sha256"] = canonical_digest(m2_config)
    metadata["limit"] = 1
    write_json_atomic(supported / "run.json", metadata)
    with pytest.raises(ValueError, match="full manifest"):
        compare_support(baseline, supported)

    metadata["limit"] = None
    changed_manifest = {**manifest, "tasks": list(reversed(manifest["tasks"]))}
    write_json_atomic(supported / "manifest.json", changed_manifest)
    metadata["manifest_sha256"] = canonical_digest(changed_manifest)
    metadata["selected_manifest_sha256"] = canonical_digest(changed_manifest)
    write_json_atomic(supported / "run.json", metadata)
    with pytest.raises(ValueError, match="manifests differ"):
        compare_support(baseline, supported)
