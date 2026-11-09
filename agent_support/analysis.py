"""Strict local comparison of two completed runs on identical tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .common import canonical_digest, read_json


def load_run(directory: str | Path) -> tuple[dict, dict, dict, dict[str, dict]]:
    root = Path(directory)
    metadata = read_json(root / "run.json")
    if metadata.get("status") != "complete":
        raise ValueError(f"Run is not complete: {root}")
    config = read_json(root / "config.json")
    manifest = read_json(root / "manifest.json")
    if canonical_digest(config) != metadata["config_sha256"]:
        raise ValueError(f"Config snapshot changed: {root}")
    if canonical_digest(manifest) != metadata["selected_manifest_sha256"]:
        raise ValueError(f"Manifest snapshot changed: {root}")
    if metadata.get("alfworld_config_sha256") is not None:
        source = root / "alfworld_config.yaml"
        if hashlib.sha256(source.read_bytes()).hexdigest() != metadata["alfworld_config_sha256"]:
            raise ValueError(f"ALFWorld YAML snapshot changed: {root}")
    results = {}
    for path in sorted((root / "episodes").glob("*.json")):
        episode = read_json(path)
        task_id = episode["task_id"]
        if task_id in results:
            raise ValueError(f"Duplicate task result: {task_id}")
        results[task_id] = episode
    expected = {task["task_id"] for task in manifest["tasks"]}
    if set(results) != expected or len(results) != metadata["selected_task_count"]:
        raise ValueError(f"Run has missing or unexpected task results: {root}")
    return metadata, config, manifest, results


def compare(left: str | Path, right: str | Path) -> dict:
    left_meta, left_config, left_manifest, left_results = load_run(left)
    right_meta, right_config, right_manifest, right_results = load_run(right)
    if left_meta["benchmark"] != right_meta["benchmark"]:
        raise ValueError("Benchmarks differ")
    if left_meta["selected_manifest_sha256"] != right_meta["selected_manifest_sha256"]:
        raise ValueError("Task manifests differ, including task content or order")
    # All baseline settings must match. A future support implementation may explicitly
    # exclude support-only keys after defining its own comparison contract.
    if left_config != right_config:
        raise ValueError("Baseline configurations differ")
    if left_meta["package_versions"] != right_meta["package_versions"]:
        raise ValueError("Package versions differ")
    for field in ("git_commit", "python_version", "alfworld_config_sha256"):
        if left_meta.get(field) != right_meta.get(field):
            raise ValueError(f"Run metadata differs: {field}")
    task_ids = [task["task_id"] for task in left_manifest["tasks"]]

    def totals(results: dict[str, dict]) -> dict:
        episodes = list(results.values())
        n = len(episodes)
        return {
            "tasks": n,
            "successes": sum(bool(item["success"]) for item in episodes),
            "success_rate": sum(bool(item["success"]) for item in episodes) / n,
            "mean_environment_actions": sum(item["environment_actions"] for item in episodes) / n,
            "mean_model_decisions": sum(item["model_decisions"] for item in episodes) / n,
            "mean_total_tokens": sum(item["input_tokens"] + item["output_tokens"] for item in episodes) / n,
            "parser_errors": sum(item["parser_errors"] for item in episodes),
            "failed_actions": sum(item["failed_actions"] for item in episodes),
        }

    return {
        "benchmark": left_meta["benchmark"],
        "left_run": left_meta["run_id"],
        "right_run": right_meta["run_id"],
        "left": totals(left_results),
        "right": totals(right_results),
        "changed_outcomes": [
            {"task_id": task_id, "left_success": left_results[task_id]["success"], "right_success": right_results[task_id]["success"]}
            for task_id in task_ids if left_results[task_id]["success"] != right_results[task_id]["success"]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left_run")
    parser.add_argument("right_run")
    args = parser.parse_args()
    print(json.dumps(compare(args.left_run, args.right_run), indent=2))


if __name__ == "__main__":
    main()
