"""Strict local comparison of two completed runs on identical tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .common import canonical_digest, read_json, write_json_atomic


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
    if metadata.get("textcraft_recipe_order_sha256") is not None:
        source = root / "textcraft_recipe_order.json"
        if hashlib.sha256(source.read_bytes()).hexdigest() != metadata["textcraft_recipe_order_sha256"]:
            raise ValueError(f"TextCraft recipe order snapshot changed: {root}")
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
    for field in ("git_commit", "python_version", "alfworld_config_sha256", "textcraft_recipe_order_sha256"):
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


SUPPORT_ONLY_CONFIG_KEYS = {"condition", "memory_corpus", "memory_index", "memory_top_k"}


def compare_support(baseline_dir: str | Path, supported_dir: str | Path) -> dict:
    """Paired comparison of full development runs with different support conditions."""
    base_meta, base_config, base_manifest, base_results = load_run(baseline_dir)
    sup_meta, sup_config, sup_manifest, sup_results = load_run(supported_dir)
    if (base_config.get("condition"), sup_config.get("condition")) != ("baseline", "m2_always"):
        raise ValueError("Expected baseline run first and M2 run second")
    if base_meta.get("condition") != "baseline" or sup_meta.get("condition") != "m2_always":
        raise ValueError("Run metadata condition differs from config")
    if base_meta["benchmark"] != sup_meta["benchmark"] or base_manifest != sup_manifest:
        raise ValueError("Benchmarks or task manifests differ")
    if base_meta["selected_manifest_sha256"] != sup_meta["selected_manifest_sha256"]:
        raise ValueError("Task manifest hashes differ")
    if (base_meta.get("partition") != "development" or sup_meta.get("partition") != "development"
            or base_manifest.get("partition") != "development"):
        raise ValueError("Support comparison requires development runs")
    for name, meta in (("baseline", base_meta), ("M2", sup_meta)):
        if (meta.get("limit") is not None or meta.get("selected_task_count") != meta.get("total_manifest_task_count")
                or meta.get("selected_manifest_sha256") != meta.get("manifest_sha256")):
            raise ValueError(f"{name} run does not cover its full manifest")
    base_settings = {key: value for key, value in base_config.items() if key not in SUPPORT_ONLY_CONFIG_KEYS}
    sup_settings = {key: value for key, value in sup_config.items() if key not in SUPPORT_ONLY_CONFIG_KEYS}
    if base_settings != sup_settings:
        raise ValueError("Baseline agent settings differ between conditions")
    if base_meta["package_versions"] != sup_meta["package_versions"]:
        raise ValueError("Package versions differ between conditions")
    for field in ("python_version", "alfworld_config_sha256", "textcraft_recipe_order_sha256"):
        if base_meta.get(field) != sup_meta.get(field):
            raise ValueError(f"Run metadata differs: {field}")
    if not all(sup_meta.get(field) for field in
               ("memory_corpus_sha256", "memory_index_sha256", "memory_selection_policy")):
        raise ValueError("M2 run is missing memory provenance")

    task_ids = [task["task_id"] for task in base_manifest["tasks"]]

    def metrics(episodes: dict[str, dict], supported: bool) -> dict:
        values = list(episodes.values())
        count = len(values)
        model_calls = sum(item.get("model_calls", item["model_decisions"]) for item in values)
        total_tokens = sum(item["input_tokens"] + item["output_tokens"] for item in values)
        return {
            "tasks": count,
            "successes": sum(bool(item["success"]) for item in values),
            "success_rate": sum(bool(item["success"]) for item in values) / count,
            "environment_actions": sum(item["environment_actions"] for item in values),
            "mean_environment_actions": sum(item["environment_actions"] for item in values) / count,
            "model_decisions": sum(item["model_decisions"] for item in values),
            "model_calls": model_calls,
            "total_tokens": total_tokens,
            "mean_total_tokens": total_tokens / count,
            "inference_seconds": round(sum(item["inference_seconds"] for item in values), 3),
            "failed_actions": sum(item["failed_actions"] for item in values),
            "parser_errors": sum(item["parser_errors"] for item in values),
            "support_calls": sum(item.get("support_calls", 0) for item in values) if supported else 0,
            "support_model_calls": sum(item.get("support_model_calls", 0) for item in values) if supported else 0,
            "support_input_tokens": sum(item.get("support_input_tokens", 0) for item in values) if supported else 0,
            "support_output_tokens": sum(item.get("support_output_tokens", 0) for item in values) if supported else 0,
            "actions_changed_by_support": sum(item.get("actions_changed_by_support", 0) for item in values) if supported else 0,
            "retrieval_seconds": round(sum(item.get("retrieval_seconds", 0) for item in values), 3) if supported else 0,
        }

    rows = []
    for task_id in task_ids:
        base = base_results[task_id]
        sup = sup_results[task_id]
        outcome = ("helped" if sup["success"] and not base["success"] else
                   "hurt" if base["success"] and not sup["success"] else "same")
        rows.append({
            "task_id": task_id,
            "outcome": outcome,
            "baseline_success": bool(base["success"]),
            "m2_success": bool(sup["success"]),
            "baseline_status": base["status"],
            "m2_status": sup["status"],
            "baseline_environment_actions": base["environment_actions"],
            "m2_environment_actions": sup["environment_actions"],
            "delta_environment_actions": sup["environment_actions"] - base["environment_actions"],
            "baseline_model_calls": base.get("model_calls", base["model_decisions"]),
            "m2_model_calls": sup.get("model_calls", sup["model_decisions"]),
            "delta_model_calls": sup.get("model_calls", sup["model_decisions"]) - base.get("model_calls", base["model_decisions"]),
            "baseline_total_tokens": base["input_tokens"] + base["output_tokens"],
            "m2_total_tokens": sup["input_tokens"] + sup["output_tokens"],
            "delta_total_tokens": (sup["input_tokens"] + sup["output_tokens"]
                                   - base["input_tokens"] - base["output_tokens"]),
            "m2_support_calls": sup.get("support_calls", 0),
            "m2_actions_changed": sup.get("actions_changed_by_support", 0),
        })
    base_totals = metrics(base_results, supported=False)
    sup_totals = metrics(sup_results, supported=True)
    sup_totals["support_hit_rate"] = (sup_totals["support_model_calls"] / sup_totals["support_calls"]
                                      if sup_totals["support_calls"] else 0.0)
    return {
        "schema_version": 1,
        "benchmark": base_meta["benchmark"],
        "partition": "development",
        "manifest_sha256": base_meta["selected_manifest_sha256"],
        "baseline_run_id": base_meta["run_id"],
        "baseline_git_commit": base_meta.get("git_commit"),
        "m2_run_id": sup_meta["run_id"],
        "m2_git_commit": sup_meta.get("git_commit"),
        "memory_corpus_sha256": sup_meta["memory_corpus_sha256"],
        "memory_index_sha256": sup_meta["memory_index_sha256"],
        "memory_selection_policy": sup_meta["memory_selection_policy"],
        "baseline": base_totals,
        "m2": sup_totals,
        "paired": {
            "helped": sum(row["outcome"] == "helped" for row in rows),
            "hurt": sum(row["outcome"] == "hurt" for row in rows),
            "same": sum(row["outcome"] == "same" for row in rows),
            "success_rate_delta": sup_totals["success_rate"] - base_totals["success_rate"],
            "environment_action_delta": sup_totals["environment_actions"] - base_totals["environment_actions"],
            "model_call_delta": sup_totals["model_calls"] - base_totals["model_calls"],
            "total_token_delta": sup_totals["total_tokens"] - base_totals["total_tokens"],
        },
        "tasks": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left_run")
    parser.add_argument("right_run")
    parser.add_argument("--support-comparison", action="store_true", help="Compare a full baseline run with a full M2 development run")
    parser.add_argument("--output", help="Write the comparison JSON to this path")
    args = parser.parse_args()
    result = (compare_support(args.left_run, args.right_run) if args.support_comparison
              else compare(args.left_run, args.right_run))
    if args.output:
        write_json_atomic(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
