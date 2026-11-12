"""Compare matched Qwen, GPT-OSS, and two hybrid development runs."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from .analysis import load_run
from .common import canonical_digest, read_json, write_json_atomic
from .runner import summarize
from .standalone_model_report import build_report as standalone_report


NAMES = ("qwen", "gpt_oss_20b", "random", "failure_triggered")
METRICS = ("environment_actions", "failed_actions", "model_calls", "support_calls",
           "input_tokens", "output_tokens", "reasoning_tokens", "inference_seconds",
           "elapsed_seconds", "delegation_selections", "blocked_delegations",
           "delegation_attempts", "valid_delegated_actions", "fallback_calls")


def build_report(paths: dict[str, Path]) -> tuple[dict, str]:
    if set(paths) != set(NAMES):
        raise ValueError("Expected Qwen, GPT-OSS, random, and failure-triggered runs")
    standalone_report(paths["qwen"], paths["gpt_oss_20b"], 20)
    runs = {name: load_run(path) for name, path in paths.items()}
    reference_meta, qconfig, reference_manifest, _ = runs["qwen"]
    gconfig = runs["gpt_oss_20b"][1]
    commit = reference_meta["git_commit"]
    task_ids = [task["task_id"] for task in reference_manifest["tasks"]]
    for name in NAMES:
        meta, config, manifest, episodes = runs[name]
        if (meta.get("git_commit") != commit or meta.get("git_dirty") or
                meta.get("limit") is not None or manifest != reference_manifest or
                len(episodes) != 20):
            raise ValueError(f"{name} lacks matched complete clean provenance")
        checkpoint = read_json(paths[name] / "checkpoint.json")["episodes"]
        actual = {path.name: canonical_digest(read_json(path))
                  for path in (paths[name] / "episodes").glob("*.json")}
        if checkpoint != actual:
            raise ValueError(f"{name} episode integrity differs from checkpoint")
        if name in ("random", "failure_triggered"):
            expected_policy = ({"id": "random_v1", "support_id": "stronger_model_delegation",
                                "phase": "before_proposal", "probability": 0.25, "seed": 20261112}
                               if name == "random" else
                               {"id": "failure_triggered_v1", "support_id": "stronger_model_delegation",
                                "phase": "before_proposal", "threshold": 2, "cooldown": 0})
            if (config.get("model_type") != "hybrid_delegation" or
                    config.get("selection_policy") != expected_policy or
                    config.get("max_support_calls") != 5 or
                    config.get("supports") != ["stronger_model_delegation"] or
                    config.get("delegation_fallback_to_primary") is not True or
                    config["worker_models"]["primary"]["model"] != qconfig or
                    config["worker_models"]["delegated"]["model"] != gconfig or
                    any(config.get(key) != qconfig.get(key) for key in
                        ("manifest", "seed", "history_window", "max_actions", "max_decisions", "max_model_calls"))):
                raise ValueError(f"{name} hybrid settings differ from the references")
            workers = meta.get("model_metadata", {}).get("workers", {})
            if (workers.get("primary", {}).get("metadata", {}).get("model_revision") != qconfig["model_revision"] or
                    workers.get("delegated", {}).get("metadata", {}).get("model_revision") != gconfig["model_revision"]):
                raise ValueError(f"{name} worker revision differs from configuration")
    rows = []
    for task_id in task_ids:
        row = {"task_id": task_id}
        expected_hash = runs["qwen"][3][task_id]["task_sha256"]
        for name in NAMES:
            episode = runs[name][3][task_id]
            if episode["task_sha256"] != expected_hash:
                raise ValueError(f"Task identity differs: {task_id}")
            row[name] = {"success": episode["success"],
                         **{key: episode.get(key, 0) for key in METRICS},
                         "total_tokens": episode["input_tokens"] + episode["output_tokens"],
                         "model_usage": episode.get("model_usage", {})}
        rows.append(row)
    totals = {}
    for name in NAMES:
        values = list(runs[name][3].values())
        totals[name] = summarize(values)
        totals[name]["total_tokens"] = totals[name]["input_tokens"] + totals[name]["output_tokens"]
        started = datetime.fromisoformat(runs[name][0]["started_utc"])
        finished = datetime.fromisoformat(runs[name][0]["finished_utc"])
        totals[name]["run_wall_seconds"] = round((finished - started).total_seconds(), 3)
    paired = {}
    for name in NAMES[1:]:
        paired[name] = {"helped_vs_qwen": [row["task_id"] for row in rows
                                          if row[name]["success"] and not row["qwen"]["success"]],
                        "hurt_vs_qwen": [row["task_id"] for row in rows
                                        if row["qwen"]["success"] and not row[name]["success"]]}
    report = {"git_commit": commit, "manifest_sha256": canonical_digest(reference_manifest),
              "task_ids": task_ids, "totals": totals, "paired": paired, "tasks": rows,
              "provenance": {name: {"path": str(paths[name]), "run_metadata": runs[name][0],
                                    "config": runs[name][1]} for name in NAMES}}
    markdown = ["# Hybrid TextCraft development comparison", "",
                f"Commit `{commit}`; 20 ordered development tasks. Output tokens include reasoning tokens.", "",
                "| Condition | Success | Model calls | Qwen calls | GPT-OSS calls | Delegations selected/attempted/blocked | Total tokens | Inference s | Failed actions | Combined peak reserved GiB |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in NAMES:
        item = totals[name]
        qwen_calls = (item["model_calls"] if name == "qwen" else
                      item.get("model_usage", {}).get("qwen", {}).get("model_calls", 0))
        gpt_calls = (item["model_calls"] if name == "gpt_oss_20b" else
                     item.get("model_usage", {}).get("gpt_oss_20b", {}).get("model_calls", 0))
        selected = item.get("delegation_selections", 0)
        attempted = item.get("delegation_attempts", 0)
        blocked = item.get("blocked_delegations", 0)
        combined = item.get("peak_combined_worker_cuda_reserved_bytes")
        combined_text = f"{combined / 2**30:.2f}" if combined else "unavailable"
        markdown.append(f"| {name} | {item['successes']}/20 | {item['model_calls']} | "
                        f"{qwen_calls} | {gpt_calls} | "
                        f"{selected}/{attempted}/{blocked} | {item['total_tokens']} | "
                        f"{item['inference_seconds']:.1f} | {item['failed_actions']} | {combined_text} |")
    for name in NAMES[1:]:
        markdown += ["", f"{name} vs Qwen: helped {', '.join(paired[name]['helped_vs_qwen']) or 'none'}; "
                     f"hurt {', '.join(paired[name]['hurt_vs_qwen']) or 'none'}. "]
    markdown += ["", "These 20 paired tasks are descriptive; compare realized calls, tokens, and latency alongside success."]
    return report, "\n".join(markdown) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in NAMES:
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report, markdown = build_report({name: getattr(args, name) for name in NAMES})
    args.output.mkdir(parents=True, exist_ok=False)
    write_json_atomic(args.output / "report.json", report)
    (args.output / "report.md").write_text(markdown)
    print(args.output / "report.md")


if __name__ == "__main__":
    main()
