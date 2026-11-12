"""Compare complete Qwen and GPT-OSS standalone TextCraft development runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import load_run
from .common import canonical_digest, read_json, write_json_atomic


SHARED = ("benchmark", "manifest", "partition_manifests", "seed", "history_window",
          "max_actions", "max_decisions", "max_model_calls", "selection_policy",
          "supports", "support_versions", "max_support_calls", "max_support_tokens")
METRICS = ("environment_actions", "failed_actions", "model_calls", "input_tokens",
           "output_tokens", "reasoning_tokens", "parser_errors", "generation_failures",
           "inference_seconds")


def _metrics(episode: dict) -> dict:
    return {key: episode.get(key, 0) for key in METRICS} | {
        "success": bool(episode["success"]),
        "total_tokens": episode["input_tokens"] + episode["output_tokens"],
        "peak_gpu_memory_allocated_bytes": episode.get("peak_gpu_memory_allocated_bytes"),
        "peak_gpu_memory_reserved_bytes": episode.get("peak_gpu_memory_reserved_bytes")}


def build_report(qwen_dir: Path, gpt_dir: Path, expected_tasks: int = 20) -> tuple[dict, str]:
    qmeta, qconfig, qmanifest, qepisodes = load_run(qwen_dir)
    gmeta, gconfig, gmanifest, gepisodes = load_run(gpt_dir)
    if (qconfig.get("model_type"), gconfig.get("model_type")) != ("qwen", "gpt_oss_20b"):
        raise ValueError("Expected Qwen and GPT-OSS configurations in that order")
    if any(qconfig.get(key) != gconfig.get(key) for key in SHARED):
        raise ValueError("Shared environment or interaction settings differ")
    if (qconfig["selection_policy"] != {"id": "never_v1"} or qconfig["supports"] or
            gconfig["supports"] or qconfig["max_support_calls"] or gconfig["max_support_calls"]):
        raise ValueError("Standalone runs must have no support")
    if (qmanifest != gmanifest or qmanifest.get("partition") != "development" or
            len(qmanifest["tasks"]) != expected_tasks):
        raise ValueError("Ordered development task manifests differ or task count is unexpected")
    if expected_tasks == 20 and any(meta.get("limit") is not None for meta in (qmeta, gmeta)):
        raise ValueError("Full comparison requires untruncated runs")
    if expected_tasks == 3 and any(meta.get("limit") != 3 for meta in (qmeta, gmeta)):
        raise ValueError("Smoke comparison requires both runs limited to three tasks")
    for name, meta, config in (("qwen", qmeta, qconfig), ("gpt_oss_20b", gmeta, gconfig)):
        if (meta.get("schema_version") != 2 or meta.get("model_revision") != config["model_revision"]
                or meta.get("model_id") != config["model_id"] or meta.get("model_type") != name
                or meta.get("selected_manifest_sha256") != canonical_digest(qmanifest)
                or meta.get("git_dirty") or not meta.get("git_commit")
                or meta.get("model_metadata", {}).get("model_revision") != config["model_revision"]
                or meta.get("status") != "complete"):
            raise ValueError(f"{name} run lacks clean, pinned, complete provenance")
        directory = qwen_dir if name == "qwen" else gpt_dir
        if directory.is_dir():
            stored = read_json(directory / "checkpoint.json")["episodes"]
            actual = {path.name: canonical_digest(read_json(path))
                      for path in (directory / "episodes").glob("*.json")}
            if stored != actual:
                raise ValueError(f"{name} episode files differ from the checkpoint")
    if qmeta["git_commit"] != gmeta["git_commit"]:
        raise ValueError("Runs use different framework commits")
    task_ids = [task["task_id"] for task in qmanifest["tasks"]]
    rows = []
    for task_id in task_ids:
        q, g = qepisodes[task_id], gepisodes[task_id]
        if (q.get("support_calls", 0) or g.get("support_calls", 0) or
                q.get("task_sha256") != g.get("task_sha256")):
            raise ValueError(f"Support use or task identity mismatch: {task_id}")
        rows.append({"task_id": task_id, "qwen": _metrics(q), "gpt_oss_20b": _metrics(g)})
    totals = {}
    for name in ("qwen", "gpt_oss_20b"):
        totals[name] = {key: round(sum(row[name][key] for row in rows), 3)
                        for key in METRICS + ("total_tokens",)}
        totals[name]["successes"] = sum(row[name]["success"] for row in rows)
        totals[name]["tasks"] = len(rows)
        totals[name]["success_rate"] = totals[name]["successes"] / len(rows)
        for key in ("peak_gpu_memory_allocated_bytes", "peak_gpu_memory_reserved_bytes"):
            values = [row[name][key] for row in rows if row[name][key] is not None]
            totals[name][key] = max(values) if values else None
    paired = {"gpt_only_success": [row["task_id"] for row in rows
                                   if row["gpt_oss_20b"]["success"] and not row["qwen"]["success"]],
              "qwen_only_success": [row["task_id"] for row in rows
                                    if row["qwen"]["success"] and not row["gpt_oss_20b"]["success"]],
              "both_success": [row["task_id"] for row in rows
                               if row["qwen"]["success"] and row["gpt_oss_20b"]["success"]],
              "neither_success": [row["task_id"] for row in rows
                                  if not row["qwen"]["success"] and not row["gpt_oss_20b"]["success"]]}
    report = {"task_count": len(rows), "manifest_sha256": canonical_digest(qmanifest),
              "git_commit": qmeta["git_commit"], "paired": paired, "totals": totals,
              "tasks": rows, "provenance": {
                  "qwen": {"run": str(qwen_dir), "config": qconfig, "run_metadata": qmeta},
                  "gpt_oss_20b": {"run": str(gpt_dir), "config": gconfig, "run_metadata": gmeta}}}
    md = ["# Standalone TextCraft model comparison", "",
          f"Framework commit: `{report['git_commit']}`. Ordered development tasks: {len(rows)}.", "",
          "| Model | Success | Actions | Failed actions | Model calls | Input tokens | Output tokens | Reasoning tokens | Parser errors | Generation failures | Inference seconds | Peak allocated GPU GiB |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ("qwen", "gpt_oss_20b"):
        item = totals[name]
        peak = item["peak_gpu_memory_allocated_bytes"]
        memory = "unavailable" if peak is None else f"{peak / 2**30:.2f}"
        md.append(f"| {name} | {item['successes']}/{item['tasks']} | {item['environment_actions']} | "
                  f"{item['failed_actions']} | {item['model_calls']} | {item['input_tokens']} | "
                  f"{item['output_tokens']} | {item['reasoning_tokens']} | {item['parser_errors']} | "
                  f"{item['generation_failures']} | {item['inference_seconds']:.1f} | {memory} |")
    md += ["", f"GPT-OSS only successes: {', '.join(paired['gpt_only_success']) or 'none'}.",
           f"Qwen only successes: {', '.join(paired['qwen_only_success']) or 'none'}.", "",
           "The rows compare standalone capability under equal environment action limits. "
           "They do not establish GPT-OSS as a useful on-demand support. Token and time costs are model specific.", ""]
    return report, "\n".join(md)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--gpt-oss", type=Path, required=True)
    parser.add_argument("--expected-tasks", type=int, choices=(3, 20), default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report, markdown = build_report(args.qwen, args.gpt_oss, args.expected_tasks)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output / "report.json", report)
    (args.output / "report.md").write_text(markdown, encoding="utf-8")
    print(args.output / "report.md")


if __name__ == "__main__":
    main()
