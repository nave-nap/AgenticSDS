"""Audit the failure smoke run and compare Experiment 1 development runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
from collections import Counter
from pathlib import Path

from .analysis import load_run
from .common import canonical_digest, read_json, write_json_atomic


CONDITIONS = ("never", "always", "random", "failure_triggered")
SUPPORT = "proposal_reconsideration_without_retrieval"
METRICS = ("support_calls", "model_calls", "total_tokens", "failed_actions",
           "actions_changed_by_support", "inference_seconds")
PRE_CALL_BLOCKS = {"support_call_budget", "model_call_budget"}
BOOTSTRAP_SEED = 20261110
BOOTSTRAP_REPLICATES = 10000
REPO = Path(__file__).resolve().parent.parent


def trace_counts(episode: dict, *, policy_spec: dict | None = None) -> dict:
    """Count selections, calls, and block reasons without conflating them."""
    selections = decisions = events = 0
    blocks: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    prior_failed = False
    rng = None
    if policy_spec and policy_spec["id"] == "random_v1":
        seed = f"{policy_spec['seed']}\0{episode['task_id']}\0random_v1".encode()
        rng = random.Random(int.from_bytes(hashlib.sha256(seed).digest(), "big"))
    for step in episode["trace"]:
        policy = step.get("support_policy", {})
        if "before_decision" not in policy or "after_decision" not in policy:
            raise ValueError(f"Missing policy decision in {episode['task_id']}")
        before, after = policy["before_decision"], policy["after_decision"]
        decisions += 2
        selections += len(before["support_ids"]) + len(after["support_ids"])
        reasons[after["reason"]] += 1
        if before["support_ids"] or any(item != SUPPORT for item in after["support_ids"]):
            raise ValueError("Experiment 1 expects only post-proposal reconsideration")
        if policy_spec:
            valid = step["action"] is not None
            kind = policy_spec["id"]
            if kind == "never_v1":
                expected, reason, before_reason = False, "never", "never"
            elif not valid:
                expected, reason, before_reason = False, "ineligible_phase_or_proposal", "ineligible_phase_or_proposal"
            elif kind == "always_v1":
                expected, reason, before_reason = True, "always", "ineligible_phase_or_proposal"
            elif kind == "failure_triggered_v1":
                expected = prior_failed
                reason = "failure_threshold_met" if expected else "failure_threshold_not_met"
                before_reason = "ineligible_phase_or_proposal"
            elif kind == "random_v1":
                expected = rng.random() < policy_spec["probability"]
                reason = "random_draw_selected" if expected else "random_draw_skipped"
                before_reason = "ineligible_phase_or_proposal"
            else:
                raise ValueError(f"Unexpected policy: {kind}")
            if (after["invoke"] != expected or after["reason"] != reason
                    or before["reason"] != before_reason):
                raise ValueError(f"Policy decision disagrees with available feedback in {episode['task_id']}")
        for blocked in policy.get("blocked", []):
            blocks[blocked["reason"]] += 1
        support_events = step.get("support_events", [])
        if any(event.get("mechanism") != "reconsideration_only_v1" for event in support_events):
            raise ValueError(f"Unexpected support event in {episode['task_id']}")
        events += len(support_events)
        if "outcome" in step:
            prior_failed = bool(step["outcome"].get("failed_action", False))
    pre_call_blocks = sum(blocks[reason] for reason in PRE_CALL_BLOCKS)
    calls = episode["support_calls"]
    if selections != calls + pre_call_blocks or events != calls:
        raise ValueError(f"Selection/call accounting mismatch in {episode['task_id']}: "
                         f"selected={selections}, calls={calls}, "
                         f"pre_call_blocks={pre_call_blocks}, events={events}")
    return {"decisions": decisions, "selections": selections, "actual_invocations": calls,
            "support_events": events, "blocked": dict(sorted(blocks.items())),
            "after_reasons": dict(sorted(reasons.items()))}


def audit_failure_smoke(directory: Path, expected: dict[str, int | None]) -> dict:
    metadata, config, manifest, episodes = load_run(directory)
    summary = read_json(directory / "summary.json")
    policy = config["selection_policy"]
    if (metadata.get("schema_version") != 2 or manifest.get("partition") != "development"
            or policy != {"id": "failure_triggered_v1", "support_id": SUPPORT,
                          "threshold": 1, "cooldown": 0}):
        raise ValueError("This is not the threshold-1 failure-triggered development smoke run")
    if len(episodes) != 3 or metadata.get("limit") != 3:
        raise ValueError("Expected the three-task failure-triggered smoke run")
    totals = Counter()
    blocked: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    per_task = []
    for task in manifest["tasks"]:
        counts = trace_counts(episodes[task["task_id"]], policy_spec=policy)
        per_task.append({"task_id": task["task_id"], **counts})
        for key in ("decisions", "selections", "actual_invocations", "support_events"):
            totals[key] += counts[key]
        blocked.update(counts["blocked"])
        reasons.update(counts["after_reasons"])
    if totals["selections"] - totals["actual_invocations"] != sum(
            blocked[reason] for reason in PRE_CALL_BLOCKS):
        raise ValueError("Unexplained selected-but-not-invoked support calls")
    model_calls = sum(episode["model_calls"] for episode in episodes.values())
    if sum(episode["model_decisions"] for episode in episodes.values()) != sum(reasons.values()):
        raise ValueError("Smoke policy decisions disagree with episode decision counts")
    if (summary.get("support_calls") != totals["actual_invocations"] or
            summary.get("model_calls") != model_calls):
        raise ValueError("Smoke summary disagrees with saved episodes")
    checks = {"selections": totals["selections"],
              "actual_invocations": totals["actual_invocations"],
              "not_met": reasons["failure_threshold_not_met"],
              "model_calls": model_calls}
    for key, value in expected.items():
        if value is not None and checks[key] != value:
            raise ValueError(f"Smoke {key} was {checks[key]}, expected {value}")
    return {"run_id": metadata["run_id"], "totals": {**dict(totals), "model_calls": model_calls},
            "blocked": dict(sorted(blocked.items())),
            "after_reasons": dict(sorted(reasons.items())), "per_task": per_task}


def percentile(sorted_values: list[float], fraction: float) -> float:
    offset = (len(sorted_values) - 1) * fraction
    low = math.floor(offset)
    high = math.ceil(offset)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (offset - low)


def bootstrap_interval(values: list[float], samples: list[list[int]]) -> list[float]:
    estimates = sorted(sum(values[index] for index in sample) / len(sample) for sample in samples)
    return [percentile(estimates, 0.025), percentile(estimates, 0.975)]


def wilson_interval(successes: int, count: int) -> list[float]:
    z = 1.959963984540054
    fraction = successes / count
    denominator = 1 + z * z / count
    center = (fraction + z * z / (2 * count)) / denominator
    half = z * math.sqrt(fraction * (1 - fraction) / count + z * z / (4 * count * count)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def task_metrics(episode: dict) -> dict:
    return {"success": int(episode["success"]), "support_calls": episode["support_calls"],
            "model_calls": episode["model_calls"],
            "total_tokens": episode["input_tokens"] + episode["output_tokens"],
            "failed_actions": episode["failed_actions"],
            "actions_changed_by_support": episode["actions_changed_by_support"],
            "inference_seconds": episode["inference_seconds"]}


def checked_runs(paths: dict[str, Path]) -> tuple[dict, list[str]]:
    runs = {}
    development = read_json(REPO / "manifests/textcraft/development.json")
    for name in CONDITIONS:
        directory = paths[name]
        metadata, config, manifest, episodes = load_run(directory)
        config_path = REPO / f"configs/experiment1_textcraft_dev_{name}.json"
        pinned = read_json(config_path)
        if config != pinned or manifest != development:
            raise ValueError(f"{name}: run does not use the committed Experiment 1 config and manifest")
        if (metadata.get("schema_version") != 2 or metadata.get("partition") != "development"
                or metadata.get("limit") is not None or metadata.get("selected_task_count") != 20
                or metadata.get("selected_manifest_sha256") != metadata.get("manifest_sha256")
                or metadata.get("git_dirty") is not False
                or metadata.get("python_hash_seed") != "0"):
            raise ValueError(f"{name}: incomplete or non-frozen development run")
        if (metadata.get("config_file_sha256") != hashlib.sha256(config_path.read_bytes()).hexdigest()
                or metadata.get("manifest_file_sha256") != hashlib.sha256(
                    (REPO / "manifests/textcraft/development.json").read_bytes()).hexdigest()
                or metadata.get("model_revision") != config["model_revision"]
                or metadata.get("model_metadata", {}).get("model_revision") != config["model_revision"]
                or metadata.get("policy_parameters") != config["selection_policy"]
                or metadata.get("support_versions") != config["support_versions"]
                or metadata.get("budget_limits") != {key: config[key] for key in (
                    "max_actions", "max_decisions", "max_model_calls", "max_support_calls",
                    "max_support_tokens")}):
            raise ValueError(f"{name}: run provenance differs from pinned configuration")
        if directory.parent.resolve() != Path(metadata["effective_output_root"]).resolve():
            raise ValueError(f"{name}: output root differs from run provenance")
        if directory.parent.name != "experiment1_textcraft_development":
            raise ValueError(f"{name}: output root is not isolated for Experiment 1")
        checkpoint = read_json(directory / "checkpoint.json")["episodes"]
        files = list((directory / "episodes").glob("*.json"))
        if set(checkpoint) != {path.name for path in files}:
            raise ValueError(f"{name}: checkpoint file list differs from episodes")
        for path in files:
            episode = read_json(path)
            if (checkpoint[path.name] != canonical_digest(episode)
                    or episode.get("config_sha256") != canonical_digest(config)):
                raise ValueError(f"{name}: episode or checkpoint hash mismatch: {path}")
        for task in manifest["tasks"]:
            episode = episodes[task["task_id"]]
            if episode.get("task_sha256") != canonical_digest(task):
                raise ValueError(f"{name}: task hash mismatch: {task['task_id']}")
            trace_counts(episode, policy_spec=config["selection_policy"])
        runs[name] = (metadata, config, manifest, episodes)
    reference_meta, reference_config, reference_manifest, _ = runs["never"]
    shared_metadata = ("git_commit", "model_id", "model_revision", "model_metadata",
                       "package_versions", "python_version", "gpu_name", "torch_cuda_version",
                       "seed", "python_hash_seed", "decoding", "budget_limits",
                       "manifest_sha256", "manifest_file_sha256", "partition_manifest_sha256",
                       "textcraft_recipe_order_sha256", "prompt_versions", "support_versions",
                       "component_modules", "effective_output_root")
    for name in CONDITIONS[1:]:
        metadata, config, manifest, _ = runs[name]
        if manifest != reference_manifest:
            raise ValueError(f"{name}: task order, content, or seeds differ")
        if {key: value for key, value in config.items() if key != "selection_policy"} != {
                key: value for key, value in reference_config.items() if key != "selection_policy"}:
            raise ValueError(f"{name}: non-policy configuration differs")
        for key in shared_metadata:
            if metadata.get(key) != reference_meta.get(key):
                raise ValueError(f"{name}: provenance differs in {key}")
    if len({paths[name].parent.resolve() for name in CONDITIONS}) != 1:
        raise ValueError("Experiment 1 runs must share their isolated output root")
    return runs, [task["task_id"] for task in reference_manifest["tasks"]]


def build_report(paths: dict[str, Path]) -> tuple[dict, str, str]:
    runs, task_ids = checked_runs(paths)
    count = len(task_ids)
    rng = random.Random(BOOTSTRAP_SEED)
    samples = [[rng.randrange(count) for _ in range(count)] for _ in range(BOOTSTRAP_REPLICATES)]
    rows = {name: {task_id: task_metrics(runs[name][3][task_id]) for task_id in task_ids}
            for name in CONDITIONS}
    report = {"protocol": "experiment1_textcraft_development_v1", "task_count": count,
              "git_commit": runs["never"][0]["git_commit"],
              "model_revision": runs["never"][0]["model_revision"],
              "manifest_sha256": runs["never"][0]["manifest_sha256"],
              "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_replicates": BOOTSTRAP_REPLICATES,
              "intervals": "Wilson 95% for success; paired task bootstrap percentile 95% for differences; task bootstrap percentile 95% for mean compute",
              "conditions": {}, "comparisons_vs_never": {}}
    for name in CONDITIONS:
        metadata, config, _, episodes = runs[name]
        success = sum(rows[name][task_id]["success"] for task_id in task_ids)
        policy = Counter()
        blocked = Counter()
        for task_id in task_ids:
            counts = trace_counts(episodes[task_id], policy_spec=config["selection_policy"])
            policy.update({key: counts[key] for key in ("decisions", "selections", "actual_invocations")})
            blocked.update(counts["blocked"])
        metrics = {}
        for metric in METRICS:
            values = [rows[name][task_id][metric] for task_id in task_ids]
            metrics[metric] = {"total": sum(values), "per_task_mean": sum(values) / count,
                               "per_task_mean_ci95": bootstrap_interval(values, samples)}
        report["conditions"][name] = {
            "run_id": metadata["run_id"], "run_directory": str(paths[name].resolve()),
            "config_sha256": metadata["config_sha256"], "policy": config["selection_policy"],
            "successes": success, "success_rate": success / count,
            "success_wilson_ci95": wilson_interval(success, count),
            "policy_decisions": policy["decisions"], "policy_selections": policy["selections"],
            "actual_invocations": policy["actual_invocations"],
            "blocked_selections": sum(blocked.values()),
            "blocked_by_reason": dict(sorted(blocked.items())), "metrics": metrics}
    for name in CONDITIONS[1:]:
        success_diffs = [rows[name][task_id]["success"] - rows["never"][task_id]["success"]
                         for task_id in task_ids]
        metric_differences = {}
        for metric in METRICS:
            diffs = [rows[name][task_id][metric] - rows["never"][task_id][metric]
                     for task_id in task_ids]
            metric_differences[metric] = {"total": sum(diffs), "per_task_mean": sum(diffs) / count,
                                          "paired_mean_ci95": bootstrap_interval(diffs, samples)}
        report["comparisons_vs_never"][name] = {
            "helped": sum(value == 1 for value in success_diffs),
            "hurt": sum(value == -1 for value in success_diffs),
            "success_difference_pp": 100 * sum(success_diffs) / count,
            "success_difference_pp_ci95": [100 * value for value in
                                            bootstrap_interval(success_diffs, samples)],
            "metric_differences": metric_differences}
    output = io.StringIO()
    columns = ("task_id", "condition", "success", "baseline_success", *METRICS,
               "policy_selections", "actual_invocations", "blocked_selections",
               "delta_success", "delta_model_calls", "delta_total_tokens")
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    for task_id in task_ids:
        for name in CONDITIONS:
            values = rows[name][task_id]
            counts = trace_counts(runs[name][3][task_id])
            writer.writerow({"task_id": task_id, "condition": name,
                             "success": values["success"],
                             "baseline_success": rows["never"][task_id]["success"],
                             **{metric: values[metric] for metric in METRICS},
                             "policy_selections": counts["selections"],
                             "actual_invocations": counts["actual_invocations"],
                             "blocked_selections": sum(counts["blocked"].values()),
                             "delta_success": values["success"] - rows["never"][task_id]["success"],
                             "delta_model_calls": values["model_calls"] - rows["never"][task_id]["model_calls"],
                             "delta_total_tokens": values["total_tokens"] - rows["never"][task_id]["total_tokens"]})
    return report, markdown_report(report), output.getvalue()


def markdown_report(report: dict) -> str:
    lines = ["# Experiment 1: TextCraft development policies", "",
             f"Git commit: `{report['git_commit']}`. Model revision: `{report['model_revision']}`.",
             f"Tasks: {report['task_count']} development tasks in frozen manifest order.", "",
             "| Policy | Success / 20 (Wilson 95%) | Selections / calls / blocked | Model calls | Tokens | Failed actions | Actions changed | Inference seconds |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name in CONDITIONS:
        item = report["conditions"][name]
        low, high = item["success_wilson_ci95"]
        metrics = item["metrics"]
        lines.append(f"| {name} | {item['successes']}/20 ({low:.2f}–{high:.2f}) | "
                     f"{item['policy_selections']} / {item['actual_invocations']} / {item['blocked_selections']} | "
                     f"{metrics['model_calls']['total']} | {metrics['total_tokens']['total']} | "
                     f"{metrics['failed_actions']['total']} | {metrics['actions_changed_by_support']['total']} | "
                     f"{metrics['inference_seconds']['total']:.1f} |")
    lines.extend(["", "## Paired differences from never support", "",
                  "Differences use the same 20 tasks; positive compute values mean more usage. Intervals are descriptive with limited power.", "",
                  "| Policy | Helped / hurt | Δ success (pp; paired 95%) | Δ model calls | Δ tokens | Δ inference seconds |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for name in CONDITIONS[1:]:
        item = report["comparisons_vs_never"][name]
        low, high = item["success_difference_pp_ci95"]
        differences = item["metric_differences"]
        lines.append(f"| {name} | {item['helped']} / {item['hurt']} | "
                     f"{item['success_difference_pp']:+.1f} ({low:+.1f}–{high:+.1f}) | "
                     f"{differences['model_calls']['total']:+} | "
                     f"{differences['total_tokens']['total']:+} | "
                     f"{differences['inference_seconds']['total']:+.1f} |")
    lines.extend(["", "## Selection accounting", ""])
    for name in CONDITIONS:
        item = report["conditions"][name]
        lines.append(f"- **{name}:** {item['policy_selections']} selections, "
                     f"{item['actual_invocations']} invocations, "
                     f"{item['blocked_selections']} blocked; reasons: "
                     f"`{json.dumps(item['blocked_by_reason'], sort_keys=True)}`.")
    lines.extend(["", "See `task_comparisons.csv` for paired task outcomes and costs, and "
                  "`report.json` for all metric intervals and provenance. These 20 development tasks "
                  "are exploratory; do not tune on held-out tasks.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser("audit-smoke", help="Reconcile the failure smoke selection gap")
    smoke.add_argument("run_directory", type=Path)
    smoke.add_argument("--expect-selections", type=int)
    smoke.add_argument("--expect-calls", type=int)
    smoke.add_argument("--expect-not-met", type=int)
    smoke.add_argument("--expect-model-calls", type=int)
    report_command = commands.add_parser("report", help="Validate and compare four complete 20-task runs")
    for name in CONDITIONS:
        report_command.add_argument(f"--{name.replace('_', '-')}", required=True, type=Path)
    report_command.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "audit-smoke":
        result = audit_failure_smoke(args.run_directory, {
            "selections": args.expect_selections, "actual_invocations": args.expect_calls,
            "not_met": args.expect_not_met, "model_calls": args.expect_model_calls})
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        paths = {name: getattr(args, name) for name in CONDITIONS}
        report, markdown, task_csv = build_report(paths)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        write_json_atomic(args.output_dir / "report.json", report)
        (args.output_dir / "report.md").write_text(markdown, encoding="utf-8")
        (args.output_dir / "task_comparisons.csv").write_text(task_csv, encoding="utf-8")
        print(args.output_dir / "report.md")


if __name__ == "__main__":
    main()
