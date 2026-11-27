"""Read-only integrity, trajectory, and matched-policy campaign diagnostics."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import random
import re
import subprocess
from collections import Counter
from pathlib import Path

from .common import canonical_digest, read_json
from .recovery_v4_eval import MANIFEST_SHA256, config_for, manifest, memory_checks, preflight

POLICIES = ("original_v4", "state_aware_v1", "accuracy_first_v1",
            "accuracy_first_terminal_v1")
INVALID_STATUSES = {"controller_cuda_oom", "controller_cuda_fatal",
                    "controller_worker_unavailable", "environment_error"}
POLICY_SOURCE_COMMIT = "701c25d"
POLICY_SOURCE_FILES = ("agent_support/recovery_accuracy_policy.py",
                       "agent_support/recovery_policy_registry.py",
                       "agent_support/recovery_policy_eval.py",
                       "agent_support/recovery_policy_telemetry.py")
METRICS = ("environment_actions", "failed_actions", "support_calls", "model_calls",
           "qwen_model_calls", "gptoss_model_calls", "controller_model_calls",
           "support_model_calls", "input_tokens", "output_tokens", "inference_seconds",
           "elapsed_seconds", "strategy_changes", "replanning_events")


def integrity() -> dict:
    selected = manifest()  # Checks frozen digest, recipe order, source, and held-out exclusion.
    tasks = selected["tasks"]
    if len({item["task_id"] for item in tasks}) != len(tasks):
        raise ValueError("Duplicate recovery task ID")
    if len({item["observation_sha256"] for item in tasks}) != len(tasks):
        raise ValueError("Duplicate recovery observation")
    configurations = {arm: config_for(arm) for arm in
                      ("expanded_v3_v1", "autonomous_recovery_v4")}
    checks = {arm: preflight(arm) for arm in configurations}
    return {"kind": "integrity", "manifest_sha256": MANIFEST_SHA256,
            "task_count": len(tasks), "seeds": [item["seed"] for item in tasks],
            "config_sha256": {arm: canonical_digest(config) for arm, config in configurations.items()},
            "memory": memory_checks(configurations["autonomous_recovery_v4"]),
            "protocol_checks": {arm: check["protocol_checks"] for arm, check in checks.items()},
            "runtime_checks": {arm: check["runtime_checks"] for arm, check in checks.items()},
            "scientific_status": "static_integrity_only"}


def policy_source(repo: Path | None = None) -> dict:
    """Audit the immutable four-policy Git source without importing its old runtime."""
    repository = repo or Path(__file__).resolve().parent.parent
    digests = {}
    sources = {}
    for name in POLICY_SOURCE_FILES:
        result = subprocess.run(["git", "show", f"{POLICY_SOURCE_COMMIT}:{name}"],
                                cwd=repository, capture_output=True, check=True)
        sources[name] = result.stdout.decode("utf-8")
        digests[name] = hashlib.sha256(result.stdout).hexdigest()
    registry = ast.parse(sources["agent_support/recovery_policy_registry.py"])
    entries = next(node.value for node in registry.body if isinstance(node, ast.AnnAssign)
                   and isinstance(node.target, ast.Name) and node.target.id == "POLICY_REGISTRY")
    policy_ids = tuple(key.value for key in entries.keys)
    if policy_ids != POLICIES:
        raise ValueError(f"Four-policy source registry differs: {policy_ids}")
    frozen = subprocess.run(["git", "diff", "--quiet", POLICY_SOURCE_COMMIT,
                             "HEAD", "--", "configs", "manifests"], cwd=repository)
    if frozen.returncode != 0:
        raise ValueError("Frozen config/manifest tree differs from policy source commit")
    stage = subprocess.run(["git", "show", f"{POLICY_SOURCE_COMMIT}:scripts/recovery_policy_stage.sbatch"],
                           cwd=repository, capture_output=True, check=True).stdout.decode("utf-8")
    return {"kind": "policy_source", "source_commit": POLICY_SOURCE_COMMIT,
            "policy_ids": policy_ids, "source_file_sha256": digests,
            "frozen_config_manifest_tree_matches": True,
            "archived_stage_requests_one_gpu": "#SBATCH --gpus=1" in stage,
            "archived_stage_requests_flex_attention": "AGENT_SUPPORT_GPTOSS_FLEX_ATTENTION=1" in stage,
            "scientific_status": "source_audit_only_runtime_integration_required"}


def trajectory(root: Path) -> dict:
    """Describe available episodes; never infer completeness from a directory scan."""
    paths = sorted(root.rglob("episodes/*.json"))
    if not paths:
        raise ValueError(f"No episode JSON files beneath {root}")
    rows = []
    operations = Counter()
    support_statuses = Counter()
    stop_reasons = Counter()
    for path in paths:
        episode = read_json(path)
        if not isinstance(episode.get("trace"), list) or type(episode.get("success")) is not bool:
            raise ValueError(f"Malformed episode: {path}")
        decisions = episode["trace"]
        failed_pairs = Counter()
        support_decisions = 0
        no_support_decisions = 0
        policy_interventions = 0
        progress_judgments = Counter()
        for decision in decisions:
            events = decision.get("support_trace", {}).get("events", [])
            selected = [event["request"]["operation"] for event in events
                        if event.get("request") and event.get("result")]
            support_decisions += bool(selected)
            no_support_decisions += not bool(selected)
            operations.update(selected)
            for event in events:
                if event.get("request"):
                    support_statuses[str((event.get("result") or {}).get("status", "unresolved"))] += 1
                policy_interventions += bool(event.get("policy_intervention"))
                if event.get("progress_assessment"):
                    progress_judgments[str(event["progress_assessment"].get("assessment"))] += 1
            stop_reasons[str(decision.get("support_trace", {}).get("termination_reason", "unknown"))] += 1
            outcome = decision.get("outcome") or {}
            if outcome.get("failed_action"):
                failed_pairs[(decision.get("action"), decision.get("input_observation"))] += 1
        invalid = (episode.get("infrastructure_invalid") is True or
                   episode.get("status") in INVALID_STATUSES or
                   any(decision.get("controller_fallback_reason") in
                       {"controller_backend_error", "controller_worker_unavailable",
                        "controller_cuda_oom", "support_error", "policy_error"}
                       for decision in decisions))
        rows.append({"path": str(path), "task_id": episode.get("task_id"),
                     "status": episode.get("status"), "success": episode["success"],
                     "infrastructure_invalid": invalid, "decisions": len(decisions),
                     "support_decisions": support_decisions,
                     "no_support_decisions": no_support_decisions,
                     "policy_interventions": policy_interventions,
                     "progress_judgments": dict(progress_judgments),
                     "repeated_failed_state_actions": sum(count - 1 for count in failed_pairs.values()
                                                          if count > 1),
                     **{key: episode.get(key) for key in METRICS}})
    return {"kind": "trajectory", "root": str(root), "episode_count": len(rows),
            "unique_task_count": len({row["task_id"] for row in rows}),
            "infrastructure_invalid_count": sum(row["infrastructure_invalid"] for row in rows),
            "support_operations": dict(sorted(operations.items())), "episodes": rows,
            "support_result_statuses": dict(sorted(support_statuses.items())),
            "controller_stop_reasons": dict(sorted(stop_reasons.items())),
            "scientific_status": "descriptive_partial_archive"}


def mcnemar_exact(wins: int, losses: int) -> float:
    """Two-sided conditional exact binomial test on discordant task pairs."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_bootstrap_interval(differences: list[int | float], samples: int = 10000) -> list[float]:
    if not differences:
        raise ValueError("Bootstrap requires paired observations")
    rng = random.Random(20261126)
    n = len(differences)
    estimates = sorted(sum(differences[rng.randrange(n)] for _ in range(n)) / n
                       for _ in range(samples))
    return [round(estimates[int(0.025 * (samples - 1))], 4),
            round(estimates[int(0.975 * (samples - 1))], 4)]


def allocation_cost(path: Path, gpu_hour_rate: float | None = None,
                    cpu_hour_rate: float | None = None) -> dict:
    """Count allocated resources from `sacct -X -n -P`, including failed jobs."""
    if any(rate is not None and rate < 0 for rate in (gpu_hour_rate, cpu_hour_rate)):
        raise ValueError("Rates must be nonnegative")
    rows = []
    with path.open(newline="", encoding="utf-8") as stream:
        for fields in csv.reader(stream, delimiter="|"):
            if not fields or len(fields) != 6:
                raise ValueError("Expected JobIDRaw|State|ExitCode|ElapsedRaw|AllocTRES|MaxRSS")
            job, state, exit_code, seconds, tres, max_rss = fields
            if not job.isdigit() or not seconds.isdigit():
                raise ValueError("Use one allocation row per job with numeric ElapsedRaw")
            resources = dict(re.findall(r"(?:^|,)([^=,]+)=([^,]+)", tres))
            gpu = int(resources.get("gres/gpu", resources.get("gpu", "0")))
            cpu = int(resources.get("cpu", "0"))
            if cpu < 1 or gpu < 0:
                raise ValueError(f"Missing or invalid allocated resources for job {job}")
            hours = int(seconds) / 3600
            rows.append({"job_id": job, "state": state, "exit_code": exit_code,
                         "elapsed_seconds": int(seconds), "allocated_gpus": gpu,
                         "allocated_cpus": cpu, "gpu_hours": round(gpu * hours, 4),
                         "cpu_hours": round(cpu * hours, 4), "max_rss": max_rss})
    if len({row["job_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate allocation job ID")
    gpu_hours = sum(row["gpu_hours"] for row in rows)
    cpu_hours = sum(row["cpu_hours"] for row in rows)
    return {"kind": "allocation_cost", "jobs": rows,
            "gpu_hours": round(gpu_hours, 4), "cpu_hours": round(cpu_hours, 4),
            "estimated_dollars": (None if gpu_hour_rate is None or cpu_hour_rate is None
                                  else round(gpu_hours * gpu_hour_rate +
                                             cpu_hours * cpu_hour_rate, 4)),
            "rates": {"gpu_hour": gpu_hour_rate, "cpu_hour": cpu_hour_rate},
            "scientific_status": "resource_accounting_not_episode_validity"}


def compare_reports(paths: dict[str, Path]) -> dict:
    if set(paths) != set(POLICIES):
        raise ValueError("Exactly the four frozen policy IDs are required")
    reports = {policy: read_json(path) for policy, path in paths.items()}
    plans = {report.get("plan_sha256") for report in reports.values()}
    protocols = {report.get("protocol") for report in reports.values()}
    if len(plans) != 1 or None in plans or len(protocols) != 1:
        raise ValueError("Policy reports have different or missing protocol/plan digests")
    tasks = manifest()["tasks"]
    task_ids = [task["task_id"] for task in tasks]
    by_policy = {}
    for policy, report in reports.items():
        if (report.get("policy_id") != policy or report.get("primary_score_ready") is not True or
                report.get("complete") is not True or
                report.get("infrastructure_invalid_task_ids") != [] or
                report.get("tasks_completed") != len(tasks) or
                report.get("manifest_sha256") != MANIFEST_SHA256):
            raise ValueError(f"Incomplete or infrastructure-invalid report: {policy}")
        rows = report.get("tasks", [])
        if [row.get("task_id") for row in rows] != task_ids:
            raise ValueError(f"Task order or coverage mismatch: {policy}")
        if any(row.get("seed") != task["seed"] or
               row.get("observation_sha256") != task["observation_sha256"] or
               type(row.get("success")) is not bool or
               row.get("infrastructure_invalid") is not False or
               row.get("status") in INVALID_STATUSES
               for row, task in zip(rows, tasks)):
            raise ValueError(f"Task identity or validity mismatch: {policy}")
        if report.get("successes") != sum(row["success"] for row in rows):
            raise ValueError(f"Success count mismatch: {policy}")
        by_policy[policy] = rows
    comparisons = {}
    for left, right in (("state_aware_v1", "original_v4"),
                        ("accuracy_first_v1", "state_aware_v1"),
                        ("accuracy_first_terminal_v1", "accuracy_first_v1"),
                        ("accuracy_first_terminal_v1", "original_v4")):
        a, b = by_policy[left], by_policy[right]
        wins = sum(x["success"] and not y["success"] for x, y in zip(a, b))
        losses = sum(y["success"] and not x["success"] for x, y in zip(a, b))
        success_differences = [int(x["success"]) - int(y["success"]) for x, y in zip(a, b)]
        deltas = {}
        for key in METRICS:
            values = [(x.get(key), y.get(key)) for x, y in zip(a, b)]
            if all(isinstance(x, (int, float)) and isinstance(y, (int, float))
                   for x, y in values):
                deltas[key] = round(sum(x - y for x, y in values) / len(values), 4)
        comparisons[f"{left}_minus_{right}"] = {
            "wins": wins, "losses": losses,
            "success_rate_difference": round((wins - losses) / len(tasks), 4),
            "paired_bootstrap_95": paired_bootstrap_interval(success_differences),
            "mcnemar_exact_two_sided_p": round(mcnemar_exact(wins, losses), 6),
            "mean_paired_metric_differences": deltas}
    return {"kind": "matched_four_policy", "protocol": protocols.pop(),
            "plan_sha256": plans.pop(), "task_count": len(tasks),
            "comparisons": comparisons,
            "limitations": "Development-set paired estimates; post-failure and support-use subsets are descriptive, not randomized effects."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("integrity")
    sub.add_parser("policy-source")
    trails = sub.add_parser("trajectory")
    trails.add_argument("--root", type=Path, required=True)
    paired = sub.add_parser("compare")
    for policy in POLICIES:
        paired.add_argument(f"--{policy.replace('_', '-')}", type=Path, required=True)
    cost = sub.add_parser("allocation-cost")
    cost.add_argument("--sacct", type=Path, required=True)
    cost.add_argument("--gpu-hour-rate", type=float)
    cost.add_argument("--cpu-hour-rate", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "integrity":
        result = integrity()
    elif args.command == "policy-source":
        result = policy_source()
    elif args.command == "trajectory":
        result = trajectory(args.root)
    elif args.command == "allocation-cost":
        result = allocation_cost(args.sacct, args.gpu_hour_rate, args.cpu_hour_rate)
    else:
        result = compare_reports({policy: getattr(args, policy) for policy in POLICIES})
    data = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(data)
    else:
        print(data, end="")


if __name__ == "__main__":
    main()
