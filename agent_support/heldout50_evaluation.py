"""Frozen TextCraft held-out50 preflight, batched execution, and aggregation."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import platform
from pathlib import Path

from .analysis import load_run
from .autonomous_evaluation import preflight as development_preflight
from .autonomous_evaluation import require_clean_commit, resolved_config
from .common import canonical_digest, digest_text, read_json, validate_manifest, write_json_atomic
from .runner import git_commit, package_versions, run


REPO = Path(__file__).resolve().parent.parent
PARENT_PATH = REPO / "manifests/textcraft/heldout.json"
FROZEN_PATH = REPO / "manifests/textcraft/heldout50_v1.json"
PARENT_SHA256 = "5c8dbc6caa2db221cc6ed4ce3208d9c53157af515563895a927957c3ac8349a0"
FROZEN_SHA256 = "614de2034ad3fa14063a9fb5d6013a6754212584020b3e994a7cdb65f5b84c88"
RECIPE_SHA256 = "5d3c08b0404cf0d10571f2ae31edd6375facbc51928ff719ed36106837374c8c"
ARMS = ("autonomous_v3", "qwen_standalone")
AGENT_KEYS = ("model_id", "model_revision", "quantization", "max_actions",
              "max_decisions", "history_window", "max_new_tokens", "seed")
LEGACY_M2_CORPUS_SHA256 = "d6e2e0650c6cc44459e23dcce8e597f85cbdf66aa9a268f15cb94d2b92d3bf66"
LEGACY_M2_INDEX_SHA256 = "6f5eabf78bc3d06e320bf58588720aafe5dce83d3a3040b65c35190666a82e89"


def manifests() -> tuple[dict, dict]:
    parent, frozen = read_json(PARENT_PATH), read_json(FROZEN_PATH)
    validate_manifest(parent)
    validate_manifest(frozen)
    eligible = [task for task in parent["tasks"] if task["seed"] != 51]
    if (canonical_digest(parent) != PARENT_SHA256 or
            canonical_digest(frozen) != FROZEN_SHA256 or
            frozen.get("parent_manifest_sha256") != PARENT_SHA256 or
            frozen.get("excluded_seeds") != [51] or
            frozen.get("selection_seed") != parent.get("selection_seed") or
            frozen.get("partition") != "heldout" or
            len(frozen["tasks"]) != 50 or
            frozen["tasks"] != eligible[:50] or
            hashlib.sha256((REPO / "agent_support/textcraft_recipe_order.json").read_bytes()).hexdigest()
            != RECIPE_SHA256):
        raise ValueError("Frozen held-out50 selection, manifest, or recipe ordering differs")
    return parent, frozen


def preflight(arm: str) -> dict:
    if arm not in ARMS:
        raise ValueError(f"Unknown held-out50 arm: {arm}")
    parent, frozen = manifests()
    config = resolved_config(arm)
    if arm == "autonomous_v3":
        development_preflight(arm)
        memory = config["autonomous_support"]["memory"]
        corpus = read_json(Path(memory["corpus_path"]).expanduser())
        ids = {record.get("task_id") for record in corpus["records"]}
        hashes = {record.get("task_fingerprint") for record in corpus["records"]}
        if (PARENT_SHA256 not in corpus.get("excluded_manifest_sha256", {}).values() or
                any(task["task_id"] in ids or task["observation_sha256"] in hashes
                    for task in parent["tasks"])):
            raise ValueError("M2-200 source overlaps held-out tasks or lacks parent exclusion")
    return {"arm": arm, "manifest_sha256": FROZEN_SHA256,
            "parent_manifest_sha256": PARENT_SHA256,
            "config_sha256": canonical_digest(config), "task_count": len(frozen["tasks"]),
            "recipe_order_sha256": RECIPE_SHA256,
            "memory_checked": arm == "autonomous_v3", "package_versions": package_versions()}


def historical_run(path: str | Path, *, kind: str, reviewed_commit: str | None = None,
                   runtime: dict | None = None) -> dict:
    """Validate a complete read-only held-out run before any task is reused."""
    meta, config, snapshot, episodes = load_run(path)
    saved_summary = read_json(Path(path) / "summary.json")
    if saved_summary.get("successes") != sum(bool(e["success"]) for e in episodes.values()):
        raise ValueError("Historical summary disagrees with episode results")
    parent, frozen = manifests()
    expected = {task["task_id"]: task for task in parent["tasks"]}
    selected = {task["task_id"]: task for task in frozen["tasks"]}
    if (meta.get("condition") != config.get("condition") or
            meta.get("manifest_sha256") != PARENT_SHA256 or
            meta.get("textcraft_recipe_order_sha256") != RECIPE_SHA256 or
            meta.get("partition") != "heldout" or
            meta.get("package_versions", {}).get("textcraft") != "0.0.3" or
            (runtime is not None and (meta.get("python_version") != platform.python_version() or
                                      meta.get("package_versions") != runtime)) or
            any(config.get(key) != resolved_config("qwen_standalone").get(key)
                for key in AGENT_KEYS) or
            any(expected.get(task["task_id"]) != task for task in snapshot["tasks"])):
        raise ValueError("Historical run has incompatible tasks, Qwen settings, or runtime")
    if kind == "qwen_standalone":
        if (config.get("selection_policy", {}).get("id") != "never_v1" or
                config.get("do_sample") is not False or
                config.get("model_type") != "qwen" or
                config.get("supports") or meta.get("prompt_versions", {}).get("textcraft")
                not in (None, "single_command_v1")):
            raise ValueError("Historical Qwen run is not a standalone protocol")
    elif kind == "qwen_legacy_m2":
        if (config.get("condition") != "m2_trajectory_v1" or
                meta.get("trajectory_corpus_sha256") != LEGACY_M2_CORPUS_SHA256 or
                meta.get("trajectory_index_sha256") != LEGACY_M2_INDEX_SHA256):
            raise ValueError("Historical Qwen M2 corpus or condition differs")
    else:
        raise ValueError("Unknown historical baseline kind")
    if meta.get("git_commit") != git_commit() and meta.get("git_commit") != reviewed_commit:
        raise ValueError("Different historical commit requires --reviewed-legacy-commit")
    rows = {}
    for task_id, task in selected.items():
        episode = episodes.get(task_id)
        if episode is None:
            continue
        if (episode.get("initial_observation") != task["initial_observation"] or
                type(episode.get("success")) is not bool or
                not isinstance(episode.get("trace"), list) or
                digest_text(episode["initial_observation"]) != task["observation_sha256"]):
            raise ValueError(f"Historical observation differs: {task_id}")
        rows[task_id] = episode
    return {"kind": kind, "path": str(Path(path).resolve()),
            "run_id": meta["run_id"], "commit": meta["git_commit"],
            "config_sha256": meta["config_sha256"],
            "coverage": len(rows), "episodes_sha256": canonical_digest(rows),
            "episodes": rows}


def _plan_path(root: str | Path) -> Path:
    return Path(root) / "plan.json"


def initialize(root: str | Path, batch_size: int,
               historical_qwen: str | None = None,
               reviewed_legacy_commit: str | None = None,
               legacy_m2: str | None = None,
               reviewed_m2_commit: str | None = None) -> dict:
    if type(batch_size) is not int or not 1 <= batch_size <= 10:
        raise ValueError("batch_size must be 1 through 10")
    commit = require_clean_commit()
    checks = {arm: preflight(arm) for arm in ARMS}
    runtime = package_versions()
    if runtime.get("textcraft") != "0.0.3":
        raise ValueError("Execution requires TextCraft 0.0.3")
    sources = {}
    for kind, path, reviewed in (("qwen_standalone", historical_qwen, reviewed_legacy_commit),
                                 ("qwen_legacy_m2", legacy_m2, reviewed_m2_commit)):
        if path:
            source = historical_run(path, kind=kind, reviewed_commit=reviewed, runtime=runtime)
            sources[kind] = {key: value for key, value in source.items() if key != "episodes"}
    plan = {"schema_version": 1, "protocol": "textcraft_heldout50_v1",
            "batch_size": batch_size, "manifest_sha256": FROZEN_SHA256,
            "parent_manifest_sha256": PARENT_SHA256,
            "recipe_order_sha256": RECIPE_SHA256,
            "git_commit": commit, "python_version": platform.python_version(),
            "package_versions": runtime,
            "model_revisions": {
                "qwen_task_agent": resolved_config("qwen_standalone")["model_revision"],
                "gpt_oss_controller": resolved_config("autonomous_v3")
                ["autonomous_support"]["model"]["model"]["model_revision"]},
            "config_sha256": {arm: checks[arm]["config_sha256"] for arm in ARMS},
            "historical_sources": sources}
    destination = _plan_path(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return plan


def checked_plan(root: str | Path) -> dict:
    plan = read_json(_plan_path(root))
    if (plan.get("schema_version") != 1 or
            plan.get("protocol") != "textcraft_heldout50_v1" or
            plan.get("manifest_sha256") != FROZEN_SHA256 or
            plan.get("parent_manifest_sha256") != PARENT_SHA256 or
            plan.get("recipe_order_sha256") != RECIPE_SHA256 or
            plan.get("git_commit") != git_commit() or
            plan.get("python_version") != platform.python_version() or
            plan.get("package_versions") != package_versions() or
            plan.get("model_revisions") != {
                "qwen_task_agent": resolved_config("qwen_standalone")["model_revision"],
                "gpt_oss_controller": resolved_config("autonomous_v3")
                ["autonomous_support"]["model"]["model"]["model_revision"]} or
            type(plan.get("batch_size")) is not int or not 1 <= plan["batch_size"] <= 10):
        raise ValueError("Held-out50 plan differs from code, manifest, or runtime")
    manifests()
    for arm in ARMS:
        if plan["config_sha256"].get(arm) != canonical_digest(resolved_config(arm)):
            raise ValueError(f"Held-out50 {arm} configuration differs")
    for kind, source in plan["historical_sources"].items():
        checked = historical_run(source["path"], kind=kind,
                                 reviewed_commit=source["commit"],
                                 runtime=plan["package_versions"])
        if any(checked[key] != source[key] for key in source):
            raise ValueError(f"Historical {kind} source differs from locked plan")
    return plan


def batch_indices(batch: int, size: int) -> list[int]:
    if type(batch) is not int or batch < 0 or type(size) is not int or not 1 <= size <= 10:
        raise ValueError("Invalid batch or batch size")
    first = batch * size
    if first >= 50:
        raise ValueError("Batch is outside held-out50")
    return list(range(first, min(first + size, 50)))


def completed_episodes(root: str | Path, arm: str, plan: dict) -> dict[str, dict]:
    """Read verified task snapshots from complete or interrupted batched runs."""
    parent, frozen = manifests()
    tasks = {task["task_id"]: task for task in frozen["tasks"]}
    results: dict[str, dict] = {}
    batch_size = plan["batch_size"]
    for batch in range((50 + batch_size - 1) // batch_size):
        batch_root = Path(root) / "runs" / arm / f"batch_{batch:02d}"
        if not batch_root.exists():
            continue
        allowed = {frozen["tasks"][i]["task_id"] for i in batch_indices(batch, batch_size)}
        for directory in sorted(batch_root.glob("textcraft_*")):
            if not directory.is_dir():
                continue
            if not (directory / "run.json").exists():
                if list((directory / "episodes").glob("*.json")):
                    raise ValueError(f"Episodes without run metadata: {directory}")
                continue
            meta = read_json(directory / "run.json")
            config = read_json(directory / "config.json")
            selected = read_json(directory / "manifest.json")
            indices = meta.get("task_indices")
            if (meta.get("condition") != config.get("condition") or
                    config.get("condition") != resolved_config(arm)["condition"] or
                    meta.get("manifest_sha256") != PARENT_SHA256 or
                    meta.get("config_sha256") != plan["config_sha256"][arm] or
                    canonical_digest(config) != meta["config_sha256"] or
                    meta.get("selected_manifest_sha256") != canonical_digest(selected) or
                    meta.get("git_commit") != plan["git_commit"] or
                    meta.get("package_versions") != plan["package_versions"] or
                    meta.get("python_version") != plan["python_version"] or
                    meta.get("textcraft_recipe_order_sha256") != RECIPE_SHA256 or
                    not isinstance(indices, list) or not indices or
                    indices != sorted(set(indices)) or
                    any(type(i) is not int or i not in batch_indices(batch, batch_size)
                        for i in indices) or
                    selected != {**parent, "tasks": [parent["tasks"][i] for i in indices]}):
                raise ValueError(f"Incompatible batch run: {directory}")
            expected = {parent["tasks"][i]["task_id"] for i in indices}
            if not expected <= allowed:
                raise ValueError(f"Batch run contains tasks outside its batch: {directory}")
            for path in sorted((directory / "episodes").glob("*.json")):
                episode = read_json(path)
                task_id = episode.get("task_id")
                task = tasks.get(task_id)
                if (task_id not in expected or task_id in results or
                        episode.get("benchmark") != "textcraft" or
                        episode.get("initial_observation") != task["initial_observation"] or
                        digest_text(episode["initial_observation"]) != task["observation_sha256"] or
                        type(episode.get("success")) is not bool or
                        not isinstance(episode.get("trace"), list)):
                    raise ValueError(f"Duplicate or invalid batch episode: {path}")
                results[task_id] = episode
            if meta.get("status") == "complete" and (
                    meta.get("completed_task_count") != len(indices) or
                    len(list((directory / "episodes").glob("*.json"))) != len(indices)):
                raise ValueError(f"Complete run has missing episode snapshots: {directory}")
    return results


def coverage(root: str | Path, arm: str, plan: dict) -> dict[str, dict]:
    completed = completed_episodes(root, arm, plan)
    if arm == "qwen_standalone" and "qwen_standalone" in plan["historical_sources"]:
        source = plan["historical_sources"]["qwen_standalone"]
        historical = historical_run(source["path"], kind="qwen_standalone",
                                    reviewed_commit=source["commit"],
                                    runtime=plan["package_versions"])["episodes"]
        if set(completed) & set(historical):
            raise ValueError("Historical and new Qwen results overlap")
        completed.update(historical)
    return completed


def status(root: str | Path, arm: str, batch: int) -> dict:
    if arm not in ARMS:
        raise ValueError("Unknown arm")
    plan = checked_plan(root)
    _, frozen = manifests()
    indices = batch_indices(batch, plan["batch_size"])
    done = coverage(root, arm, plan)
    missing = [i for i in indices if frozen["tasks"][i]["task_id"] not in done]
    return {"arm": arm, "batch": batch, "completed": len(indices) - len(missing),
            "missing_indices": missing,
            "missing_task_ids": [frozen["tasks"][i]["task_id"] for i in missing]}


def run_batch(root: str | Path, arm: str, batch: int,
              max_tasks_per_attempt: int | None = None) -> dict:
    if arm not in ARMS:
        raise ValueError("Unknown arm")
    if (max_tasks_per_attempt is not None and
            (type(max_tasks_per_attempt) is not int or not 1 <= max_tasks_per_attempt <= 10)):
        raise ValueError("max_tasks_per_attempt must be 1 through 10")
    require_clean_commit()
    plan = checked_plan(root)
    checks = preflight(arm)
    if checks["package_versions"].get("textcraft") != "0.0.3":
        raise ValueError("TextCraft 0.0.3 is required for GPU execution")
    destination = Path(root) / "runs" / arm / f"batch_{batch:02d}"
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / ".batch.lock").open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Batch already running: {destination}") from exc
        current = status(root, arm, batch)
        if not current["missing_indices"]:
            return {**current, "run_dir": None, "reason": "already_complete"}
        parent, _ = manifests()
        selected = current["missing_indices"][:max_tasks_per_attempt]
        path = run(resolved_config(arm), parent, destination, task_indices=selected)
        updated = status(root, arm, batch)
        return {**updated, "run_dir": str(path),
                "reason": "completed" if not updated["missing_indices"] else "partial_batch"}


def aggregate(root: str | Path, arm: str) -> dict:
    if arm not in ARMS:
        raise ValueError("Unknown arm")
    plan = checked_plan(root)
    _, frozen = manifests()
    done = coverage(root, arm, plan)
    rows = []
    for task in frozen["tasks"]:
        episode = done.get(task["task_id"])
        if episode is None:
            continue
        row = {"task_id": task["task_id"], "seed": task["seed"],
               "observation_sha256": task["observation_sha256"]}
        for key in ("success", "status", "environment_actions", "failed_actions",
                    "model_decisions", "model_calls", "input_tokens", "output_tokens",
                    "inference_seconds", "elapsed_seconds", "environment_seconds",
                    "proposal_inference_seconds", "parser_errors",
                    "controller_model_calls", "controller_input_tokens",
                    "controller_output_tokens", "controller_inference_seconds",
                    "support_calls", "support_model_calls", "support_inference_seconds",
                    "support_input_tokens", "support_output_tokens", "support_parse_errors",
                    "actions_changed_by_support",
                    "controller_parse_errors", "controller_generation_failures",
                    "invalid_support_requests"):
            row[key] = episode.get(key)
        rows.append(row)
    numeric = ("environment_actions", "failed_actions", "model_decisions", "model_calls",
               "input_tokens", "output_tokens", "inference_seconds", "elapsed_seconds",
               "environment_seconds", "proposal_inference_seconds", "parser_errors",
               "controller_model_calls", "controller_input_tokens", "controller_output_tokens",
               "controller_inference_seconds", "support_calls", "support_model_calls",
               "support_inference_seconds", "support_input_tokens", "support_output_tokens",
               "support_parse_errors", "actions_changed_by_support", "controller_parse_errors",
               "controller_generation_failures", "invalid_support_requests")
    result = {"arm": arm, "protocol": plan["protocol"],
              "plan_sha256": canonical_digest(plan),
              "manifest_sha256": FROZEN_SHA256,
              "config_sha256": plan["config_sha256"][arm],
              "git_commit": plan["git_commit"],
              "historical_source": plan["historical_sources"].get("qwen_standalone")
              if arm == "qwen_standalone" else None,
              "complete": len(rows) == 50, "tasks_completed": len(rows),
              "successes": sum(bool(row["success"]) for row in rows),
              "success_rate_observed": (sum(bool(row["success"]) for row in rows) / len(rows)
                                        if rows else None),
              "totals": {key: round(sum(row.get(key) or 0 for row in rows), 3)
                         for key in numeric},
              "missing_task_ids": [task["task_id"] for task in frozen["tasks"]
                                   if task["task_id"] not in done],
              "tasks": rows}
    if arm == "autonomous_v3":
        from .support_sequence_analysis import analyze_episodes
        result["support_sequences"] = analyze_episodes(done)
    return result


def write_aggregate(root: str | Path, arm: str, output: str | Path) -> dict:
    result = aggregate(root, arm)
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"Aggregate already exists: {destination}")
    write_json_atomic(destination, result)
    return {key: value for key, value in result.items()
            if key not in {"tasks", "support_sequences"}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("preflight")
    p.add_argument("--arm", choices=ARMS, required=True)
    p = sub.add_parser("init")
    p.add_argument("--root", required=True)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--historical-qwen-run")
    p.add_argument("--reviewed-legacy-commit")
    p.add_argument("--legacy-m2-run")
    p.add_argument("--reviewed-m2-commit")
    for name in ("status", "run-batch", "aggregate"):
        p = sub.add_parser(name)
        p.add_argument("--root", required=True)
        p.add_argument("--arm", choices=ARMS, required=True)
        if name in {"status", "run-batch"}:
            p.add_argument("--batch", type=int, required=True)
        if name == "run-batch":
            p.add_argument("--max-tasks-per-attempt", type=int)
        if name == "aggregate":
            p.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        result = preflight(args.arm)
    elif args.command == "init":
        result = initialize(args.root, args.batch_size, args.historical_qwen_run,
                            args.reviewed_legacy_commit, args.legacy_m2_run,
                            args.reviewed_m2_commit)
    elif args.command == "status":
        result = status(args.root, args.arm, args.batch)
    elif args.command == "run-batch":
        result = run_batch(args.root, args.arm, args.batch, args.max_tasks_per_attempt)
    else:
        result = write_aggregate(args.root, args.arm, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
