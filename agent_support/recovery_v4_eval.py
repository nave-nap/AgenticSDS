"""Frozen 30-task exploratory recovery evaluation, smoke, and safe resume."""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

from .autonomous_evaluation import FROZEN_MANIFEST_SHA256, require_clean_commit
from .autonomous_memory import FrozenTrajectoryRetrieval
from .common import canonical_digest, read_json, validate_manifest, write_json_atomic
from .controller_backend import GptOssControllerWorker
from .heldout50_evaluation import FROZEN_SHA256, RECIPE_SHA256
from .model import QwenModel
from .recovery_v4 import ARMS, run_recovery_episode
from .runner import package_versions
from .trajectory_memory import MPNetEmbedder

REPO = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO / "manifests/textcraft/autonomous_recovery30_v1.json"
MANIFEST_SHA256 = "150c7bbd973936e0f3ef8fe1266060e88de0181f0068ce98bc7bce0989a2d562"
CONFIG_DIR = REPO / "configs/autonomous_recovery_v4"
PROTOCOL = "textcraft_autonomous_recovery30_v1"
BATCH_SIZE = 3


def manifest() -> dict:
    value = read_json(MANIFEST_PATH)
    validate_manifest(value)
    dev = read_json(REPO / "manifests/textcraft/development.json")
    held = read_json(REPO / "manifests/textcraft/heldout.json")
    frozen50 = read_json(REPO / "manifests/textcraft/heldout50_v1.json")
    other = {task["observation_sha256"] for task in held["tasks"]}
    other_ids = {task["task_id"] for task in held["tasks"]}
    if (canonical_digest(value) != MANIFEST_SHA256 or value.get("partition") != "development" or
            len(value["tasks"]) != 30 or
            value["tasks"][:20] != dev["tasks"] or
            value.get("source_manifest_sha256") != FROZEN_MANIFEST_SHA256 or
            [task["seed"] for task in value["tasks"][20:]] != list(range(200, 210)) or
            any(task["observation_sha256"] in other or task["task_id"] in other_ids
                for task in value["tasks"]) or
            FROZEN_SHA256 != canonical_digest(frozen50) or
            hashlib.sha256((REPO / "agent_support/textcraft_recipe_order.json").read_bytes()).hexdigest()
            != RECIPE_SHA256):
        raise ValueError("Recovery development manifest or recipe order differs")
    return value


def config_for(arm: str) -> dict:
    if arm not in ARMS:
        raise ValueError("Unknown recovery arm")
    value = read_json(CONFIG_DIR / f"{arm}.json")
    v3 = read_json(REPO / "configs/autonomous_evaluation/autonomous_v3.json")
    expected = copy.deepcopy(v3)
    expected.update(schema_version=1, arm=arm, max_actions=150, max_decisions=220,
                    limits={"wall_seconds": 2400, "inference_seconds": 2100,
                            "max_model_calls_total": 720})
    expected["autonomous_support"]["budget"] = {
        "max_operations_per_decision": 7, "max_operations_per_task": 100,
        "max_additional_model_calls": 500, "max_additional_tokens": 3000000,
        "max_depth": 10}
    if arm == ARMS[1]:
        expected["condition"] = "autonomous_recovery_v4"
        expected["autonomous_support"]["operations"] += [
            "inspect_candidate_v1", "analyze_failures_v1", "verify_prerequisites_v1"]
    if value != expected:
        raise ValueError("Recovery config differs from versioned v3-derived protocol")
    return value


def _memory_paths(config: dict) -> tuple[Path, Path]:
    root = os.environ.get("AGENT_SUPPORT_M2_200_ARTIFACT_DIR")
    if root:
        location = Path(root).expanduser()
        if not location.is_absolute():
            raise ValueError("Memory artifact root must be absolute")
        return location / "corpus_200.json", location / "index_200.json"
    spec = config["autonomous_support"]["memory"]
    return Path(spec["corpus_path"]).expanduser(), Path(spec["index_path"]).expanduser()


def memory_checks(config: dict, require_files: bool = False) -> dict:
    corpus_path, index_path = _memory_paths(config)
    if not corpus_path.is_file() or not index_path.is_file():
        if require_files:
            raise ValueError("Pinned M2-200 memory artifacts are unavailable")
        return {"available": False, "corpus_path": str(corpus_path), "index_path": str(index_path)}
    corpus, index = read_json(corpus_path), read_json(index_path)
    spec = config["autonomous_support"]["memory"]
    selected = manifest()["tasks"]
    ids = {task["task_id"] for task in selected}
    hashes = {task["observation_sha256"] for task in selected}
    if (canonical_digest(corpus) != spec["corpus_sha256"] or
            canonical_digest(index) != spec["index_sha256"] or
            len(corpus.get("records", [])) != spec["expected_trajectories"] or
            any(record.get("task_id") in ids or record.get("task_fingerprint") in hashes
                for record in corpus["records"])):
        raise ValueError("Memory artifacts differ or overlap recovery tasks")
    return {"available": True, "corpus_path": str(corpus_path),
            "index_path": str(index_path), "corpus_sha256": spec["corpus_sha256"],
            "index_sha256": spec["index_sha256"], "overlap": 0}


def load_memory(config: dict, qwen) -> FrozenTrajectoryRetrieval:
    memory_checks(config, require_files=True)
    corpus_path, index_path = _memory_paths(config)
    corpus, index = read_json(corpus_path), read_json(index_path)
    spec = config["autonomous_support"]["memory"]
    tokenizer = qwen.tokenizer
    return FrozenTrajectoryRetrieval(corpus, index,
        MPNetEmbedder(index["embedding_model"], revision=index.get("embedding_revision")),
        spec["max_top_k"], spec["max_result_tokens"],
        lambda value: len(tokenizer.encode(value, add_special_tokens=False)))


def preflight(arm: str) -> dict:
    selected = manifest()
    config = config_for(arm)
    spec = config["autonomous_support"]["model"]
    python = os.environ.get(spec["python_env"])
    cache = os.environ.get(spec["hf_home_env"])
    versions = package_versions()
    checks = {"textcraft_0_0_3": versions.get("textcraft") == "0.0.3",
              "qwen_dependencies_present": all(versions.get(name) for name in
                  ("torch", "transformers", "bitsandbytes")),
              "gptoss_python_exists": bool(python and Path(python).is_file()),
              "gptoss_cache_exists": bool(cache and Path(cache).is_dir())}
    return {"arm": arm, "protocol": PROTOCOL, "task_count": len(selected["tasks"]),
            "manifest_sha256": MANIFEST_SHA256, "recipe_order_sha256": RECIPE_SHA256,
            "config_sha256": canonical_digest(config), "memory": memory_checks(config),
            "runtime_checks": checks, "package_versions": versions}


def initialize(root: str | Path) -> dict:
    commit = require_clean_commit()
    checks = {arm: preflight(arm) for arm in ARMS}
    if any(not all(item["runtime_checks"].values()) or not item["memory"]["available"]
           for item in checks.values()):
        raise ValueError("Locust preflight requires Qwen, GPT-OSS, TextCraft, and disjoint memory")
    plan = {"schema_version": 1, "protocol": PROTOCOL, "git_commit": commit,
            "manifest_sha256": MANIFEST_SHA256, "recipe_order_sha256": RECIPE_SHA256,
            "batch_size": BATCH_SIZE,
            "config_sha256": {arm: checks[arm]["config_sha256"] for arm in ARMS},
            "memory": checks[ARMS[0]]["memory"],
            "python_version": platform.python_version(), "package_versions": package_versions()}
    path = Path(root) / "plan.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return plan


def checked_plan(root: str | Path) -> dict:
    plan = read_json(Path(root) / "plan.json")
    manifest()
    if (plan.get("schema_version") != 1 or plan.get("protocol") != PROTOCOL or
            plan.get("git_commit") != require_clean_commit() or
            plan.get("manifest_sha256") != MANIFEST_SHA256 or
            plan.get("recipe_order_sha256") != RECIPE_SHA256 or
            plan.get("batch_size") != BATCH_SIZE or
            plan.get("python_version") != platform.python_version() or
            plan.get("package_versions") != package_versions() or
            plan.get("memory") != memory_checks(config_for(ARMS[0]), require_files=True) or
            any(plan["config_sha256"].get(arm) != canonical_digest(config_for(arm)) for arm in ARMS)):
        raise ValueError("Recovery plan differs from committed protocol or runtime")
    return plan


def completed(root: str | Path, arm: str, plan: dict) -> dict:
    tasks = {task["task_id"]: (index, task) for index, task in enumerate(manifest()["tasks"])}
    found = {}
    for run_dir in sorted((Path(root) / "runs" / arm).glob("batch_*/textcraft_*")):
        if not run_dir.is_dir():
            continue
        if not (run_dir / "run.json").exists():
            if list((run_dir / "episodes").glob("*.json")):
                raise ValueError(f"Episodes without run metadata: {run_dir}")
            continue
        meta = read_json(run_dir / "run.json")
        config = read_json(run_dir / "config.json")
        snapshot = read_json(run_dir / "manifest.json")
        indices = meta.get("task_indices")
        if (meta.get("arm") != arm or meta.get("git_commit") != plan["git_commit"] or
                meta.get("manifest_sha256") != MANIFEST_SHA256 or
                meta.get("config_sha256") != plan["config_sha256"][arm] or
                meta.get("recipe_order_sha256") != RECIPE_SHA256 or
                meta.get("python_version") != plan["python_version"] or
                meta.get("package_versions") != plan["package_versions"] or
                config != config_for(arm) or
                not isinstance(indices, list) or not indices or
                indices != sorted(set(indices)) or
                snapshot != {"schema_version": 1, "benchmark": "textcraft",
                             "partition": "development",
                             "tasks": [task for index, task in enumerate(manifest()["tasks"])
                                       if index in indices]} or
                any(type(i) is not int or not 0 <= i < 30 or
                    run_dir.parent.name != f"batch_{i // BATCH_SIZE:02d}" for i in indices)):
            raise ValueError(f"Incompatible recovery run: {run_dir}")
        paths = sorted((run_dir / "episodes").glob("*.json"))
        if (meta.get("status") == "complete" and
                (len(paths) != len(indices) or meta.get("completed_task_count") != len(indices))):
            raise ValueError(f"Complete run has missing snapshots: {run_dir}")
        for path in paths:
            episode = read_json(path)
            task_id = episode.get("task_id")
            if task_id not in tasks or task_id in found:
                raise ValueError(f"Invalid or duplicate recovery task: {path}")
            index, task = tasks[task_id]
            if (index not in indices or not path.name.startswith(f"{index:04d}_") or
                    episode.get("initial_observation") != task["initial_observation"] or
                    type(episode.get("success")) is not bool or
                    episode.get("arm") != arm or not isinstance(episode.get("trace"), list)):
                raise ValueError(f"Invalid recovery episode: {path}")
            found[task_id] = episode
    return found


def status(root: str | Path, arm: str, batch: int) -> dict:
    if arm not in ARMS or type(batch) is not int or not 0 <= batch < 10:
        raise ValueError("Invalid arm or batch")
    plan = checked_plan(root)
    done = completed(root, arm, plan)
    tasks = manifest()["tasks"]
    indices = range(batch * BATCH_SIZE, (batch + 1) * BATCH_SIZE)
    missing = [i for i in indices if tasks[i]["task_id"] not in done]
    return {"arm": arm, "batch": batch, "completed": BATCH_SIZE - len(missing),
            "missing_indices": missing}


def _run_tasks(root: Path, arm: str, indices: list[int], plan: dict,
               *, qwen_factory=QwenModel, gpt_factory=GptOssControllerWorker,
               memory_factory=load_memory, episode_runner=run_recovery_episode) -> Path:
    config = config_for(arm)
    tasks = manifest()["tasks"]
    run_dir = root / ("textcraft_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + os.urandom(4).hex())
    (run_dir / "episodes").mkdir(parents=True)
    (run_dir / "checkpoints").mkdir()
    write_json_atomic(run_dir / "config.json", config)
    write_json_atomic(run_dir / "manifest.json", {"schema_version": 1,
        "benchmark": "textcraft", "partition": "development", "tasks": [tasks[i] for i in indices]})
    meta = {"schema_version": 1, "arm": arm, "protocol": PROTOCOL, "status": "running",
            "manifest_sha256": MANIFEST_SHA256, "config_sha256": plan["config_sha256"][arm],
            "recipe_order_sha256": RECIPE_SHA256, "git_commit": plan["git_commit"],
            "python_version": plan["python_version"], "package_versions": plan["package_versions"],
            "task_indices": indices, "completed_task_count": 0,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "started_utc": datetime.now(timezone.utc).isoformat()}
    write_json_atomic(run_dir / "run.json", meta)
    gpt = None
    try:
        gpt = gpt_factory(config["autonomous_support"]["model"], run_dir / "workers")
        meta["gptoss_metadata"] = gpt.metadata()
        if "L40S" not in meta["gptoss_metadata"].get("gpu_name", ""):
            raise ValueError("Recovery evaluation requires an L40S")
        write_json_atomic(run_dir / "run.json", meta)
        qwen = qwen_factory(config)
        memory = memory_factory(config, qwen)
        for index in indices:
            task = tasks[index]
            gpt.begin_episode(task["task_id"], config["seed"])
            checkpoint_path = run_dir / "checkpoints" / f"{index:04d}.json"
            def checkpoint(value, destination=checkpoint_path):
                write_json_atomic(destination, value)
            result = episode_runner(task, config, qwen, gpt, memory=memory,
                                    checkpoint=checkpoint)
            write_json_atomic(run_dir / "episodes" / f"{index:04d}_{hashlib.sha256(task['task_id'].encode()).hexdigest()[:12]}.json", result)
            meta["completed_task_count"] += 1
            write_json_atomic(run_dir / "run.json", meta)
            print(f"{task['task_id']} {result['status']} actions={result['environment_actions']}", flush=True)
        meta["status"] = "complete"
    except Exception as exc:
        meta.update(status="failed", error=repr(exc))
        raise
    finally:
        if gpt is not None:
            gpt.close()
        meta["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json_atomic(run_dir / "run.json", meta)
    return run_dir


def run_batch(root: str | Path, arm: str, batch: int, max_tasks_per_attempt: int = BATCH_SIZE) -> dict:
    if (arm not in ARMS or type(batch) is not int or not 0 <= batch < 10 or
            type(max_tasks_per_attempt) is not int or not 1 <= max_tasks_per_attempt <= BATCH_SIZE):
        raise ValueError("Invalid arm, batch, or attempt size")
    plan = checked_plan(root)
    directory = Path(root) / "runs" / arm / f"batch_{batch:02d}"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".batch.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        indices = status(root, arm, batch)["missing_indices"][:max_tasks_per_attempt]
        if not indices:
            return {**status(root, arm, batch), "run_dir": None}
        path = _run_tasks(directory, arm, indices, plan)
        return {**status(root, arm, batch), "run_dir": str(path)}


def smoke(root: str | Path, arm: str, count: int = 2) -> Path:
    if arm not in ARMS or count not in (2, 3):
        raise ValueError("Smoke uses 2 or 3 development tasks")
    # No held-out task is involved. Smoke output is separate from the planned 30-task batches.
    plan = {"git_commit": require_clean_commit(),
            "config_sha256": {arm: canonical_digest(config_for(arm))},
            "python_version": platform.python_version(), "package_versions": package_versions()}
    root = Path(root) / "smoke" / arm
    root.mkdir(parents=True, exist_ok=True)
    return _run_tasks(root, arm, list(range(count)), plan)


def aggregate(root: str | Path, arm: str) -> dict:
    plan = checked_plan(root)
    tasks = manifest()["tasks"]
    done = completed(root, arm, plan)
    rows = []
    for task in tasks:
        episode = done.get(task["task_id"])
        if episode is None:
            continue
        checkpoints = {str(limit): bool(episode["success"] and episode["environment_actions"] <= limit)
                       for limit in (30, 60, 90, 120, 150)}
        rows.append({"task_id": task["task_id"], "observation_sha256": task["observation_sha256"],
                     "seed": task["seed"], "success_checkpoints": checkpoints,
                     **{key: value for key, value in episode.items()
                        if key not in {"trace", "initial_observation", "task_prompt", "recovery_state_final"}}})
    numeric = ("environment_actions", "failed_actions", "model_decisions", "model_calls",
               "qwen_model_calls", "gptoss_model_calls", "embedding_model_calls",
               "controller_model_calls",
               "support_model_calls", "input_tokens", "output_tokens", "inference_seconds",
               "qwen_input_tokens", "qwen_output_tokens", "qwen_inference_seconds",
               "gptoss_inference_seconds", "embedding_inference_seconds",
               "controller_input_tokens", "controller_output_tokens", "controller_inference_seconds",
               "support_input_tokens", "support_output_tokens", "support_inference_seconds",
               "support_calls", "environment_seconds", "elapsed_seconds", "parser_errors",
               "controller_errors", "strategy_changes", "replanning_events",
               "progress_assessments", "support_useful_judgments", "qwen_generation_failures")
    return {"arm": arm, "protocol": PROTOCOL, "manifest_sha256": MANIFEST_SHA256,
            "plan_sha256": canonical_digest(plan), "complete": len(rows) == 30,
            "tasks_completed": len(rows), "successes": sum(bool(row["success"]) for row in rows),
            "success_rate_observed": sum(bool(row["success"]) for row in rows) / len(rows) if rows else None,
            "success_at_checkpoints": {str(limit): sum(row["success_checkpoints"][str(limit)] for row in rows)
                                       for limit in (30, 60, 90, 120, 150)},
            "totals": {key: round(sum(row.get(key, 0) for row in rows), 3) for key in numeric},
            "missing_task_ids": [task["task_id"] for task in tasks if task["task_id"] not in done],
            "tasks": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("preflight")
    p.add_argument("--arm", choices=ARMS, required=True)
    p = sub.add_parser("init")
    p.add_argument("--root", required=True)
    p = sub.add_parser("smoke")
    p.add_argument("--root", required=True)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--count", type=int, default=2)
    for name in ("status", "run-batch", "aggregate"):
        p = sub.add_parser(name)
        p.add_argument("--root", required=True)
        p.add_argument("--arm", choices=ARMS, required=True)
        if name in {"status", "run-batch"}:
            p.add_argument("--batch", type=int, required=True)
        if name == "run-batch":
            p.add_argument("--max-tasks-per-attempt", type=int, default=BATCH_SIZE)
        if name == "aggregate":
            p.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        result = preflight(args.arm)
    elif args.command == "init":
        result = initialize(args.root)
    elif args.command == "smoke":
        result = {"run_dir": str(smoke(args.root, args.arm, args.count))}
    elif args.command == "status":
        result = status(args.root, args.arm, args.batch)
    elif args.command == "run-batch":
        result = run_batch(args.root, args.arm, args.batch, args.max_tasks_per_attempt)
    else:
        report = aggregate(args.root, args.arm)
        if Path(args.output).exists():
            raise FileExistsError(args.output)
        write_json_atomic(args.output, report)
        result = {key: value for key, value in report.items() if key != "tasks"}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
