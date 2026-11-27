"""Isolated, matched four-policy TextCraft recovery evaluation CLI."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .autonomous_evaluation import require_clean_commit
from .common import canonical_digest, read_json, write_json_atomic
from .controller_backend import GptOssControllerWorker
from .gpu_memory_diagnostics import enabled as gpu_memory_diagnostics_enabled
from .gpt_oss_chunked_eager import ENABLE_ENV as CHUNKED_ENV, NAME as CHUNKED_NAME
from .gpt_oss_two_gpu import SHARD_ENV, VISIBLE_ENV
from .heldout50_analysis import paired_interval, wilson
from .model import QwenModel
from .recovery_policy_registry import (POLICIES, make_policy, policy_configuration,
                                       policy_digest)
from .recovery_policy_telemetry import annotate_episode, episode_telemetry
from .recovery_v4 import TERMINAL_WORKER_STATUSES, run_recovery_episode
from .recovery_v4_eval import (MANIFEST_SHA256, RECIPE_SHA256, config_for,
                               load_memory, manifest, memory_checks,
                               preflight as v4_preflight)
from .runner import package_versions
from .three_gpu_feasibility import preflight as feasibility_preflight

PROTOCOL = "textcraft_recovery_policy30_terminal_review_three_gpu_v1"
BASE_ARM = "autonomous_recovery_v4"
REQUESTED_ATTENTION = CHUNKED_NAME
TRACEBACK_ENV = "AGENT_SUPPORT_GPTOSS_WORKER_TRACEBACKS"
BATCH_SIZE = 3
TASK_COUNT = 30
STAGES = {"smoke": (0,), "pilot": (0, 1, 2)}
RUNTIME_FILES = ("agent_support/controller_backend.py",
                 "agent_support/gpt_oss_controller_worker.py",
                 "agent_support/gpt_oss_model.py",
                 "agent_support/gpt_oss_two_gpu.py",
                 "agent_support/gpt_oss_chunked_eager.py",
                 "agent_support/three_gpu_feasibility.py",
                 "agent_support/model.py")


def feasibility_gate(path: str | Path) -> dict:
    """Accept a prior validated request only with identical model runtime source."""
    result = read_json(path)
    expected = feasibility_preflight()
    generation = result.get("generation") or {}
    if (result.get("status") != "infrastructure_valid" or
            any(result.get(key) != value for key, value in expected.items()) or
            generation.get("input_tokens") != 8887 or
            not isinstance(generation.get("output_tokens"), int) or
            generation["output_tokens"] < 1 or
            generation.get("generation_status") != "complete" or
            result.get("forward_execution", {}).get("prefill_layers_completed") != 24 or
            result.get("qwen_hf_device_map") != {"": "0"}):
        raise ValueError("Full three-GPU feasibility evidence is invalid or incomplete")
    source_commit = result.get("git_commit")
    if not isinstance(source_commit, str) or not source_commit:
        raise ValueError("Feasibility source commit is missing")
    comparison = subprocess.run(["git", "diff", "--quiet", source_commit,
                                 require_clean_commit(), "--", *RUNTIME_FILES,
                                 "configs", "manifests"], cwd=Path(__file__).resolve().parent.parent)
    if comparison.returncode != 0:
        raise ValueError("Model runtime or frozen inputs differ from feasibility source commit")
    return {"status": "gate_passed", "feasibility_result": str(path),
            "feasibility_git_commit": source_commit}


def preflight(policy_id: str) -> dict:
    policy = make_policy(policy_id)
    base = v4_preflight(BASE_ARM)  # CPU-only; also verifies frozen config and manifest.
    config = config_for(BASE_ARM)
    return {"protocol": PROTOCOL, "policy_id": policy_id,
        "policy_configuration": policy_configuration(policy_id),
        "policy_sha256": policy_digest(policy_id),
        "selector_matches_id": policy is None if policy_id == "original_v4"
                               else policy.policy_id == policy_id,
        "base_arm": BASE_ARM, "base_config_sha256": canonical_digest(config),
        "manifest_sha256": MANIFEST_SHA256,
        "requested_attention_implementation": REQUESTED_ATTENTION,
        "runtime_feature_checks": {
            "chunked_eager_opted_in": os.environ.get(CHUNKED_ENV) == "1",
            "two_gpu_sharding_opted_in": os.environ.get(SHARD_ENV) == "1",
            "gpu_memory_diagnostics_opted_in": gpu_memory_diagnostics_enabled(),
            "three_distinct_visible_devices": (
                len([x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x]) == 1 and
                len([x for x in os.environ.get(VISIBLE_ENV, "").split(",") if x]) == 2 and
                len(set([os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                         *os.environ.get(VISIBLE_ENV, "").split(",")])) == 3),
            "worker_tracebacks_enabled": os.environ.get(TRACEBACK_ENV) == "1"},
        "model_settings": {"qwen": {key: config[key] for key in
            ("model_id", "model_revision", "quantization", "max_new_tokens", "seed")},
            "gpt_oss": config["autonomous_support"]["model"],
            "support_max_new_tokens": config["autonomous_support"]["support_max_new_tokens"]},
        "limits": {"max_actions": config["max_actions"],
                   "max_decisions": config["max_decisions"],
                   "autonomous_support": config["autonomous_support"]["budget"],
                   **config["limits"]},
        "v4_preflight": base}


def runtime_preflight_passed(check: dict) -> bool:
    base = check["v4_preflight"]
    return (check["selector_matches_id"] and
            all(check["runtime_feature_checks"].values()) and
            all(base["protocol_checks"].values()) and
            all(base["runtime_checks"].values()) and
            base["memory"]["available"])


def _plan(require_runtime: bool) -> dict:
    commit = require_clean_commit()
    checks = {policy: preflight(policy) for policy in POLICIES}
    if require_runtime and not all(runtime_preflight_passed(check)
                                   for check in checks.values()):
        raise ValueError("Policy evaluation runtime preflight failed")
    base = checks["original_v4"]
    return {"schema_version": 1, "protocol": PROTOCOL,
        "implementation_commit": commit, "base_arm": BASE_ARM,
        "requested_attention_implementation": REQUESTED_ATTENTION,
        "worker_tracebacks_enabled": True,
        "base_config_sha256": base["base_config_sha256"],
        "manifest_sha256": MANIFEST_SHA256,
        "recipe_order_sha256": RECIPE_SHA256,
        "batch_size": BATCH_SIZE, "task_count": TASK_COUNT,
        "policy_configurations": {key: checks[key]["policy_configuration"] for key in POLICIES},
        "policy_sha256": {key: checks[key]["policy_sha256"] for key in POLICIES},
        "model_settings": base["model_settings"], "limits": base["limits"],
        "memory": base["v4_preflight"]["memory"],
        "python_version": platform.python_version(),
        "package_versions": package_versions()}


def initialize(root: str | Path) -> dict:
    for stage in STAGES:
        stage_report(root, stage)
    plan = _plan(require_runtime=True)
    path = Path(root) / "plan.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return plan


def checked_plan(root: str | Path) -> dict:
    saved = read_json(Path(root) / "plan.json")
    if saved != _plan(require_runtime=True):
        raise ValueError("Policy plan differs from committed protocol or runtime")
    return saved


def _placement_valid(meta: dict) -> bool:
    worker = meta.get("gptoss_metadata", {})
    return (worker.get("sharding") == "explicit_decoder_layers_12_12" and
            worker.get("verified_expert_modules_by_device") == {"cuda:0": 12, "cuda:1": 12} and
            len(worker.get("gpu_names", [])) == 2 and
            all("L40S" in name for name in worker["gpu_names"]) and
            meta.get("qwen_hf_device_map") == {"": "0"})


def _run_tasks(root: Path, policy_id: str, indices: list[int], plan: dict,
               *, qwen_factory=QwenModel, gpt_factory=GptOssControllerWorker,
               memory_factory=load_memory, episode_runner=run_recovery_episode,
               shared_models=None) -> Path:
    if policy_id not in POLICIES or not indices or indices != sorted(set(indices)) or any(
            type(index) is not int or not 0 <= index < TASK_COUNT for index in indices):
        raise ValueError("Invalid policy or task indices")
    config = config_for(BASE_ARM)
    tasks = manifest()["tasks"]
    run_dir = root / ("textcraft_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                      + "_" + os.urandom(4).hex())
    (run_dir / "episodes").mkdir(parents=True)
    (run_dir / "checkpoints").mkdir()
    write_json_atomic(run_dir / "config.json", config)
    write_json_atomic(run_dir / "policy.json", policy_configuration(policy_id))
    write_json_atomic(run_dir / "manifest.json", {"schema_version": 1,
        "benchmark": "textcraft", "partition": "development",
        "tasks": [tasks[index] for index in indices]})
    meta = {"schema_version": 1, "protocol": PROTOCOL, "status": "running",
        "policy_id": policy_id, "policy_sha256": plan["policy_sha256"][policy_id],
        "policy_configuration": policy_configuration(policy_id),
        "implementation_commit": plan["implementation_commit"],
        "requested_attention_implementation": plan["requested_attention_implementation"],
        "worker_tracebacks_enabled": plan["worker_tracebacks_enabled"],
        "base_arm": BASE_ARM, "base_config_sha256": plan["base_config_sha256"],
        "model_settings": plan["model_settings"], "limits": plan["limits"],
        "manifest_sha256": MANIFEST_SHA256, "recipe_order_sha256": RECIPE_SHA256,
        "memory": plan["memory"], "python_version": plan["python_version"],
        "package_versions": plan["package_versions"], "task_indices": indices,
        "completed_task_count": 0, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "started_utc": datetime.now(timezone.utc).isoformat()}
    write_json_atomic(run_dir / "run.json", meta)
    gpt = None
    try:
        gpt = (shared_models[1] if shared_models is not None else
               gpt_factory(config["autonomous_support"]["model"], run_dir / "workers"))
        meta["gptoss_metadata"] = gpt.metadata()
        meta["attention_implementation"] = meta["gptoss_metadata"].get("attention_implementation")
        if "L40S" not in meta["gptoss_metadata"].get("gpu_name", ""):
            raise ValueError("Recovery policy evaluation requires an L40S")
        if meta["attention_implementation"] != plan["requested_attention_implementation"]:
            raise ValueError("GPT-OSS active attention backend differs from requested backend")
        if (meta["gptoss_metadata"].get("sharding") != "explicit_decoder_layers_12_12" or
                meta["gptoss_metadata"].get("verified_expert_modules_by_device") !=
                {"cuda:0": 12, "cuda:1": 12} or
                len(meta["gptoss_metadata"].get("gpu_names", [])) != 2 or
                any("L40S" not in name for name in meta["gptoss_metadata"]["gpu_names"])):
            raise ValueError("GPT-OSS did not verify two L40S 12/12 placement")
        write_json_atomic(run_dir / "run.json", meta)
        qwen = shared_models[0] if shared_models is not None else qwen_factory(config)
        qwen_map = getattr(getattr(qwen, "model", None), "hf_device_map", None)
        if qwen_map not in ({"": 0}, {"": "0"}):
            raise ValueError("Qwen did not remain wholly on GPU 0")
        meta["qwen_hf_device_map"] = {str(key): str(value) for key, value in qwen_map.items()}
        write_json_atomic(run_dir / "run.json", meta)
        if gpu_memory_diagnostics_enabled():
            configure = getattr(qwen, "enable_gpu_memory_diagnostics", None)
            if callable(configure):
                configure(run_dir / "diagnostics")
        memory = shared_models[2] if shared_models is not None else memory_factory(config, qwen)
        for index in indices:
            task = tasks[index]
            gpt.begin_episode(task["task_id"], config["seed"])
            checkpoint_path = run_dir / "checkpoints" / f"{index:04d}.json"
            telemetry_so_far = []
            def checkpoint(value, destination=checkpoint_path):
                snapshot = dict(value)
                snapshot["policy_id"] = policy_id
                snapshot["implementation_commit"] = plan["implementation_commit"]
                telemetry_so_far.extend(episode_telemetry(
                    snapshot["trace"][len(telemetry_so_far):], policy_id))
                snapshot["decision_telemetry"] = list(telemetry_so_far)
                write_json_atomic(destination, snapshot)
            result = episode_runner(task, config, qwen, gpt, memory=memory,
                                    checkpoint=checkpoint, policy=make_policy(policy_id))
            result = annotate_episode(result, policy_id, plan["implementation_commit"], PROTOCOL)
            write_json_atomic(run_dir / "episodes" /
                f"{index:04d}_{hashlib.sha256(task['task_id'].encode()).hexdigest()[:12]}.json", result)
            meta["completed_task_count"] += 1
            write_json_atomic(run_dir / "run.json", meta)
            print(f"{policy_id} {task['task_id']} {result['status']} "
                  f"actions={result['environment_actions']}", flush=True)
            if result["status"] in TERMINAL_WORKER_STATUSES:
                meta.update(status="stopped_worker_failure", stop_reason=result["status"])
                write_json_atomic(run_dir / "run.json", meta)
                break
            if result["infrastructure_invalid"]:
                meta.update(status="stopped_infrastructure_failure",
                            stop_reason="infrastructure_invalid_episode",
                            stop_task_id=task["task_id"])
                write_json_atomic(run_dir / "run.json", meta)
                break
        else:
            meta["status"] = "complete"
    except Exception as exc:
        meta.update(status="failed", error=repr(exc))
        raise
    finally:
        if gpt is not None and shared_models is None:
            gpt.close()
        meta["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json_atomic(run_dir / "run.json", meta)
    return run_dir


def run_stage(root: str | Path, policy_id: str, stage: str) -> Path:
    if policy_id not in POLICIES or stage not in STAGES:
        raise ValueError("Unknown policy or stage")
    if stage == "pilot":
        stage_report(root, "smoke")
    feasibility_gate(os.environ["POLICY_FEASIBILITY_RESULT"])
    plan = _plan(require_runtime=True)
    directory = Path(root) / stage / policy_id
    directory.mkdir(parents=True, exist_ok=True)
    if list(directory.glob("textcraft_*")):
        raise FileExistsError(f"Stage already has a run for {policy_id}: {directory}")
    return _run_tasks(directory, policy_id, list(STAGES[stage]), plan)


def run_matched_stage(root: str | Path, stage: str, policy_ids: tuple[str, ...],
                      *, qwen_factory=QwenModel, gpt_factory=GptOssControllerWorker,
                      memory_factory=load_memory, runner=_run_tasks) -> dict:
    """Run a bounded matched stage with one load of each pinned model."""
    if stage not in STAGES or not policy_ids or len(set(policy_ids)) != len(policy_ids) or \
            any(policy not in POLICIES for policy in policy_ids):
        raise ValueError("Invalid matched stage or policies")
    if stage == "smoke" and policy_ids != POLICIES:
        raise ValueError("Matched smoke requires all four frozen policies")
    if stage == "pilot":
        stage_report(root, "smoke")
        if len(policy_ids) > 2:
            raise ValueError("Pilot allocation is bounded to two policies")
    feasibility_gate(os.environ["POLICY_FEASIBILITY_RESULT"])
    plan = _plan(require_runtime=True)
    root = Path(root)
    for policy in policy_ids:
        if (root / stage / policy).exists():
            raise FileExistsError(f"Policy stage output already exists: {stage}/{policy}")
    attempt = root / "allocations" / (f"{stage}_job_{os.environ.get('SLURM_JOB_ID', 'local')}_"
                                      f"restart_{os.environ.get('SLURM_RESTART_COUNT', '0')}")
    attempt.mkdir(parents=True, exist_ok=False)
    record = {"schema_version": 1, "protocol": PROTOCOL, "stage": stage,
              "policies": list(policy_ids), "task_indices": list(STAGES[stage]),
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
              "status": "running", "runs": {},
              "started_utc": datetime.now(timezone.utc).isoformat()}
    write_json_atomic(attempt / "allocation.json", record)
    gpt = None
    try:
        config = config_for(BASE_ARM)
        gpt = gpt_factory(config["autonomous_support"]["model"], attempt / "workers")
        qwen = qwen_factory(config)
        memory = memory_factory(config, qwen)
        shared = (qwen, gpt, memory)
        for policy in policy_ids:
            directory = root / stage / policy
            directory.mkdir(parents=True)
            path = runner(directory, policy, list(STAGES[stage]), plan,
                          shared_models=shared)
            record["runs"][policy] = str(path)
            meta = read_json(path / "run.json")
            if meta.get("status") != "complete":
                raise RuntimeError(f"Matched {stage} run is infrastructure-invalid: {path}")
            write_json_atomic(attempt / "allocation.json", record)
        record["status"] = "complete"
    except BaseException as exc:
        record.update(status="failed", error=repr(exc))
        raise
    finally:
        if gpt is not None:
            gpt.close()
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json_atomic(attempt / "allocation.json", record)
    return record


def run_matched_batch(root: str | Path, batch: int, policy_ids: tuple[str, ...],
                      *, qwen_factory=QwenModel, gpt_factory=GptOssControllerWorker,
                      memory_factory=load_memory) -> dict:
    """Execute at most two three-task policy batches after the matched pilot gate."""
    if (type(batch) is not int or not 0 <= batch < 10 or not 1 <= len(policy_ids) <= 2 or
            len(set(policy_ids)) != len(policy_ids) or any(p not in POLICIES for p in policy_ids)):
        raise ValueError("Invalid matched batch selection")
    stage_report(root, "pilot")
    feasibility_gate(os.environ["POLICY_FEASIBILITY_RESULT"])
    plan = checked_plan(root)
    pending = {policy: status(root, policy, batch) for policy in policy_ids}
    if any(item["blocking_run_status"] or item["completed"] != 0 or
           item["missing_indices"] != list(range(batch * 3, batch * 3 + 3))
           for item in pending.values()):
        raise ValueError("Matched batch requires fresh complete three-task slots")
    root = Path(root)
    attempt = root / "allocations" / (f"batch_{batch:02d}_job_{os.environ.get('SLURM_JOB_ID', 'local')}_"
                                      f"restart_{os.environ.get('SLURM_RESTART_COUNT', '0')}")
    attempt.mkdir(parents=True, exist_ok=False)
    record = {"schema_version": 1, "protocol": PROTOCOL, "batch": batch,
              "policies": list(policy_ids), "task_indices": pending[policy_ids[0]]["missing_indices"],
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "status": "running", "runs": {},
              "started_utc": datetime.now(timezone.utc).isoformat()}
    write_json_atomic(attempt / "allocation.json", record)
    gpt = None
    try:
        config = config_for(BASE_ARM)
        gpt = gpt_factory(config["autonomous_support"]["model"], attempt / "workers")
        qwen = qwen_factory(config)
        memory = memory_factory(config, qwen)
        shared = (qwen, gpt, memory)
        for policy in policy_ids:
            def shared_runner(directory, selected_policy, indices, selected_plan):
                return _run_tasks(directory, selected_policy, indices, selected_plan,
                                  shared_models=shared)
            outcome = run_batch(root, policy, batch, runner=shared_runner)
            record["runs"][policy] = outcome["run_dir"]
            if outcome["completed"] != 3:
                raise RuntimeError(f"Incomplete matched batch: {policy} {batch}")
            write_json_atomic(attempt / "allocation.json", record)
        record["status"] = "complete"
    except BaseException as exc:
        record.update(status="failed", error=repr(exc))
        raise
    finally:
        if gpt is not None:
            gpt.close()
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json_atomic(attempt / "allocation.json", record)
    return record


def stage_report(root: str | Path, stage: str) -> dict:
    """CPU-only gate: one matched, infrastructure-valid run per policy."""
    if stage not in STAGES:
        raise ValueError("Unknown stage")
    plan = _plan(require_runtime=False)
    tasks = manifest()["tasks"]
    indices = list(STAGES[stage])
    results = {}
    for policy_id in POLICIES:
        paths = sorted((Path(root) / stage / policy_id).glob("textcraft_*"))
        if len(paths) != 1:
            raise ValueError(f"{stage} requires exactly one {policy_id} run")
        run_dir = paths[0]
        meta = read_json(run_dir / "run.json")
        if (meta.get("status") != "complete" or meta.get("protocol") != PROTOCOL or
                not _placement_valid(meta) or
                "L40S" not in meta.get("gptoss_metadata", {}).get("gpu_name", "") or
                meta.get("requested_attention_implementation") != REQUESTED_ATTENTION or
                meta.get("attention_implementation") != REQUESTED_ATTENTION or
                meta.get("gptoss_metadata", {}).get("attention_implementation") != REQUESTED_ATTENTION or
                meta.get("worker_tracebacks_enabled") is not True or
                meta.get("policy_id") != policy_id or
                meta.get("policy_sha256") != plan["policy_sha256"][policy_id] or
                meta.get("policy_configuration") != policy_configuration(policy_id) or
                meta.get("implementation_commit") != plan["implementation_commit"] or
                meta.get("base_config_sha256") != plan["base_config_sha256"] or
                meta.get("model_settings") != plan["model_settings"] or
                meta.get("limits") != plan["limits"] or
                meta.get("memory") != plan["memory"] or
                meta.get("manifest_sha256") != MANIFEST_SHA256 or
                meta.get("recipe_order_sha256") != RECIPE_SHA256 or
                meta.get("python_version") != plan["python_version"] or
                meta.get("package_versions") != plan["package_versions"] or
                meta.get("task_indices") != indices or
                meta.get("completed_task_count") != len(indices) or
                read_json(run_dir / "config.json") != config_for(BASE_ARM) or
                read_json(run_dir / "policy.json") != policy_configuration(policy_id) or
                read_json(run_dir / "manifest.json") != {"schema_version": 1,
                    "benchmark": "textcraft", "partition": "development",
                    "tasks": [tasks[index] for index in indices]}):
            raise ValueError(f"Incompatible {stage} run: {run_dir}")
        episodes = sorted((run_dir / "episodes").glob("*.json"))
        if len(episodes) != len(indices):
            raise ValueError(f"Incomplete {stage} episodes: {run_dir}")
        rows = []
        for index, path in zip(indices, episodes):
            episode = read_json(path)
            telemetry = episode_telemetry(episode.get("trace", []), policy_id)
            if (episode.get("task_id") != tasks[index]["task_id"] or
                    episode.get("initial_observation") != tasks[index]["initial_observation"] or
                    episode.get("arm") != BASE_ARM or
                    episode.get("support_policy_id") != policy_id or
                    episode.get("implementation_commit") != plan["implementation_commit"] or
                    episode.get("experiment_protocol") != PROTOCOL or
                    episode.get("decision_telemetry") != telemetry or
                    type(episode.get("success")) is not bool or
                    episode.get("status") in TERMINAL_WORKER_STATUSES or
                    episode.get("infrastructure_invalid") is not False or
                    any(item["infrastructure_failure_signal"] for item in telemetry)):
                raise ValueError(f"Invalid or infrastructure-contaminated {stage}: {path}")
            rows.append({"task_id": episode["task_id"], "success": episode["success"],
                         "status": episode["status"], "environment_actions": episode["environment_actions"],
                         "controller_fallbacks": episode.get("controller_fallbacks", 0)})
        results[policy_id] = {"run_dir": str(run_dir), "tasks": rows,
                              "successes": sum(row["success"] for row in rows)}
    return {"protocol": PROTOCOL, "stage": stage,
            "implementation_commit": plan["implementation_commit"],
            "task_indices": indices, "policy_runs": results,
            "ready": True}


def completed(root: str | Path, policy_id: str, plan: dict) -> dict[str, dict]:
    if policy_id not in POLICIES:
        raise ValueError("Unknown policy")
    selected_tasks = manifest()["tasks"]
    tasks = {task["task_id"]: (index, task)
             for index, task in enumerate(selected_tasks)}
    found = {}
    for run_dir in sorted((Path(root) / "runs" / policy_id).glob("batch_*/textcraft_*")):
        meta = read_json(run_dir / "run.json")
        indices = meta.get("task_indices")
        if (meta.get("protocol") != PROTOCOL or meta.get("policy_id") != policy_id or
                not _placement_valid(meta) or
                "L40S" not in meta.get("gptoss_metadata", {}).get("gpu_name", "") or
                meta.get("requested_attention_implementation") != REQUESTED_ATTENTION or
                meta.get("attention_implementation") != REQUESTED_ATTENTION or
                meta.get("gptoss_metadata", {}).get("attention_implementation") != REQUESTED_ATTENTION or
                meta.get("worker_tracebacks_enabled") is not True or
                meta.get("policy_sha256") != plan["policy_sha256"][policy_id] or
                meta.get("policy_configuration") != policy_configuration(policy_id) or
                meta.get("implementation_commit") != plan["implementation_commit"] or
                meta.get("base_config_sha256") != plan["base_config_sha256"] or
                meta.get("model_settings") != plan["model_settings"] or
                meta.get("limits") != plan["limits"] or
                meta.get("manifest_sha256") != MANIFEST_SHA256 or
                meta.get("recipe_order_sha256") != RECIPE_SHA256 or
                meta.get("memory") != plan["memory"] or
                meta.get("python_version") != plan["python_version"] or
                meta.get("package_versions") != plan["package_versions"] or
                read_json(run_dir / "config.json") != config_for(BASE_ARM) or
                read_json(run_dir / "policy.json") != policy_configuration(policy_id) or
                not isinstance(indices, list) or not indices or
                indices != sorted(set(indices)) or
                any(type(i) is not int or not 0 <= i < TASK_COUNT or
                    run_dir.parent.name != f"batch_{i // BATCH_SIZE:02d}" for i in indices) or
                read_json(run_dir / "manifest.json") != {"schema_version": 1,
                    "benchmark": "textcraft", "partition": "development",
                    "tasks": [selected_tasks[index] for index in indices]}):
            raise ValueError(f"Incompatible policy run: {run_dir}")
        paths = sorted((run_dir / "episodes").glob("*.json"))
        if (meta.get("status") == "complete" and
                (len(paths) != len(indices) or meta.get("completed_task_count") != len(indices))):
            raise ValueError(f"Complete policy run has missing episodes: {run_dir}")
        for path in paths:
            episode = read_json(path)
            task_id = episode.get("task_id")
            if task_id not in tasks or task_id in found:
                raise ValueError(f"Invalid or duplicate task: {path}")
            index, task = tasks[task_id]
            telemetry = episode_telemetry(episode.get("trace", []), policy_id)
            if (index not in indices or not path.name.startswith(f"{index:04d}_") or
                    episode.get("initial_observation") != task["initial_observation"] or
                    episode.get("arm") != BASE_ARM or
                    episode.get("support_policy_id") != policy_id or
                    episode.get("implementation_commit") != plan["implementation_commit"] or
                    episode.get("experiment_protocol") != PROTOCOL or
                    type(episode.get("success")) is not bool or
                    not isinstance(episode.get("trace"), list) or
                    episode.get("decision_telemetry") != telemetry or
                    episode.get("infrastructure_invalid") != (
                        episode.get("status") in TERMINAL_WORKER_STATUSES or
                        episode.get("status") == "environment_error" or
                        any(item["infrastructure_failure_signal"] for item in telemetry))):
                raise ValueError(f"Invalid policy episode: {path}")
            found[task_id] = episode
    return found


def _blocking_run_status(root: str | Path, policy_id: str, batch: int) -> str | None:
    directory = Path(root) / "runs" / policy_id / f"batch_{batch:02d}"
    return next((status for path in directory.glob("textcraft_*/run.json")
                 if (status := read_json(path).get("status")) in {
                     "stopped_worker_failure", "stopped_infrastructure_failure", "failed"}), None)


def status(root: str | Path, policy_id: str, batch: int) -> dict:
    if policy_id not in POLICIES or type(batch) is not int or not 0 <= batch < 10:
        raise ValueError("Invalid policy or batch")
    plan = checked_plan(root)
    done = completed(root, policy_id, plan)
    tasks = manifest()["tasks"]
    missing = [i for i in range(batch * BATCH_SIZE, (batch + 1) * BATCH_SIZE)
               if tasks[i]["task_id"] not in done]
    blocked = _blocking_run_status(root, policy_id, batch)
    return {"policy_id": policy_id, "batch": batch,
            "completed": BATCH_SIZE - len(missing), "missing_indices": missing,
            "blocking_run_status": blocked}


def run_batch(root: str | Path, policy_id: str, batch: int,
              max_tasks_per_attempt: int = BATCH_SIZE,
              *, runner=_run_tasks) -> dict:
    if (policy_id not in POLICIES or type(batch) is not int or not 0 <= batch < 10 or
            type(max_tasks_per_attempt) is not int or
            not 1 <= max_tasks_per_attempt <= BATCH_SIZE):
        raise ValueError("Invalid policy, batch, or attempt size")
    feasibility_gate(os.environ["POLICY_FEASIBILITY_RESULT"])
    plan = checked_plan(root)
    directory = Path(root) / "runs" / policy_id / f"batch_{batch:02d}"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".batch.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = status(root, policy_id, batch)
        if current["blocking_run_status"]:
            raise ValueError(f"Prior {current['blocking_run_status']} requires infrastructure review")
        indices = current["missing_indices"][:max_tasks_per_attempt]
        if not indices:
            return {**current, "run_dir": None}
        path = runner(directory, policy_id, indices, plan)
        return {**status(root, policy_id, batch), "run_dir": str(path)}


def aggregate(root: str | Path, policy_id: str) -> dict:
    plan = checked_plan(root)
    done = completed(root, policy_id, plan)
    tasks = manifest()["tasks"]
    rows = []
    for task in tasks:
        episode = done.get(task["task_id"])
        if episode is None:
            continue
        telemetry = episode["decision_telemetry"]
        progress = sum(item.get("assessment") == "progressed"
                       for event in telemetry for item in event["progress_judgments"])
        observed_progress = sum(
            str((event["environment_outcome"] or {}).get("observation", "")).startswith(
                ("Got ", "Crafted ")) for event in telemetry)
        intervention_events = sum(len(event["policy_interventions"]) for event in telemetry)
        intervention_decisions = sum(bool(event["policy_interventions"]) for event in telemetry)
        fallback_counts = {}
        for event in telemetry:
            reason = event["controller_fallback_reason"]
            if reason:
                fallback_counts[reason] = fallback_counts.get(reason, 0) + 1
        rows.append({"task_id": task["task_id"], "seed": task["seed"],
            "observation_sha256": task["observation_sha256"],
            "success": episode["success"], "status": episode["status"],
            "infrastructure_invalid": episode["infrastructure_invalid"],
            "recovery_success": bool(episode["success"] and episode["failed_actions"] > 0),
            "progressed_assessments": progress,
            "observed_progress_actions": observed_progress,
            "intervention_events": intervention_events,
            "intervention_decisions": intervention_decisions,
            "controller_fallbacks_by_reason": fallback_counts,
            **{key: episode.get(key, 0) for key in (
                "environment_actions", "failed_actions", "model_decisions",
                "strategy_changes", "replanning_events", "support_calls",
                "qwen_model_calls", "controller_model_calls", "support_model_calls",
                "embedding_model_calls", "input_tokens", "output_tokens",
                "inference_seconds", "elapsed_seconds")}})
    complete = len(rows) == TASK_COUNT
    invalid = [row["task_id"] for row in rows if row["infrastructure_invalid"]]
    valid = [row for row in rows if not row["infrastructure_invalid"]]
    numeric = ("environment_actions", "failed_actions", "model_decisions",
               "strategy_changes", "replanning_events", "support_calls",
               "qwen_model_calls", "controller_model_calls", "support_model_calls",
               "embedding_model_calls", "input_tokens", "output_tokens",
               "inference_seconds", "elapsed_seconds", "intervention_events",
               "intervention_decisions",
               "progressed_assessments", "observed_progress_actions")
    support_counts = {}
    policy_selected_calls = 0
    controller_selected_calls = 0
    for episode in done.values():
        for decision in episode["decision_telemetry"]:
            for support in decision["supports"]:
                if support["result"] is None:
                    continue
                operation = support["request"]["operation"]
                support_counts[operation] = support_counts.get(operation, 0) + 1
                if support["selected_by_policy"]:
                    policy_selected_calls += 1
                else:
                    controller_selected_calls += 1
    return {"protocol": PROTOCOL, "policy_id": policy_id,
        "plan_sha256": canonical_digest(plan),
        "implementation_commit": plan["implementation_commit"],
        "manifest_sha256": MANIFEST_SHA256, "complete": complete,
        "tasks_completed": len(rows), "infrastructure_invalid_task_ids": invalid,
        "primary_score_ready": complete and not invalid,
        "successes": sum(row["success"] for row in rows),
        "success_rate": (sum(row["success"] for row in rows) / TASK_COUNT
                         if complete and not invalid else None),
        "wilson_95": (wilson(sum(row["success"] for row in rows), TASK_COUNT)
                      if complete and not invalid else None),
        "recovery_successes": sum(row["recovery_success"] for row in valid),
        "intervention_frequency": (sum(row["intervention_decisions"] for row in rows) /
                                   sum(row["model_decisions"] for row in rows)
                                   if sum(row["model_decisions"] for row in rows) else None),
        "controller_fallbacks_by_reason": {
            reason: sum(row["controller_fallbacks_by_reason"].get(reason, 0) for row in rows)
            for reason in sorted({reason for row in rows
                                  for reason in row["controller_fallbacks_by_reason"]})},
        "support_calls_by_operation": dict(sorted(support_counts.items())),
        "support_calls_selected_by_policy": policy_selected_calls,
        "support_calls_selected_by_controller": controller_selected_calls,
        "totals": {key: round(sum(row[key] for row in rows), 3) for key in numeric},
        "missing_task_ids": [task["task_id"] for task in tasks if task["task_id"] not in done],
        "tasks": rows}


def compare(root: str | Path) -> dict:
    reports = {policy: aggregate(root, policy) for policy in POLICIES}
    if any(not item["primary_score_ready"] for item in reports.values()):
        raise ValueError("All policies need complete infrastructure-valid runs")
    if len({item["plan_sha256"] for item in reports.values()}) != 1:
        raise ValueError("Policy runs do not share one locked plan")
    rows = {policy: {row["task_id"]: row for row in report["tasks"]}
            for policy, report in reports.items()}
    task_ids = [task["task_id"] for task in manifest()["tasks"]]
    comparisons = {}
    for left, right in (("state_aware_v1", "original_v4"),
                        ("accuracy_first_v1", "original_v4"),
                        ("accuracy_first_v1", "state_aware_v1"),
                        ("accuracy_first_terminal_v1", "original_v4"),
                        ("accuracy_first_terminal_v1", "state_aware_v1"),
                        ("accuracy_first_terminal_v1", "accuracy_first_v1")):
        differences = [int(rows[left][task]["success"]) - int(rows[right][task]["success"])
                       for task in task_ids]
        comparisons[f"{left}_minus_{right}"] = {
            "paired_wins": differences.count(1), "paired_losses": differences.count(-1),
            "success_rate_difference": round(sum(differences) / TASK_COUNT, 4),
            "paired_bootstrap_95": paired_interval(differences)}
    development_gate_met = all(
        comparisons[key]["paired_wins"] - comparisons[key]["paired_losses"] >= 2
        for key in ("accuracy_first_terminal_v1_minus_original_v4",
                    "accuracy_first_terminal_v1_minus_state_aware_v1",
                    "accuracy_first_terminal_v1_minus_accuracy_first_v1"))
    return {"protocol": PROTOCOL, "plan_sha256": next(iter(reports.values()))["plan_sha256"],
        "primary_metric": "environment_verified_success_rate",
        "exploratory_development_gate_met": development_gate_met,
        "policy_reports": {policy: {key: value for key, value in report.items()
                                    if key != "tasks"} for policy, report in reports.items()},
        "paired": comparisons,
        "task_rows": [{"task_id": task, **{policy: rows[policy][task]["success"]
                                          for policy in POLICIES}} for task in task_ids]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--policy", choices=POLICIES, required=True)
    pre.add_argument("--strict-runtime", action="store_true")
    gate = sub.add_parser("feasibility-gate")
    gate.add_argument("--result", required=True)
    init = sub.add_parser("init")
    init.add_argument("--root", required=True)
    for stage in STAGES:
        command = sub.add_parser(stage)
        command.add_argument("--root", required=True)
        command.add_argument("--policy", choices=POLICIES, required=True)
    matched = sub.add_parser("matched-stage")
    matched.add_argument("--root", required=True)
    matched.add_argument("--stage", choices=tuple(STAGES), required=True)
    matched.add_argument("--policies", nargs="+", choices=POLICIES, required=True)
    matched_batch = sub.add_parser("matched-batch")
    matched_batch.add_argument("--root", required=True)
    matched_batch.add_argument("--batch", type=int, required=True)
    matched_batch.add_argument("--policies", nargs="+", choices=POLICIES, required=True)
    for name in ("status", "run-batch", "aggregate"):
        command = sub.add_parser(name)
        command.add_argument("--root", required=True)
        command.add_argument("--policy", choices=POLICIES, required=True)
        if name in {"status", "run-batch"}:
            command.add_argument("--batch", type=int, required=True)
        if name == "run-batch":
            command.add_argument("--max-tasks-per-attempt", type=int, default=BATCH_SIZE)
        if name == "aggregate":
            command.add_argument("--output", required=True)
    comparison = sub.add_parser("compare")
    comparison.add_argument("--root", required=True)
    comparison.add_argument("--output", required=True)
    stage_check = sub.add_parser("stage-report")
    stage_check.add_argument("--root", required=True)
    stage_check.add_argument("--stage", choices=tuple(STAGES), required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        result = preflight(args.policy)
        if args.strict_runtime and not runtime_preflight_passed(result):
            raise ValueError("CPU-only Locust runtime preflight failed")
    elif args.command == "feasibility-gate":
        result = feasibility_gate(args.result)
    elif args.command == "init":
        result = initialize(args.root)
    elif args.command in STAGES:
        result = {"run_dir": str(run_stage(args.root, args.policy, args.command))}
    elif args.command == "matched-stage":
        result = run_matched_stage(args.root, args.stage, tuple(args.policies))
    elif args.command == "matched-batch":
        result = run_matched_batch(args.root, args.batch, tuple(args.policies))
    elif args.command == "stage-report":
        result = stage_report(args.root, args.stage)
    elif args.command == "status":
        result = status(args.root, args.policy, args.batch)
    elif args.command == "run-batch":
        result = run_batch(args.root, args.policy, args.batch, args.max_tasks_per_attempt)
    elif args.command == "aggregate":
        report = aggregate(args.root, args.policy)
        if Path(args.output).exists():
            raise FileExistsError(args.output)
        write_json_atomic(args.output, report)
        result = {key: value for key, value in report.items() if key != "tasks"}
    else:
        report = compare(args.root)
        if Path(args.output).exists():
            raise FileExistsError(args.output)
        write_json_atomic(args.output, report)
        result = {key: value for key, value in report.items() if key != "task_rows"}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
