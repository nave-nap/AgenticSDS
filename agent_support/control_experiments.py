"""Versioned TextCraft GPT-OSS controls: standalone and review-only."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import platform
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from .autonomous_evaluation import require_clean_commit
from .benchmarks import make_episode
from .common import canonical_digest, read_json, write_json_atomic
from .controller_backend import GptOssControllerWorker
from .heldout50_evaluation import FROZEN_SHA256, RECIPE_SHA256, manifests
from .model import QwenModel
from .policy import parse_textcraft, textcraft_messages
from .runner import package_versions

REPO = Path(__file__).resolve().parent.parent
ARMS = ("gptoss_standalone_v1", "gptoss_review_only_v1")
CONFIG_DIR = REPO / "configs/control_experiments"
PROTOCOL = "textcraft_gptoss_controls_v1"
STANDALONE_SYSTEM = (
    "You solve the TextCraft task directly. Use only the displayed recipes, current observation, "
    "and action feedback. You may use native get, craft, and inventory commands. "
    "Return exactly one line in the final channel: Action: <one command>. "
    "Use exact item names and positive integer quantities; no prose or markdown."
)
REVIEW_SYSTEM = (
    "Review the task agent's proposed TextCraft action using only the visible task, "
    "current observation, and prior environment actions and feedback. "
    "Return exactly one JSON object in the final channel: "
    '{"decision":"accept"} or {"decision":"revise","action":"get 1 item"}. '
    "A revised action may be get, craft, or inventory. Use an exact native TextCraft command. "
    "Do not request tools, support, memory, or additional information."
)


def config_for(arm: str) -> dict:
    if arm not in ARMS:
        raise ValueError("Unknown control arm")
    config = read_json(CONFIG_DIR / f"{arm}.json")
    v3 = read_json(REPO / "configs/autonomous_evaluation/autonomous_v3.json")
    qwen = read_json(REPO / "configs/autonomous_evaluation/qwen_standalone.json")
    gpt = v3["autonomous_support"]["model"]
    expected = {key: qwen[key] for key in ("max_actions", "max_decisions", "history_window", "seed")}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Control task limits, history, or seed differ from Qwen/v3")
    if (config.get("schema_version") != 1 or config.get("benchmark") != "textcraft" or
            config.get("arm") != arm or config.get("gpt_oss") != gpt or
            set(config) != {"schema_version", "benchmark", "arm", "max_actions",
                            "max_decisions", "history_window", "seed", "gpt_oss", "qwen"}):
        raise ValueError("Control protocol or GPT-OSS revision differs from v3")
    if arm == ARMS[0] and config["qwen"] is not None:
        raise ValueError("Standalone must not load Qwen")
    if arm == ARMS[1] and config["qwen"] != {key: qwen[key] for key in
            ("model_id", "model_revision", "quantization", "max_new_tokens", "seed")}:
        raise ValueError("Review Qwen model differs from the standalone baseline")
    if gpt["model"].get("completion_parser") != "harmony_final_v3":
        raise ValueError("Control requires the pinned v3 Harmony parser")
    return config


def preflight(arm: str) -> dict:
    _, frozen = manifests()
    config = config_for(arm)
    worker = config["gpt_oss"]
    python = os.environ.get(worker["python_env"])
    cache = os.environ.get(worker["hf_home_env"])
    versions = package_versions()
    checks = {"gptoss_python_exists": bool(python and Path(python).is_file()),
              "gptoss_hf_home_exists": bool(cache and Path(cache).is_dir()),
              "textcraft_0_0_3": versions.get("textcraft") == "0.0.3"}
    if arm == ARMS[1]:
        checks["qwen_dependencies_present"] = all(versions.get(name) for name in
                                                   ("torch", "transformers", "bitsandbytes"))
    return {"arm": arm, "protocol": PROTOCOL, "task_count": len(frozen["tasks"]),
            "manifest_sha256": FROZEN_SHA256, "recipe_order_sha256": RECIPE_SHA256,
            "config_sha256": canonical_digest(config), "runtime_checks": checks,
            "package_versions": versions}


def _valid_review(raw: str, proposal: str) -> tuple[str, str, str | None]:
    try:
        def unique_pairs(pairs):
            value = {}
            for key, item in pairs:
                if key in value:
                    raise ValueError("duplicate_key")
                value[key] = item
            return value
        value = json.loads(raw, object_pairs_hook=unique_pairs)
    except (TypeError, ValueError):
        return proposal, "fallback", "invalid_json"
    if value == {"decision": "accept"}:
        return proposal, "accepted", None
    if (isinstance(value, dict) and set(value) == {"decision", "action"} and
            value["decision"] == "revise" and isinstance(value["action"], str) and
            parse_textcraft(value["action"]) == value["action"]):
        return value["action"], "revised" if value["action"] != proposal else "accepted", None
    return proposal, "fallback", "invalid_review_decision"


def _strict_gpt_action(raw: str) -> str | None:
    if not isinstance(raw, str) or not re.fullmatch(r"Action: [^\n\r]+", raw.strip()):
        return None
    return parse_textcraft(raw) if parse_textcraft(raw) == raw.strip()[8:] else None


def _gpt_messages(initial: str, observation: str, history: list[dict], window: int,
                  proposal: str | None = None, *, decision: int | None = None,
                  remaining_actions: int | None = None,
                  remaining_decisions: int | None = None) -> list[dict]:
    recent = history[-window:]
    state = {"task": initial, "current_observation": observation,
             "history": [{"action": item["action"], "feedback": item["observation"]}
                         for item in recent]}
    if proposal is not None:
        state["proposed_action"] = proposal
    if decision is not None:
        state.update(decision=decision, remaining_actions=remaining_actions,
                     remaining_decisions=remaining_decisions)
    return [{"role": "system", "content": REVIEW_SYSTEM if proposal is not None else STANDALONE_SYSTEM},
            {"role": "user", "content": json.dumps(state, ensure_ascii=False, separators=(",", ":"))}]


def run_episode(task: dict, config: dict, gpt, qwen=None, episode_factory=make_episode) -> dict:
    arm = config["arm"]
    episode = episode_factory("textcraft", task, config)
    started = time.perf_counter()
    history: list[dict] = []
    trace: list[dict] = []
    totals = {key: 0 for key in ("environment_actions", "failed_actions", "parser_errors",
              "gptoss_model_calls", "gptoss_input_tokens", "gptoss_output_tokens",
              "gptoss_inference_seconds", "qwen_model_calls", "qwen_input_tokens",
              "qwen_output_tokens", "qwen_inference_seconds", "review_parse_errors",
              "gptoss_generation_failures", "gptoss_backend_errors", "qwen_generation_failures",
              "action_revisions", "actions_accepted", "review_fallbacks", "environment_seconds")}
    status, success = "max_decisions", False
    try:
        episode.reset()
        for decision in range(1, config["max_decisions"] + 1):
            if totals["environment_actions"] >= config["max_actions"]:
                status = "max_actions"
                break
            record = {"decision": decision, "input_observation": episode.observation}
            proposal = None
            if arm == ARMS[1]:
                messages = textcraft_messages(episode.initial_observation, history, config["history_window"])
                totals["qwen_model_calls"] += 1
                try:
                    generated = qwen.generate(messages)
                except Exception as exc:
                    generated = {"text": "", "input_tokens": 0, "output_tokens": 0,
                                 "inference_seconds": 0.0, "error": repr(exc),
                                 "usage_unavailable": True}
                    totals["qwen_generation_failures"] += 1
                for key in ("input_tokens", "output_tokens", "inference_seconds"):
                    totals["qwen_" + key] += generated[key]
                proposal = parse_textcraft(generated["text"])
                record.update(qwen_messages=messages, qwen_generation=generated,
                              proposed_action=proposal)
                if proposal is None:
                    # Same invalid-proposal behavior as autonomous v3: no controller call.
                    action = None
                    record["review_decision"] = "skipped_invalid_qwen_proposal"
            if arm == ARMS[0] or proposal is not None:
                messages = _gpt_messages(episode.initial_observation, episode.observation,
                                         history, config["history_window"], proposal,
                                         decision=decision,
                                         remaining_actions=config["max_actions"] - totals["environment_actions"],
                                         remaining_decisions=config["max_decisions"] - decision + 1)
                record["gptoss_messages"] = messages
                totals["gptoss_model_calls"] += 1
                try:
                    generated = gpt.generate(messages, max_new_tokens=config["gpt_oss"]["model"]["max_new_tokens"])
                except Exception as exc:
                    generated = {"text": "", "raw_completion": "", "generation_status": "backend_error",
                                 "error": repr(exc), "input_tokens": 0, "output_tokens": 0,
                                 "inference_seconds": 0.0, "usage_unavailable": True}
                    totals["gptoss_backend_errors"] += 1
                record["gptoss_generation"] = generated
                for key in ("input_tokens", "output_tokens", "inference_seconds"):
                    totals["gptoss_" + key] += generated[key]
                if generated.get("generation_status") != "complete":
                    totals["gptoss_generation_failures"] += 1
                if arm == ARMS[0]:
                    action = (_strict_gpt_action(generated["text"])
                              if generated.get("generation_status") == "complete" else None)
                elif generated.get("generation_status") == "complete":
                    action, decision_label, error = _valid_review(generated["text"], proposal)
                    record["review_decision"] = decision_label
                    if error:
                        totals["review_parse_errors"] += 1
                        record["review_error"] = error
                        totals["review_fallbacks"] += 1
                    totals["action_revisions"] += int(decision_label == "revised")
                    totals["actions_accepted"] += int(decision_label == "accepted")
                else:
                    action = proposal
                    record["review_decision"] = "fallback"
                    record["review_error"] = generated.get("generation_status")
                    totals["review_parse_errors"] += 1
                    totals["review_fallbacks"] += 1
            record["action"] = action
            if action is None:
                totals["parser_errors"] += 1
                record.update(error="parse_error", output_observation="Invalid action format. Reply with exactly one valid command.")
                trace.append(record)
                history.append({"action": "<PARSE_ERROR>", "observation": record["output_observation"]})
                continue
            try:
                step_started = time.perf_counter()
                step = episode.step(action)
            except Exception as exc:
                record.update(error="environment_error", details=repr(exc))
                trace.append(record)
                status = "environment_error"
                break
            elapsed = round(time.perf_counter() - step_started, 3)
            totals["environment_seconds"] += elapsed
            totals["environment_actions"] += 1
            totals["failed_actions"] += int(step.get("failed_action", False))
            record.update(environment_seconds=elapsed,
                          environment_action_number=totals["environment_actions"],
                          output_observation=step["observation"], outcome=step)
            trace.append(record)
            history.append({"action": action, "observation": step["observation"]})
            if step["done"]:
                success = bool(step["success"])
                status = "success" if success else "task_failed"
                break
        else:
            if totals["environment_actions"] >= config["max_actions"]:
                status = "max_actions"
    finally:
        episode.close()
    return {"task_id": task["task_id"], "benchmark": "textcraft",
            "initial_observation": episode.initial_observation, "task_prompt": episode.task_prompt,
            "success": success, "status": status, "model_decisions": len(trace),
            "model_calls": totals["qwen_model_calls"] + totals["gptoss_model_calls"],
            "input_tokens": totals["qwen_input_tokens"] + totals["gptoss_input_tokens"],
            "output_tokens": totals["qwen_output_tokens"] + totals["gptoss_output_tokens"],
            "inference_seconds": round(totals["qwen_inference_seconds"] + totals["gptoss_inference_seconds"], 3),
            "support_calls": 0, "support_model_calls": 0, "external_support_operations": 0,
            "elapsed_seconds": round(time.perf_counter() - started, 3), "trace": trace,
            **{key: round(value, 3) if key.endswith("seconds") else value
               for key, value in totals.items()}}


def _runtime_lock() -> dict:
    return {"python_version": platform.python_version(), "package_versions": package_versions()}


def _checked_gpu_metadata(gpt) -> dict:
    metadata = gpt.metadata()
    if (metadata.get("gpu_name") is not None and
            ("L40S" not in metadata["gpu_name"] or
             metadata.get("gpu_total_memory_bytes", 0) < 40 * 1024 ** 3)):
        raise ValueError("Control evaluation requires an L40S with at least 40 GiB")
    return metadata


def initialize(root: str | Path, batch_size: int = 5) -> dict:
    if batch_size != 5:
        raise ValueError("Controls use five-task batches")
    commit = require_clean_commit()
    checks = {arm: preflight(arm) for arm in ARMS}
    if any(not all(item["runtime_checks"].values()) for item in checks.values()):
        raise ValueError("Locust Python, GPT-OSS cache, or TextCraft 0.0.3 preflight failed")
    plan = {"schema_version": 1, "protocol": PROTOCOL, "git_commit": commit,
            "manifest_sha256": FROZEN_SHA256, "recipe_order_sha256": RECIPE_SHA256,
            "batch_size": 5, "config_sha256": {arm: checks[arm]["config_sha256"] for arm in ARMS},
            **_runtime_lock()}
    path = Path(root) / "plan.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return plan


def checked_plan(root: str | Path) -> dict:
    plan = read_json(Path(root) / "plan.json")
    manifests()
    if (plan.get("schema_version") != 1 or plan.get("protocol") != PROTOCOL or
            plan.get("manifest_sha256") != FROZEN_SHA256 or
            plan.get("recipe_order_sha256") != RECIPE_SHA256 or
            plan.get("batch_size") != 5 or plan.get("git_commit") != require_clean_commit() or
            any(plan["config_sha256"].get(arm) != canonical_digest(config_for(arm)) for arm in ARMS) or
            any(plan.get(key) != value for key, value in _runtime_lock().items())):
        raise ValueError("Control plan differs from code, config, manifest, or runtime")
    return plan


def completed(root: str | Path, arm: str, plan: dict) -> dict:
    _, frozen = manifests()
    tasks = {task["task_id"]: (index, task) for index, task in enumerate(frozen["tasks"])}
    found = {}
    for run_dir in sorted((Path(root) / "runs" / arm).glob("batch_*/textcraft_*")):
        if not run_dir.is_dir():
            continue
        if not (run_dir / "run.json").exists():
            if list((run_dir / "episodes").glob("*.json")):
                raise ValueError(f"Control episodes lack run metadata: {run_dir}")
            continue
        meta = read_json(run_dir / "run.json")
        if (meta.get("arm") != arm or meta.get("git_commit") != plan["git_commit"] or
                meta.get("manifest_sha256") != FROZEN_SHA256 or
                meta.get("config_sha256") != plan["config_sha256"][arm] or
                meta.get("recipe_order_sha256") != RECIPE_SHA256 or
                meta.get("python_version") != plan["python_version"] or
                meta.get("package_versions") != plan["package_versions"] or
                read_json(run_dir / "config.json") != config_for(arm)):
            raise ValueError(f"Incompatible control run: {run_dir}")
        indices = meta.get("task_indices")
        if (not isinstance(indices, list) or not indices or
                indices != sorted(set(indices)) or
                any(type(index) is not int or not 0 <= index < 50 or
                    index // 5 != int(run_dir.parent.name.removeprefix("batch_"))
                    for index in indices)):
            raise ValueError(f"Invalid control batch indices: {run_dir}")
        paths = sorted((run_dir / "episodes").glob("*.json"))
        if (meta.get("status") == "complete" and
                (len(paths) != len(indices) or meta.get("completed_task_count") != len(indices))):
            raise ValueError(f"Complete control run has missing snapshots: {run_dir}")
        for path in paths:
            episode = read_json(path)
            task_id = episode.get("task_id")
            if task_id not in tasks or task_id in found:
                raise ValueError(f"Unknown or duplicate task: {path}")
            index, task = tasks[task_id]
            if (index not in indices or
                    run_dir.parent.name != f"batch_{index // 5:02d}" or
                    not path.name.startswith(f"{index:04d}_") or
                    episode.get("initial_observation") != task["initial_observation"] or
                    type(episode.get("success")) is not bool or
                    episode.get("external_support_operations") != 0 or
                    not isinstance(episode.get("trace"), list)):
                raise ValueError(f"Invalid control episode: {path}")
            found[task_id] = episode
    return found


def status(root: str | Path, arm: str, batch: int) -> dict:
    if arm not in ARMS or type(batch) is not int or not 0 <= batch < 10:
        raise ValueError("Invalid arm or batch")
    plan = checked_plan(root)
    _, frozen = manifests()
    done = completed(root, arm, plan)
    missing = [i for i in range(batch * 5, batch * 5 + 5)
               if frozen["tasks"][i]["task_id"] not in done]
    return {"arm": arm, "batch": batch, "completed": 5 - len(missing),
            "missing_indices": missing}


def run_batch(root: str | Path, arm: str, batch: int, max_tasks_per_attempt: int = 5,
              model_factory=None, episode_factory=make_episode) -> dict:
    if not 1 <= max_tasks_per_attempt <= 5:
        raise ValueError("max_tasks_per_attempt must be 1 through 5")
    plan = checked_plan(root)
    directory = Path(root) / "runs" / arm / f"batch_{batch:02d}"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".batch.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        indices = status(root, arm, batch)["missing_indices"][:max_tasks_per_attempt]
        if not indices:
            return {**status(root, arm, batch), "run_dir": None}
        config = config_for(arm)
        _, frozen = manifests()
        run_dir = directory / ("textcraft_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + os.urandom(4).hex())
        (run_dir / "episodes").mkdir(parents=True)
        write_json_atomic(run_dir / "config.json", config)
        meta = {"schema_version": 1, "arm": arm, "protocol": PROTOCOL, "status": "running",
                "git_commit": plan["git_commit"], "manifest_sha256": FROZEN_SHA256,
                "recipe_order_sha256": RECIPE_SHA256, "config_sha256": plan["config_sha256"][arm],
                **_runtime_lock(), "task_indices": indices, "completed_task_count": 0,
                "started_utc": datetime.now(timezone.utc).isoformat()}
        write_json_atomic(run_dir / "run.json", meta)
        factory = model_factory or GptOssControllerWorker
        gpt = None
        try:
            gpt = factory(config["gpt_oss"], run_dir / "workers")
            meta["gptoss_metadata"] = _checked_gpu_metadata(gpt)
            meta["slurm_job_id"] = os.environ.get("SLURM_JOB_ID")
            write_json_atomic(run_dir / "run.json", meta)
            qwen = QwenModel({**config["qwen"], "benchmark": "textcraft"}) if config["qwen"] else None
            for index in indices:
                task = frozen["tasks"][index]
                gpt.begin_episode(task["task_id"], config["seed"])
                result = run_episode(task, config, gpt, qwen, episode_factory)
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
        return {**status(root, arm, batch), "run_dir": str(run_dir)}


def aggregate(root: str | Path, arm: str) -> dict:
    plan = checked_plan(root)
    _, frozen = manifests()
    done = completed(root, arm, plan)
    rows = [{"task_id": task["task_id"], "seed": task["seed"],
             "observation_sha256": task["observation_sha256"],
             **{key: value for key, value in done[task["task_id"]].items()
                if key not in {"trace", "initial_observation", "task_prompt"}}}
            for task in frozen["tasks"] if task["task_id"] in done]
    keys = ("environment_actions", "failed_actions", "model_calls", "input_tokens",
            "output_tokens", "inference_seconds", "qwen_model_calls", "gptoss_model_calls",
            "qwen_input_tokens", "qwen_output_tokens", "gptoss_input_tokens",
            "gptoss_output_tokens", "qwen_inference_seconds", "gptoss_inference_seconds",
            "parser_errors", "review_parse_errors", "gptoss_generation_failures",
            "gptoss_backend_errors", "qwen_generation_failures", "action_revisions",
            "actions_accepted", "review_fallbacks",
            "external_support_operations")
    return {"arm": arm, "protocol": PROTOCOL, "manifest_sha256": FROZEN_SHA256,
            "plan_sha256": canonical_digest(plan), "complete": len(rows) == 50,
            "tasks_completed": len(rows), "successes": sum(row["success"] for row in rows),
            "success_rate_observed": sum(row["success"] for row in rows) / len(rows) if rows else None,
            "totals": {key: round(sum(row.get(key, 0) for row in rows), 3) for key in keys},
            "missing_task_ids": [task["task_id"] for task in frozen["tasks"] if task["task_id"] not in done],
            "tasks": rows}


def smoke(root: str | Path, arm: str, task_index: int = 0,
          model_factory=None, episode_factory=make_episode, qwen_factory=None) -> Path:
    if arm not in ARMS or type(task_index) is not int or not 0 <= task_index < 20:
        raise ValueError("Smoke requires a development task index from 0 through 19")
    from .autonomous_evaluation import locked_manifest
    config = config_for(arm)
    task = locked_manifest()["tasks"][task_index]
    run_dir = Path(root) / "smoke" / arm / ("textcraft_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + os.urandom(4).hex())
    run_dir.mkdir(parents=True)
    gpt = (model_factory or GptOssControllerWorker)(config["gpt_oss"], run_dir / "workers")
    try:
        metadata = _checked_gpu_metadata(gpt)
        qwen = (qwen_factory or QwenModel)({**config["qwen"], "benchmark": "textcraft"}) if config["qwen"] else None
        gpt.begin_episode(task["task_id"], config["seed"])
        result = run_episode(task, config, gpt, qwen, episode_factory)
        write_json_atomic(run_dir / "config.json", config)
        write_json_atomic(run_dir / "episode.json", result)
        write_json_atomic(run_dir / "run.json", {"arm": arm, "protocol": PROTOCOL,
                          "partition": "development", "task_index": task_index,
                          "gptoss_metadata": metadata, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                          "result": result["status"]})
    finally:
        gpt.close()
    return run_dir


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
    p.add_argument("--task-index", type=int, default=0)
    for name in ("status", "run-batch", "aggregate"):
        p = sub.add_parser(name)
        p.add_argument("--root", required=True)
        p.add_argument("--arm", choices=ARMS, required=True)
        if name in {"status", "run-batch"}:
            p.add_argument("--batch", type=int, required=True)
        if name == "run-batch":
            p.add_argument("--max-tasks-per-attempt", type=int, default=5)
        if name == "aggregate":
            p.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        result = preflight(args.arm)
    elif args.command == "init":
        result = initialize(args.root)
    elif args.command == "smoke":
        result = {"run_dir": str(smoke(args.root, args.arm, args.task_index))}
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
