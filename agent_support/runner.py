"""Run baseline episodes with one model, manifest, and durable per-task logging."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import platform
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .benchmarks import benchmark_driver, make_episode
from .common import TRAJECTORY_CONDITIONS, canonical_digest, read_json, validate_config, validate_manifest, write_json_atomic
from .model import make_model
from .memory import ActionMemory, RETRIEVAL_POLICY
from .controllers import run_controller
from .support_policy import (ACTION_TRAJECTORY, LEGACY_ACTION_MEMORY, RECONSIDERATION, TASK_TRAJECTORY,
                             SupportPolicy, TurnContext, make_support_policy, selection_decision)
from .supports import build_builtin_resources, load_builtin_artifacts, make_supports
from .trajectory_memory import POLICY as TRAJECTORY_POLICY, TrajectoryMemory


def package_versions() -> dict:
    names = ("torch", "transformers", "sentence-transformers", "accelerate", "bitsandbytes", "alfworld", "textworld", "textcraft")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def git_commit() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def run_episode(benchmark: str, task: dict, config: dict, model, episode_factory=make_episode,
                support: ActionMemory | None = None,
                trajectory_support: TrajectoryMemory | None = None,
                support_policy: SupportPolicy | None = None,
                support_adapters: dict | None = None) -> dict:
    controller_id = config.get("controller", "adapt_dynamic_replanning_v1" if
                               config.get("condition") == "adapt_dynamic_replanning_v1" else None)
    if controller_id:
        return run_controller(controller_id, task, config, model, episode_factory)
    episode = episode_factory(benchmark, task, config)
    driver = benchmark_driver(benchmark)
    started = time.perf_counter()
    history: list[dict] = []
    failure_history: list[bool] = []
    trace: list[dict] = []
    status = "max_decisions"
    actions_taken = 0
    parser_errors = 0
    failed_actions = 0
    input_tokens = output_tokens = 0
    inference_seconds = 0.0
    support_calls = support_model_calls = support_input_tokens = support_output_tokens = support_parse_errors = 0
    model_calls = 0
    retrieval_seconds = 0.0
    success = False
    condition = config.get("condition", "m2_always" if support is not None else "baseline")
    policy = support_policy or make_support_policy(config, task["task_id"])
    adapters = support_adapters or make_supports(config, model, {"legacy": support, "trajectory": trajectory_support})

    def budget_block(choice: str, require_model: bool = False) -> str | None:
        adapter = adapters[choice]
        expected = adapter.expected_calls() if hasattr(adapter, "expected_calls") else 1
        if "max_support_calls" in config and support_calls + expected > config["max_support_calls"]:
            return "support_call_budget"
        if require_model and "max_model_calls" in config and model_calls >= config["max_model_calls"]:
            return "model_call_budget"
        return None

    def context(phase: str, decision: int, available: list[str] | None,
                proposed_action: str | None = None, proposal_output: str | None = None) -> TurnContext:
        return TurnContext(
            phase=phase, benchmark=benchmark, condition=condition, decision=decision,
            task_prompt=episode.task_prompt, observation=episode.observation,
            available_actions=tuple(available or ()),
            history=tuple((item["action"], item["observation"]) for item in history),
            actions_taken=actions_taken, remaining_actions=config["max_actions"] - actions_taken,
            remaining_decisions=config["max_decisions"] - decision + 1,
            support_calls_so_far=support_calls, proposed_action=proposed_action,
            proposal_output=proposal_output, available_supports=tuple(adapters),
            remaining_model_calls=(config["max_model_calls"] - model_calls
                                   if "max_model_calls" in config else None),
            remaining_support_calls=(config["max_support_calls"] - support_calls
                                     if "max_support_calls" in config else None),
            remaining_support_tokens=(config["max_support_tokens"] - support_input_tokens - support_output_tokens
                                      if "max_support_tokens" in config else None),
            recent_failures=tuple(failure_history))

    try:
        episode.reset()
        for decision in range(1, config["max_decisions"] + 1):
            if actions_taken >= config["max_actions"]:
                status = "max_actions"
                break
            if "max_model_calls" in config and model_calls >= config["max_model_calls"]:
                status = "max_model_calls"
                break
            if reason := driver.stop_reason(episode, history):
                status = reason
                break
            available, messages = driver.prompt(episode, history, config["history_window"])
            before_decision = selection_decision(policy, context("before_proposal", decision, available))
            before = before_decision.support_ids
            blocked_supports = []
            before_events = []
            for choice in before:
                if choice not in adapters or adapters[choice].phase != "before_proposal":
                    raise ValueError(f"Policy selected unavailable or misphased support: {choice}")
                if reason := budget_block(choice):
                    blocked_supports.append({"support_id": choice, "reason": reason})
                    continue
                outcome = adapters[choice].invoke(context("before_proposal", decision, available), messages, None)
                messages = outcome.messages
                support_calls += outcome.calls
                retrieval_seconds += outcome.retrieval_seconds
                if outcome.event is not None:
                    before_events.append(outcome.event)
            generation = model.generate(messages)
            model_calls += 1
            input_tokens += generation["input_tokens"]
            output_tokens += generation["output_tokens"]
            inference_seconds += generation["inference_seconds"]
            raw = generation["text"]
            action = driver.parse(raw, available)
            after_decision = selection_decision(policy, context("after_proposal", decision, available, action, raw))
            after = after_decision.support_ids
            record = {
                "decision": decision,
                "input_observation": episode.observation,
                "available_actions": available,
                "raw_model_output": raw,
                "action": action,
                "input_tokens": generation["input_tokens"],
                "output_tokens": generation["output_tokens"],
                "inference_seconds": generation["inference_seconds"],
            }
            if config.get("schema_version") == 2 or support is not None or trajectory_support is not None or before or after:
                record["support_policy"] = {"policy_id": policy.policy_id,
                                            "before_proposal": list(before),
                                            "after_proposal": list(after)}
                if config.get("schema_version") == 2:
                    record["support_policy"].update({"before_decision": before_decision.as_dict(),
                                                     "after_decision": after_decision.as_dict(),
                                                     "blocked": blocked_supports})

            def add_support_event(event: dict) -> None:
                record.setdefault("support_events", []).append(event)
                record["support"] = event  # Backward-compatible single-event view.

            for event in before_events:
                add_support_event(event)
            for choice in after:
                if action is None:
                    break
                proposed_action = action
                if choice not in adapters or adapters[choice].phase != "after_proposal":
                    raise ValueError(f"Policy selected unavailable or misphased support: {choice}")
                if reason := budget_block(choice, require_model=True):
                    blocked_supports.append({"support_id": choice, "reason": reason})
                    continue
                outcome = adapters[choice].invoke(context("after_proposal", decision, available, action, raw), messages, action)
                event = outcome.event
                revision_messages = outcome.revision_messages
                support_calls += outcome.calls
                retrieval_seconds += outcome.retrieval_seconds
                if outcome.calls and hasattr(policy, "on_executed"):
                    policy.on_executed(decision)
                record.setdefault("proposed_action", proposed_action)
                record.setdefault("proposal_raw_model_output", raw)
                if event is not None:
                    add_support_event(event)
                if revision_messages is None:
                    if event is not None and outcome.fallback_reason:
                        event["fallback_reason"] = outcome.fallback_reason
                    continue
                if "max_support_tokens" in config:
                    counter = getattr(model, "count_messages_tokens", None)
                    if counter is None:
                        raise ValueError("A support-token budget requires model.count_messages_tokens")
                    reserve = counter(revision_messages) + config["max_new_tokens"]
                    if support_input_tokens + support_output_tokens + reserve > config["max_support_tokens"]:
                        event["fallback_reason"] = "support_token_budget"
                        blocked_supports.append({"support_id": choice, "reason": "support_token_budget",
                                                 "required_reservation": reserve})
                        continue
                revision = model.generate(revision_messages)
                model_calls += 1
                support_model_calls += 1
                support_input_tokens += revision["input_tokens"]
                support_output_tokens += revision["output_tokens"]
                input_tokens += revision["input_tokens"]
                output_tokens += revision["output_tokens"]
                inference_seconds += revision["inference_seconds"]
                record["input_tokens"] += revision["input_tokens"]
                record["output_tokens"] += revision["output_tokens"]
                record["inference_seconds"] = round(record["inference_seconds"] + revision["inference_seconds"], 3)
                event.update({"raw_model_output": revision["text"],
                              "input_tokens": revision["input_tokens"],
                              "output_tokens": revision["output_tokens"],
                              "inference_seconds": revision["inference_seconds"]})
                revised = driver.parse(revision["text"], available)
                if revised is not None:
                    action = revised
                    record["action"] = action
                    record["raw_model_output"] = revision["text"]
                else:
                    event["parse_error"] = True
                    support_parse_errors += 1
                event["action_changed"] = action != proposed_action
            if action is None:
                parser_errors += 1
                record["error"] = "parse_error"
                record["output_observation"] = "Invalid action format. Reply with exactly one valid command."
                trace.append(record)
                if driver.parse_error_ends_episode():
                    status = "parse_error"
                    break
                history.append({"action": "<PARSE_ERROR>", "observation": record["output_observation"]})
                continue
            try:
                step = episode.step(action)
            except Exception as exc:
                record["error"] = "environment_error"
                record["details"] = repr(exc)
                trace.append(record)
                status = "environment_error"
                break
            actions_taken += 1
            failed_actions += int(step.get("failed_action", False))
            failure_history.append(bool(step.get("failed_action", False)))
            record.update({
                "environment_action_number": actions_taken,
                "output_observation": step["observation"],
                "outcome": step,
            })
            trace.append(record)
            history.append({"action": action, "observation": step["observation"]})
            if step["done"]:
                success = bool(step["success"])
                status = "success" if success else "task_failed"
                break
            if reason := driver.stop_reason(episode, history):
                status = reason
                break
        else:
            if actions_taken >= config["max_actions"]:
                status = "max_actions"
    finally:
        episode.close()
    result = {
        "task_id": task["task_id"],
        "benchmark": benchmark,
        "initial_observation": episode.initial_observation,
        "task_prompt": episode.task_prompt,
        "success": success,
        "status": status,
        "environment_actions": actions_taken,
        "model_decisions": len(trace),
        "parser_errors": parser_errors,
        "failed_actions": failed_actions,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "inference_seconds": round(inference_seconds, 3),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "trace": trace,
    }
    if config.get("schema_version") == 2 or condition != "baseline" or support is not None or trajectory_support is not None or support_calls:
        result.update({"model_calls": model_calls,
                       "support_calls": support_calls, "support_model_calls": support_model_calls,
                       "support_input_tokens": support_input_tokens, "support_output_tokens": support_output_tokens,
                       "support_parse_errors": support_parse_errors,
                       "retrieval_seconds": round(retrieval_seconds, 3),
                       "actions_changed_by_support": sum(any(event.get("action_changed") for event in
                                                            item.get("support_events", [item["support"]] if "support" in item else []))
                                                         for item in trace)})
    return result


def summarize(episodes: list[dict]) -> dict:
    count = len(episodes)
    summary = {
        "tasks": count,
        "successes": sum(int(item["success"]) for item in episodes),
        "success_rate": sum(int(item["success"]) for item in episodes) / count if count else None,
        "environment_actions": sum(item["environment_actions"] for item in episodes),
        "model_decisions": sum(item["model_decisions"] for item in episodes),
        "parser_errors": sum(item["parser_errors"] for item in episodes),
        "failed_actions": sum(item["failed_actions"] for item in episodes),
        "input_tokens": sum(item["input_tokens"] for item in episodes),
        "output_tokens": sum(item["output_tokens"] for item in episodes),
        "inference_seconds": round(sum(item["inference_seconds"] for item in episodes), 3),
    }
    if episodes and "support_calls" in episodes[0]:
        for key in ("model_calls", "support_calls", "support_model_calls", "support_input_tokens",
                    "support_output_tokens", "support_parse_errors", "retrieval_seconds", "actions_changed_by_support"):
            summary[key] = round(sum(item[key] for item in episodes), 3) if key == "retrieval_seconds" else sum(item[key] for item in episodes)
    if episodes and "planner_calls" in episodes[0]:
        for key in ("planner_calls", "executor_model_calls", "decompositions",
                    "plan_parse_errors", "false_completion_claims"):
            summary[key] = sum(item[key] for item in episodes)
    return summary


def run(config: dict, manifest: dict, output_root: str | Path, limit: int | None = None, model=None, episode_factory=make_episode) -> Path:
    validate_manifest(manifest)
    validate_config(config, manifest["benchmark"])
    if config.get("schema_version") == 2 and read_json(config["manifest"]) != manifest:
        raise ValueError("Configured manifest differs from the supplied task manifest")
    selected_policy = make_support_policy(config)
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    selected = manifest["tasks"][:limit] if limit is not None else manifest["tasks"]
    legacy_memory = "memory_corpus" in config
    trajectory_memory = "trajectory_corpus" in config
    artifacts = load_builtin_artifacts(config, manifest, selected)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    directory = Path(output_root) / f"{manifest['benchmark']}_{run_id}"
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "episodes").mkdir()
    selected_manifest = dict(manifest)
    selected_manifest["tasks"] = selected
    write_json_atomic(directory / "config.json", config)
    write_json_atomic(directory / "manifest.json", selected_manifest)
    alfworld_config_sha256 = None
    textcraft_recipe_order_sha256 = None
    if manifest["benchmark"] == "alfworld":
        source = Path(config["alfworld_config"])
        config_bytes = source.read_bytes()
        alfworld_config_sha256 = hashlib.sha256(config_bytes).hexdigest()
        (directory / "alfworld_config.yaml").write_bytes(config_bytes)
    else:
        source = Path(__file__).with_name("textcraft_recipe_order.json")
        order_bytes = source.read_bytes()
        textcraft_recipe_order_sha256 = hashlib.sha256(order_bytes).hexdigest()
        (directory / "textcraft_recipe_order.json").write_bytes(order_bytes)
    metadata = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "running",
        "benchmark": manifest["benchmark"],
        "condition": config["condition"],
        "partition": manifest.get("partition"),
        "manifest_sha256": canonical_digest(manifest),
        "selected_manifest_sha256": canonical_digest(selected_manifest),
        "selected_task_count": len(selected),
        "total_manifest_task_count": len(manifest["tasks"]),
        "limit": limit,
        "config_sha256": canonical_digest(config),
        "evaluation_lock_sha256": config.get("evaluation_lock_sha256"),
        "model_revision": config.get("model_revision"),
        "alfworld_config_sha256": alfworld_config_sha256,
        "textcraft_recipe_order_sha256": textcraft_recipe_order_sha256,
        "git_commit": git_commit(),
        "python_version": platform.python_version(),
        "package_versions": package_versions(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    if config["condition"] != "baseline":
        metadata["support_policy_id"] = ("adapt_controller_v1" if config["condition"] == "adapt_dynamic_replanning_v1"
                                         else selected_policy.policy_id)
    if legacy_memory:
        legacy_corpus, legacy_index = artifacts["legacy"]["corpus"], artifacts["legacy"]["index"]
        metadata["memory_corpus_sha256"] = canonical_digest(legacy_corpus)
        metadata["memory_index_sha256"] = canonical_digest(legacy_index)
        metadata["legacy_memory_selection_policy"] = RETRIEVAL_POLICY
        if config["condition"] == "m2_always":
            metadata["memory_selection_policy"] = RETRIEVAL_POLICY
    if trajectory_memory:
        trajectory_corpus, trajectory_index = artifacts["trajectory"]["corpus"], artifacts["trajectory"]["index"]
        metadata["trajectory_corpus_sha256"] = canonical_digest(trajectory_corpus)
        metadata["trajectory_index_sha256"] = canonical_digest(trajectory_index)
        metadata["trajectory_memory_selection_policy"] = TRAJECTORY_POLICY
        if config["condition"] in TRAJECTORY_CONDITIONS:
            metadata["memory_selection_policy"] = TRAJECTORY_POLICY
    write_json_atomic(directory / "run.json", metadata)
    episodes = []
    try:
        actual_model = model if model is not None else make_model(config)
        if hasattr(actual_model, "torch"):
            metadata["gpu_name"] = actual_model.torch.cuda.get_device_name()
            metadata["torch_cuda_version"] = actual_model.torch.version.cuda
        resources = build_builtin_resources(config, actual_model, artifacts)
        support, trajectory_support = resources["legacy"], resources["trajectory"]
        for index, task in enumerate(selected):
            result = run_episode(manifest["benchmark"], task, config, actual_model,
                                 episode_factory, support=support,
                                 trajectory_support=trajectory_support,
                                 support_policy=make_support_policy(config, task["task_id"]),
                                 support_adapters=make_supports(config, actual_model, resources))
            filename = f"{index:04d}_{canonical_digest(task['task_id'])[:12]}.json"
            write_json_atomic(directory / "episodes" / filename, result)
            episodes.append(result)
            write_json_atomic(directory / "summary.json", summarize(episodes))
            print(f"{index + 1}/{len(selected)} {task['task_id']} {result['status']} actions={result['environment_actions']}", flush=True)
        metadata["status"] = "complete"
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["error"] = repr(exc)
        raise
    finally:
        metadata["completed_task_count"] = len(episodes)
        metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json_atomic(directory / "run.json", metadata)
    return directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--output-root")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config = read_json(args.config)
    manifest_path = args.manifest or config.get("manifest")
    output_root = args.output_root or config.get("output_root")
    if not manifest_path or not output_root:
        parser.error("--manifest and --output-root are required for legacy configs")
    manifest = read_json(manifest_path)
    output = run(config, manifest, output_root, args.limit)
    print(f"Run complete: {output}")


if __name__ == "__main__":
    main()
