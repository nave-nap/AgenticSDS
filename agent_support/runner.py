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

from .benchmarks import make_episode
from .common import TRAJECTORY_CONDITIONS, canonical_digest, read_json, validate_config, validate_manifest, write_json_atomic
from .model import make_model
from .policy import alfworld_messages, parse_alfworld, parse_textcraft, textcraft_messages
from .memory import ActionMemory, RETRIEVAL_POLICY, SentenceEmbedder, support_messages
from .controllers import run_controller
from .support_policy import (ACTION_TRAJECTORY, LEGACY_ACTION_MEMORY, RECONSIDERATION, TASK_TRAJECTORY,
                             SupportPolicy, TurnContext, make_support_policy, selected_supports)
from .supports import make_supports
from .trajectory_memory import MPNetEmbedder, POLICY as TRAJECTORY_POLICY, TrajectoryMemory


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
    started = time.perf_counter()
    history: list[dict] = []
    trace: list[dict] = []
    status = "max_decisions"
    actions_taken = 0
    parser_errors = 0
    failed_actions = 0
    input_tokens = output_tokens = 0
    inference_seconds = 0.0
    support_calls = support_model_calls = support_input_tokens = support_output_tokens = support_parse_errors = 0
    retrieval_seconds = 0.0
    success = False
    condition = config.get("condition", "m2_always" if support is not None else "baseline")
    policy = support_policy or make_support_policy(config)
    adapters = support_adapters or make_supports(config, model, {"legacy": support, "trajectory": trajectory_support})

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
            proposal_output=proposal_output)

    try:
        episode.reset()
        for decision in range(1, config["max_decisions"] + 1):
            if actions_taken >= config["max_actions"]:
                status = "max_actions"
                break
            if benchmark == "alfworld":
                if not episode.actions:
                    status = "no_actions"
                    break
                available = list(episode.actions)
                messages = alfworld_messages(episode.task_prompt, episode.observation, available, history, config["history_window"])
            else:
                available = None
                messages = textcraft_messages(episode.initial_observation, history, config["history_window"])
            before = selected_supports(policy, context("before_proposal", decision, available))
            before_events = []
            for choice in before:
                if choice not in adapters or adapters[choice].phase != "before_proposal":
                    raise ValueError(f"Policy selected unavailable or misphased support: {choice}")
                outcome = adapters[choice].invoke(context("before_proposal", decision, available), messages, None)
                messages = outcome.messages
                support_calls += outcome.calls
                retrieval_seconds += outcome.retrieval_seconds
                if outcome.event is not None:
                    before_events.append(outcome.event)
            generation = model.generate(messages)
            input_tokens += generation["input_tokens"]
            output_tokens += generation["output_tokens"]
            inference_seconds += generation["inference_seconds"]
            raw = generation["text"]
            action = parse_alfworld(raw, available) if benchmark == "alfworld" else parse_textcraft(raw)
            after = selected_supports(policy, context("after_proposal", decision, available, action, raw))
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
            if support is not None or trajectory_support is not None or before or after:
                record["support_policy"] = {"policy_id": policy.policy_id,
                                            "before_proposal": list(before),
                                            "after_proposal": list(after)}

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
                outcome = adapters[choice].invoke(context("after_proposal", decision, available, action, raw), messages, action)
                event = outcome.event
                revision_messages = outcome.revision_messages
                support_calls += outcome.calls
                retrieval_seconds += outcome.retrieval_seconds
                record.setdefault("proposed_action", proposed_action)
                record.setdefault("proposal_raw_model_output", raw)
                if event is not None:
                    add_support_event(event)
                if revision_messages is None:
                    if event is not None and outcome.fallback_reason:
                        event["fallback_reason"] = outcome.fallback_reason
                    continue
                revision = model.generate(revision_messages)
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
                revised = (parse_alfworld(revision["text"], available) if benchmark == "alfworld"
                           else parse_textcraft(revision["text"]))
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
                if benchmark == "alfworld":
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
            if benchmark == "alfworld" and len(history) >= 6:
                recent_actions = [item["action"] for item in history[-6:]]
                if recent_actions[:2] == recent_actions[2:4] == recent_actions[4:6]:
                    status = "repeated_action_loop"
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
    if condition != "baseline" or support is not None or trajectory_support is not None or support_calls:
        result.update({"model_calls": len(trace) + support_model_calls,
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
    selected_policy = make_support_policy(config)
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    selected = manifest["tasks"][:limit] if limit is not None else manifest["tasks"]
    legacy_memory = "memory_corpus" in config
    trajectory_memory = "trajectory_corpus" in config
    legacy_corpus = legacy_index = trajectory_corpus = trajectory_index = None

    def check_memory(corpus: dict, check_content: bool) -> None:
        if canonical_digest(manifest) not in corpus.get("excluded_manifest_sha256", {}).values():
            raise ValueError("Memory corpus was not built with this evaluation manifest excluded")
        selected_ids = {task["task_id"] for task in selected}
        if any(record["benchmark"] == manifest["benchmark"] and record["task_id"] in selected_ids
               for record in corpus["records"]):
            raise ValueError("Memory corpus contains evaluation task trajectories")
        if check_content:
            fingerprint = "game_sha256" if manifest["benchmark"] == "alfworld" else "observation_sha256"
            selected_hashes = {task[fingerprint] for task in selected}
            if any(record["benchmark"] == manifest["benchmark"]
                   and record["task_fingerprint"] in selected_hashes for record in corpus["records"]):
                raise ValueError("Trajectory corpus contains evaluation task content")

    if legacy_memory:
        legacy_corpus = read_json(config["memory_corpus"])
        legacy_index = read_json(config["memory_index"])
        check_memory(legacy_corpus, check_content=False)
    if trajectory_memory:
        trajectory_corpus = read_json(config["trajectory_corpus"])
        trajectory_index = read_json(config["trajectory_index"])
        check_memory(trajectory_corpus, check_content=True)
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
        metadata["memory_corpus_sha256"] = canonical_digest(legacy_corpus)
        metadata["memory_index_sha256"] = canonical_digest(legacy_index)
        metadata["legacy_memory_selection_policy"] = RETRIEVAL_POLICY
        if config["condition"] == "m2_always":
            metadata["memory_selection_policy"] = RETRIEVAL_POLICY
    if trajectory_memory:
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
        support = (ActionMemory(legacy_corpus, legacy_index,
                                SentenceEmbedder(legacy_index["embedding_model"],
                                                 revision=legacy_index.get("embedding_revision")),
                                config["memory_top_k"])
                   if legacy_memory else None)
        trajectory_support = None
        if trajectory_memory:
            tokenizer = getattr(actual_model, "tokenizer", None)
            token_counter = (actual_model.count_tokens if hasattr(actual_model, "count_tokens")
                             else (lambda value: len(tokenizer.encode(value, add_special_tokens=False))
                                   if tokenizer is not None else len(value.split())))
            trajectory_support = TrajectoryMemory(
                trajectory_corpus, trajectory_index,
                MPNetEmbedder(trajectory_index["embedding_model"],
                              revision=trajectory_index.get("embedding_revision")),
                config["trajectory_top_k"], config["trajectory_token_budget"], token_counter)
        for index, task in enumerate(selected):
            result = run_episode(manifest["benchmark"], task, config, actual_model,
                                 episode_factory, support=support,
                                 trajectory_support=trajectory_support,
                                 support_policy=make_support_policy(config))
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
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config = read_json(args.config)
    manifest = read_json(args.manifest)
    output = run(config, manifest, args.output_root, args.limit)
    print(f"Run complete: {output}")


if __name__ == "__main__":
    main()
